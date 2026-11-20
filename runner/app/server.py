"""
The web app: race against the AI in a 3D endless runner.

    python -m app.server
    then open in the browser:  http://localhost:5000

The game itself runs in the browser (app/static/game.js), so the controls respond instantly.
The server does three things:
  * runs the AI "ghost" on the same track (same seed) and sends its key presses in chunks
  * saves every human run as a recording (data/recordings/) - the training data for imitation learning
  * checks every recording by replaying it in the Python version of the game
"""
import argparse
import threading
import uuid

from flask import Flask, jsonify, render_template, request

from src import config as C
from src.game import RunnerGame
from src.players import LABELS, available, make_player
from src.recordings import CRASH_NAMES, load_recordings, presses_to_actions, replay, save_recording, summary

CHUNK = 200            # ticks of AI play per request (10 seconds)
MAX_SESSIONS = 30


def load_players():
    players, errors = {}, []
    for name in available():
        try:
            players[name] = make_player(name)
        except Exception as e:  # e.g. PyTorch is not installed
            errors.append(f"{LABELS[name]}: {e}")
    return players, errors


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    players, errors = load_players()
    sessions, lock = {}, threading.Lock()

    def message():
        if errors:
            return "Could not load a network: " + "; ".join(errors)
        if list(players) == ["bot"]:
            return ("There are no trained networks yet, so the AI ghost is the rule-based bot. "
                    "Play a few runs (they are recorded), then train: python -m src.train_bc")
        return None

    def run_ghost(s, n_ticks, keep=True):
        g, player = s["game"], players[s["model"]]
        actions, scores = [], []
        for _ in range(n_ticks):
            if not g.alive[0] or g.t[0] >= C.GHOST_TICKS:
                break
            a, sc = player.act(g)
            g.step(a)
            if keep:
                actions.append(int(a[0]))
                scores.append(None if sc is None else [round(float(x), 3) for x in sc[0]])
        return actions, scores

    def ghost_state(s):
        g = s["game"]
        over = (not g.alive[0]) or g.t[0] >= C.GHOST_TICKS
        return {"ticks": int(g.t[0]), "over": bool(over), "distance": float(g.pz[0]) / 100,
                "crash": CRASH_NAMES[int(g.crash_type[0])]}

    def leaderboard():
        recs = load_recordings("human")
        top = sorted(recs, key=lambda r: -r["distance"])[:10]
        h2h = [r for r in recs if r.get("ai")]
        return {
            "top": [{"player": r.get("player") or "anonymous", "distance": r["distance"],
                     "ai": r.get("ai")} for r in top],
            "summary": summary(recs),
            "head_to_head": {"races": len(h2h), "human_wins": sum(r["distance"] > r["ai"]["distance"] for r in h2h)},
            "bot_recordings": len(load_recordings("bot")),
        }

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/info")
    def info():
        order = [n for n in ("combo", "dqn", "imitation", "bot") if n in players]
        return jsonify({
            "players": [{"id": n, "label": LABELS[n],
                         "kind": "bot" if n == "bot" else players[n].kind} for n in order],
            "message": message(),
            "leaderboard": leaderboard(),
        })

    @app.post("/api/ghost/start")
    def ghost_start():
        d = request.get_json(silent=True) or {}
        model = d.get("model") if d.get("model") in players else next(iter(players))
        seed = int(d.get("seed", 0)) & 0x7FFFFFFF
        g = RunnerGame(1)
        g.reset([seed])
        sid = uuid.uuid4().hex
        s = {"game": g, "model": model, "lock": threading.Lock()}
        with lock:
            sessions[sid] = s
            while len(sessions) > MAX_SESSIONS:
                sessions.pop(next(iter(sessions)))
        with s["lock"]:
            actions, scores = run_ghost(s, CHUNK)
            return jsonify({"sid": sid, "model": model, "actions": actions, "scores": scores, **ghost_state(s)})

    @app.post("/api/ghost/more")
    def ghost_more():
        s = sessions.get((request.get_json(silent=True) or {}).get("sid"))
        if s is None:
            return jsonify({"error": "unknown session"}), 404
        with s["lock"]:
            actions, scores = run_ghost(s, CHUNK)
            return jsonify({"actions": actions, "scores": scores, **ghost_state(s)})

    @app.post("/api/ghost/finish")
    def ghost_finish():
        """The human has crashed: let the AI play to the end (without sending every tick)."""
        s = sessions.get((request.get_json(silent=True) or {}).get("sid"))
        if s is None:
            return jsonify({"error": "unknown session"}), 404
        with s["lock"]:
            run_ghost(s, C.GHOST_TICKS, keep=False)
            return jsonify(ghost_state(s))

    @app.post("/api/recording")
    def recording():
        d = request.get_json(silent=True) or {}
        try:
            seed, ticks = int(d["seed"]), int(d["ticks"])
            presses = [[int(t), int(a)] for t, a in d.get("presses", []) if 0 <= int(a) < C.N_ACTIONS]
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "bad recording"}), 400
        if not 0 < ticks <= 100_000:
            return jsonify({"error": "bad recording"}), 400
        # replay it in Python: this both checks the run and confirms the two game versions agree
        dist, crash, played = replay(seed, presses_to_actions(presses, ticks))
        if abs(dist - float(d.get("distance", -1))) > 0.011 or played != ticks:
            return jsonify({"error": f"the replay did not match ({dist:.1f} m in Python)", "saved": False}), 409
        rec = {"player": str(d.get("player") or "anonymous")[:24], "source": "human", "seed": seed,
               "ticks": ticks, "distance": dist, "crash": CRASH_NAMES[crash], "presses": presses}
        if isinstance(d.get("ai"), dict) and d["ai"].get("model") in LABELS:
            rec["ai"] = {"model": d["ai"]["model"], "distance": float(d["ai"].get("distance", 0))}
        path = save_recording(rec)
        return jsonify({"saved": True, "file": path.name, "leaderboard": leaderboard()})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    app = create_app()
    print(f"\n  Open in the browser:  http://localhost:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
