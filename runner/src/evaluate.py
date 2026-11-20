"""
Final evaluation: every AI player on the same 50 tracks it never trained on (5 minutes at most per run),
compared with the human runs that were recorded in the app.

    python -m src.evaluate

Outputs in reports/:
    eval_report.txt        the comparison table, and how the humans did against the AI in the app
    eval_distances.png     distance per run for each player, and what each one crashed into
    eval_survival.png      what share of the runs are still going after X meters
    learning_curves.png    DQN vs. combo during training (does starting from human play save practice?)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .players import LABELS, available, make_player, play
from .recordings import load_recordings

COLORS = {"combo": "#7b2cbf", "dqn": "#2e86ab", "imitation": "#f18f01", "bot": "#6c757d", "humans": "#e4572e"}
CRASH_COLORS = {"hurdle": "#f18f01", "overhead bar": "#ffd23f", "train": "#2e86ab", "no crash (time limit)": "#adb5bd"}


def crash_counts(crash_types):
    return {**{C.OBSTACLE_NAMES[t]: int((crash_types == t).sum()) for t in (C.LOW, C.HIGH, C.TRAIN)},
            "no crash (time limit)": int((crash_types == 0).sum())}


def main():
    names = available()
    if names == ["bot"]:
        print("[!] No trained networks yet - only the rule-based bot will be evaluated.")
    results = {}
    for name in names:
        dist, crashes, ticks = play(make_player(name), C.TEST_SEEDS, max_ticks=C.EVAL_TICKS)
        results[name] = {"distance": dist, "crashes": crash_counts(crashes), "survived": float((crashes == 0).mean())}
        print(f"  {LABELS[name]:32s} mean {dist.mean():6.0f} m | median {np.median(dist):6.0f} m | "
              f"survived 5 min: {results[name]['survived'] * 100:3.0f}%")

    source = "human" if load_recordings("human") else "bot"
    recs = load_recordings(source)
    human_label = "humans" if source == "human" else "practice bot runs"
    if recs:
        hd = np.array([r["distance"] for r in recs])
        hc = [r.get("crash") or "no crash (time limit)" for r in recs]
        results["humans"] = {"distance": hd,
                             "crashes": {k: sum(c == k for c in hc) for k in CRASH_COLORS},
                             "survived": float(np.mean([c == "no crash (time limit)" for c in hc]))}

    # ---------------------------------------------------------------- report
    lines = [f"Evaluation: {len(C.TEST_SEEDS)} new tracks, at most {C.EVAL_TICKS * C.TICK_MS // 60000} minutes per run",
             "", f"{'player':34s}{'mean':>8s}{'median':>8s}{'best':>8s}{'no crash':>9s}   crashed into (hurdle/bar/train)"]
    for name, r in results.items():
        label = LABELS.get(name, human_label)
        cr = r["crashes"]
        lines.append(f"{label:34s}{r['distance'].mean():8.0f}{np.median(r['distance']):8.0f}{r['distance'].max():8.0f}"
                     f"{r['survived'] * 100:8.0f}%   {cr['hurdle']}/{cr['overhead bar']}/{cr['train']}")
    if recs:
        lines += ["", f"The {human_label} played other tracks ({len(recs)} runs), so compare their spread, not run by run."]
        h2h = [r for r in recs if r.get("ai")]
        if h2h:
            won = sum(r["distance"] > r["ai"]["distance"] for r in h2h)
            lines.append(f"Head to head in the app (same track as the AI ghost): {human_label} won {won} of {len(h2h)} races.")
            for model in sorted({r["ai"]["model"] for r in h2h}):
                rs = [r for r in h2h if r["ai"]["model"] == model]
                w = sum(r["distance"] > r["ai"]["distance"] for r in rs)
                lines.append(f"    vs {LABELS.get(model, model):32s} {w} of {len(rs)}")
    text = "\n".join(lines)
    print("\n" + text)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "eval_report.txt").write_text(text + "\n")

    # ---------------------------------------------------------------- distances + crash causes
    keys = list(results)
    labels = [LABELS.get(k, human_label).replace(" (", "\n(") for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        d = results[k]["distance"]
        ax[0].scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=14, alpha=0.6, color=COLORS[k])
        ax[0].hlines(np.median(d), i - 0.3, i + 0.3, color="black", lw=2)
    ax[0].set_xticks(range(len(keys)), labels, fontsize=9)
    ax[0].set_ylabel("distance (m)")
    ax[0].set_title("Distance per run (black line = median)")
    ax[0].grid(axis="y", alpha=0.3)
    bottom = np.zeros(len(keys))
    for cause, color in CRASH_COLORS.items():
        vals = np.array([results[k]["crashes"][cause] / max(1, len(results[k]["distance"])) * 100 for k in keys])
        ax[1].bar(range(len(keys)), vals, bottom=bottom, color=color, label=cause)
        bottom += vals
    ax[1].set_xticks(range(len(keys)), labels, fontsize=9)
    ax[1].set_ylabel("% of runs")
    ax[1].set_title("How each run ended")
    ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "eval_distances.png", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- survival curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    top = max(r["distance"].max() for r in results.values())
    xs = np.linspace(0, top, 300)
    for k in keys:
        d = results[k]["distance"]
        ax.plot(xs, [(d >= x).mean() * 100 for x in xs], color=COLORS[k], lw=2, label=LABELS.get(k, human_label))
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("% of runs still going")
    ax.set_title("Survival: how far do the runs get?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "eval_survival.png", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- learning curves
    curves = {n: C.REPORTS_DIR / f"{n}_history.json" for n in ("dqn", "combo")}
    curves = {n: json.loads(p.read_text()) for n, p in curves.items() if p.exists()}
    if curves:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for n, hist in curves.items():
            ax.plot([h["step"] * C.DQN["n_envs"] / 1e6 for h in hist], [h["val_distance"] for h in hist],
                    marker="o", ms=3, color=COLORS[n], label=LABELS[n])
        bc = C.REPORTS_DIR / "imitation_history.json"
        if bc.exists():
            best = max(h["val_distance"] for h in json.loads(bc.read_text()))
            ax.axhline(best, color=COLORS["imitation"], ls="--", label="Imitation (best)")
        ax.set_xlabel("experience in the game (millions of ticks)")
        ax.set_ylabel("distance on the validation tracks (m, 2 min)")
        ax.set_title("Learning curves: does starting from human play save practice?")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(C.REPORTS_DIR / "learning_curves.png", dpi=110)
        plt.close(fig)
    print(f"\nReports saved to {C.REPORTS_DIR}.  Next step:  python -m app.server")


if __name__ == "__main__":
    main()
