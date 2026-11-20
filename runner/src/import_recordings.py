"""
Adds the runs recorded on the web page (GitHub Pages) to the training data.

    python -m src.import_recordings my_runs.json

On the web page your runs are kept in the browser; "Download my runs" saves them to my_runs.json.
Every run is checked by replaying it in the Python version of the game (exactly like the Flask server
does), then saved to data/recordings/ - where train_bc and train_dqn --combo find it.
Runs that are already there (same seed, same key presses) are skipped, so importing twice is safe.
"""
import argparse
import json
from pathlib import Path

from . import config as C
from .recordings import CRASH_NAMES, load_recordings, presses_to_actions, replay, save_recording


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="the file from the web page, e.g. my_runs.json")
    args = ap.parse_args()
    data = json.loads(Path(args.file).read_text())
    if data.get("format") != "runner-recordings-v1":
        raise SystemExit("This is not a file from the web page's 'Download my runs' button")
    have = {(r["seed"], json.dumps(r.get("presses", []))) for r in load_recordings("all")}
    added = skipped = bad = 0
    for r in data["recordings"]:
        try:
            seed, ticks = int(r["seed"]), int(r["ticks"])
            presses = [[int(t), int(a)] for t, a in r.get("presses", []) if 0 <= int(a) < C.N_ACTIONS]
        except (KeyError, TypeError, ValueError):
            bad += 1
            continue
        if (seed, json.dumps(presses)) in have:
            skipped += 1
            continue
        dist, crash, played = replay(seed, presses_to_actions(presses, ticks))
        if abs(dist - float(r.get("distance", -1))) > 0.011 or played != ticks:
            bad += 1
            continue
        rec = {"player": str(r.get("player") or "anonymous")[:24], "source": "human", "seed": seed, "ticks": ticks,
               "distance": dist, "crash": CRASH_NAMES[crash], "presses": presses, "date": r.get("date")}
        if isinstance(r.get("ai"), dict):
            rec["ai"] = r["ai"]
        save_recording(rec)
        have.add((seed, json.dumps(presses)))
        added += 1
    print(f"Added {added} run(s) to {C.RECORDINGS_DIR}" + (f", skipped {skipped} already there" if skipped else "")
          + (f", {bad} did not replay correctly and were not added" if bad else ""))


if __name__ == "__main__":
    main()
