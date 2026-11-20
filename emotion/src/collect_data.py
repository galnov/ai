"""
Collect your own data from the webcam - add pictures of yourself (and friends) to the training set.

    python -m src.collect_data

In the window that opens:
    1..7   save the current face as emotion:
           1=angry 2=disgust 3=fear 4=happy 5=neutral 6=sad 7=surprise
    space  "burst" mode: saves an image every 0.3 s for the last chosen emotion (space again to stop)
    q      quit

Images are saved to data/custom/<emotion>/ as 48x48 grayscale - exactly like FER-2013,
and the training script adds them automatically.
"""
import time

import cv2

from .config import DATA_DIR, EMOTIONS
from .vision import crop_face, detect_faces, expand_box

OUT = DATA_DIR / "custom"


def count_all():
    return {e: len(list((OUT / e).glob("*.png"))) if (OUT / e).exists() else 0 for e in EMOTIONS}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No camera found")
    counts = count_all()
    current, burst, last_save, flash = None, False, 0.0, 0
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, max_faces=1)
        box = expand_box(faces[0], gray.shape[1], gray.shape[0]) if faces else None

        key = cv2.waitKey(1) & 0xFF
        save_now = False
        if key == ord("q"):
            break
        if ord("1") <= key <= ord("7"):
            current, save_now = EMOTIONS[key - ord("1")], True
        if key == ord(" ") and current:
            burst = not burst
        if burst and time.time() - last_save > 0.3:
            save_now = True

        if save_now and box and current:
            (OUT / current).mkdir(parents=True, exist_ok=True)
            fname = OUT / current / f"{int(time.time() * 1000)}.png"
            cv2.imwrite(str(fname), crop_face(gray, box))
            counts[current] += 1
            last_save, flash = time.time(), 4

        view = frame.copy()
        if box:
            x, y, w, h = box
            color = (255, 255, 255) if flash > 0 else (80, 220, 120)
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 3 if flash > 0 else 2)
            flash -= 1
        y0 = 25
        for i, e in enumerate(EMOTIONS):
            mark = ">" if e == current else " "
            cv2.putText(view, f"{mark}{i+1} {e:9s} {counts[e]:4d}", (10, y0 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if e == current else (240, 240, 240), 2)
        status = "BURST ON" if burst else "keys 1-7 save | space burst | q quit"
        cv2.putText(view, status, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if burst else (240, 240, 240), 2)
        cv2.imshow("Collect emotion data", view)

    cap.release()
    cv2.destroyAllWindows()
    print("Saved:", count_all())


if __name__ == "__main__":
    main()
