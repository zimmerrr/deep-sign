import cv2
import os
import glob
import numpy as np
from time import sleep
from datasets import load_from_disk

WIDTH = 640
HEIGHT = 360


def play(sample):
    for data in sample["keypoints"]:
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        # Reverse exteract keypoints
        pose = data[0:132]
        face = data[132 : 1404 + 132]
        rh = data[1404 + 132 : 63 + 1404 + 132]
        lh = data[63 + 1404 + 132 : 63 + 1404 + 132 + 63 + 1]

        # Pose Playback
        for idx in range(0, len(pose), 4):
            x = pose[idx]
            y = pose[idx + 1]
            z = pose[idx + 2]
            visibility = pose[idx + 3]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 5, (255, 255, 255))

        # Face Playback
        for idx in range(0, len(face), 3):
            x = face[idx]
            y = face[idx + 1]
            z = face[idx + 2]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 5, (255, 0, 0))

        # Right Hand Playback
        for idx in range(0, len(rh), 3):
            x = rh[idx]
            y = rh[idx + 1]
            z = rh[idx + 2]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 5, (0, 255, 0))

        # Left Hand Playback
        for idx in range(0, len(lh), 3):
            x = lh[idx]
            y = lh[idx + 1]
            z = lh[idx + 2]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 5, (0, 0, 255))
        cv2.imshow("Deep Sign Playback", image)

        last_key = cv2.waitKey(int(1000 // sample["fps"]))
        if last_key != -1:
            break

    return last_key


if __name__ == "__main__":
    running = True
    sample_idx = 0
    gesture_idx = 0
    changed = True

    ds = load_from_disk("../datasets_cache/fsl-105-raw")
    gestures = ds.features["label"].names
    filtered_ds = list(filter(lambda x: x["label"] == gesture_idx, ds))
    num_samples = len(filtered_ds)

    while running:
        if changed:
            filtered_ds = list(filter(lambda x: x["label"] == gesture_idx, ds))
            num_samples = len(filtered_ds)
            sample_idx = 0
            changed = False
            print(
                f"Gesture: {gestures[gesture_idx]}, Sample: {sample_idx + 1}/{num_samples}"
            )

        sample = filtered_ds[sample_idx]
        last_key = play(sample)

        # NEXT SAMPLE
        if last_key == 100:
            sample_idx = (sample_idx + 1) % num_samples
            changed = True
            continue
        # PREVIOUS SAMPLE
        elif last_key == 97:
            sample_idx = (sample_idx + num_samples - 1) % num_samples
            changed = True
            continue

        # NEXT GESTURE
        elif last_key == 119:
            gesture_idx = (gesture_idx + len(gestures) + 1) % len(gestures)
            changed = True
            continue
        # PREVIOUS GESTURE
        elif last_key == 155:
            gesture_idx = (gesture_idx + len(gestures) - 1) % len(gestures)
            changed = True
            continue
        elif last_key == 32:
            running = False
            break
        # TODO: ADD LEFT RIGHT ARROW KEY TO NEXT
