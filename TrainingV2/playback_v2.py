import cv2
import numpy as np
from datasets import load_from_disk, concatenate_datasets

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

    ds = load_from_disk("../datasets_cache/fsl-143-v1")
    ds = concatenate_datasets([ds["train"], ds["test"]])
    feature_label = ds.features["label"]
    gestures = ds.features["label"].names

    samples_by_gesture = {}

    for example in ds:
        gesture_name = feature_label.int2str(example["label"])
        if gesture_name not in samples_by_gesture:
            samples_by_gesture[gesture_name] = []
        samples_by_gesture[gesture_name].append(example)

    ds = None
    filtered_ds = samples_by_gesture[gestures[gesture_idx]]
    num_samples = len(filtered_ds)

    while running:
        if changed:
            filtered_ds = samples_by_gesture[gestures[gesture_idx]]
            num_samples = len(filtered_ds)
            sample_idx = 0
            changed = False
            print(
                f"Gesture: {gestures[gesture_idx]}, Sample: {sample_idx + 1}/{num_samples}"
            )

        sample = filtered_ds[sample_idx]
        last_key = play(sample)

        # NEXT SAMPLE = d
        if last_key == 100:
            sample_idx = (sample_idx + 1) % num_samples
            continue
        # PREVIOUS SAMPLE = a
        elif last_key == 97:
            print("bf", sample_idx)
            sample_idx = (sample_idx + num_samples - 1) % num_samples
            print("af", sample_idx)
            continue

        # NEXT GESTURE = w
        elif last_key == 119:
            gesture_idx = (gesture_idx + 1) % len(gestures)
            changed = True
            continue
        # PREVIOUS GESTURE = s
        elif last_key == 115:
            gesture_idx = (gesture_idx + len(gestures) - 1) % len(gestures)
            changed = True
            continue
        elif last_key == 32: # space
            running = False
            break
        # TODO: ADD LEFT RIGHT ARROW KEY TO NEXT
