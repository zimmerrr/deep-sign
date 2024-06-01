import cv2
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm

DATASET_NAME = "v3-fsl-105-v3"
WIDTH = 640
HEIGHT = 360


def play(sample):
    for frame_idx in range(len(sample["pose"])):
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        pose = sample["pose"][frame_idx]
        face = sample["face"][frame_idx]
        lh = sample["lh"][frame_idx]
        rh = sample["rh"][frame_idx]

        pose_mean = sample["pose_mean"][frame_idx]
        face_mean = sample["face_mean"][frame_idx]
        lh_mean = sample["lh_mean"][frame_idx]
        rh_mean = sample["rh_mean"][frame_idx]

        # Face Playback
        for idx in range(0, len(face), 3):
            x = face[idx] + face_mean[0]
            y = face[idx + 1] + face_mean[1]
            z = face[idx + 2] + face_mean[2]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 3, (255, 0, 0))

        # Pose Playback
        for idx in range(0, len(pose), 4):
            x = pose[idx] + pose_mean[0]
            y = pose[idx + 1] + pose_mean[1]
            z = pose[idx + 2] + pose_mean[2]
            visibility = pose[idx + 3]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 3, (255, 255, 255))

        # Left Hand Playback
        for idx in range(0, len(lh), 3):
            x = lh[idx] + lh_mean[0]
            y = lh[idx + 1] + lh_mean[1]
            z = lh[idx + 2] + lh_mean[2]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 3, (0, 0, 255))

        # Right Hand Playback
        for idx in range(0, len(rh), 3):
            x = rh[idx] + rh_mean[0]
            y = rh[idx + 1] + rh_mean[1]
            z = rh[idx + 2] + rh_mean[2]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 3, (0, 255, 0))

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

    ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}")
    ds = concatenate_datasets([ds["train"], ds["test"]])
    feature_label = ds.features["label"]
    gestures = ds.features["label"].names

    samples_by_gesture = {}

    for example in tqdm(ds, "Loading dataset"):
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
                f"Gesture: {gestures[gesture_idx]}, "
                f"Sample: {sample_idx + 1}/{num_samples}, "
                f"Frames: {len(filtered_ds[sample_idx]['pose'])}"
            )

        sample = filtered_ds[sample_idx]
        last_key = play(sample)

        # NEXT SAMPLE = d
        if last_key == 100:
            sample_idx = (sample_idx + 1) % num_samples
            print(f"Sample: {sample_idx + 1}/{num_samples}, ")
            continue
        # PREVIOUS SAMPLE = a
        elif last_key == 97:
            sample_idx = (sample_idx + num_samples - 1) % num_samples
            print(f"Sample: {sample_idx + 1}/{num_samples}, ")
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
        elif last_key == 32:  # space
            running = False
            break
        # TODO: ADD LEFT RIGHT ARROW KEY TO NEXT
