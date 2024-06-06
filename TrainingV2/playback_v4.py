import cv2
import torch
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from augmentations.augmentation_v2 import AugmentationV2, Transform, Flip, Scale, Rotate
from tqdm import tqdm

DATASET_NAME = "v4-fsl-143-v1-v2-20fps-orig"
WIDTH = 640
HEIGHT = 360


def play(sample, augmentation=None):
    for frame_idx in range(len(sample["pose"])):
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        pose = sample["pose"][frame_idx]
        face = sample["face"][frame_idx]
        lh = sample["lh"][frame_idx]
        rh = sample["rh"][frame_idx]

        if augmentation is not None:
            pose, face, lh, rh = augmentation(
                torch.tensor(pose),
                torch.tensor(face),
                torch.tensor(lh),
                torch.tensor(rh),
            )
            pose = pose.numpy()
            face = face.numpy()
            lh = lh.numpy()
            rh = rh.numpy()

        pose_mean = sample["pose_mean"][frame_idx]
        face_mean = sample["face_mean"][frame_idx]
        lh_mean = sample["lh_mean"][frame_idx]
        rh_mean = sample["rh_mean"][frame_idx]

        # Face Playback
        # for idx in range(0, len(face), 3):
        #     x = face[idx] + face_mean[0]
        #     y = face[idx + 1] + face_mean[1]
        #     z = face[idx + 2] + face_mean[2]
        #     center = (int(x * WIDTH), int(y * HEIGHT))

        #     cv2.circle(image, center, 3, (255, 0, 0))

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
    augment_idx = 0
    changed = True

    augmentations = [
        None,
        Rotate(15, offset=[0, 0, 0]),
        Flip(0.5, 0, (0, 0)),
        Scale((0.5, 0, 0), offset=[0, 0, 0]),
        Transform((0.5, 0.5, 0)),
    ]

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
                f"Gesture: {gestures[gesture_idx]}",
                f"Sample: {sample_idx + 1}/{num_samples}",
                f"Frames: {len(filtered_ds[sample_idx]['pose'])}",
                f"File: {filtered_ds[sample_idx]['file']}",
            )

        sample = filtered_ds[sample_idx]
        last_key = play(sample, augmentations[augment_idx])

        # NEXT SAMPLE = d
        if last_key == 100:
            sample_idx = (sample_idx + 1) % num_samples
            print(
                f"Sample: {sample_idx + 1}/{num_samples}",
                f"File: {filtered_ds[sample_idx]['file']}",
            )
            continue
        # PREVIOUS SAMPLE = a
        elif last_key == 97:
            sample_idx = (sample_idx + num_samples - 1) % num_samples
            print(
                f"Sample: {sample_idx + 1}/{num_samples}",
                f"Frames: {len(filtered_ds[sample_idx]['pose'])}",
                f"File: {filtered_ds[sample_idx]['file']}",
            )
            continue

        elif last_key == 116:  # t
            augment_idx = (augment_idx + 1) % len(augmentations)
            if augmentations[augment_idx] is not None:
                augmentations[augment_idx].generate_vars()
            print(f"Augmentation: {augmentations[augment_idx]}")
            continue

        elif last_key == 114:  # r
            if augmentations[augment_idx] is not None:
                augmentations[augment_idx].generate_vars()
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
