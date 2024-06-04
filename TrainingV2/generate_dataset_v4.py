from collections import Counter
import time
import cv2
from datasets import Dataset
import os
import glob
from datasets import load_from_disk, Array2D, concatenate_datasets
from datasets import DatasetDict
import mediapipe as mp
import numpy as np
from utils import mediapipe_detection, extract_keypoints_v3

mp_holistic = mp.solutions.holistic

DATASET_NAME = "fsl-143-v1"
DATASET_VERSION = "v2"
DATA_PATH = os.path.join(f"../Dataset/{DATASET_NAME}")
NUM_PROC = max(int(os.cpu_count() * 0.5), 1)
TARGET_FPS = 20
WITH_FLIP = True


def get_files(examples):
    new_examples = {
        "id": [],
        "label": [],
        "category": [],
        "file": [],
    }

    # Create new example for each file
    for idx, id in enumerate(examples["id"]):
        label_dir = os.path.join(DATA_PATH, f"clips/{id}/**/*.*")
        for file in glob.glob(label_dir, recursive=True):
            new_examples["id"].append(examples["id"][idx])
            new_examples["label"].append(examples["label"][idx])
            new_examples["category"].append(examples["category"][idx])
            new_examples["file"].append(file)

    return new_examples


def get_keypoints(examples, flip_horizontal=False):
    new_examples = {
        "id": [],
        "label": [],
        "category": [],
        "file": [],
        "pose": [],
        "face": [],
        "lh": [],
        "rh": [],
        "pose_mean": [],
        "face_mean": [],
        "lh_mean": [],
        "rh_mean": [],
        "pose_angles": [],
        "lh_angles": [],
        "rh_angles": [],
        "fps": [],
        "keypoints_length": [],
    }

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:
        for idx, file in enumerate(examples["file"]):
            # Clear mediapipe cache
            mediapipe_detection(np.zeros((360, 640, 3), dtype=np.uint8), holistic)

            cap = cv2.VideoCapture(file)
            example_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num_skip = int(example_fps // TARGET_FPS)
            frame_idx = 0

            example_keypoints = []
            frame_gesture_types = []

            while cap.isOpened():
                ret, frame = cap.read()
                frame_idx += 1

                # Discard frames other than the first split
                if frame_idx % frame_num_skip != 0:
                    continue
                if not ret:
                    break

                if flip_horizontal:
                    frame = cv2.flip(frame, 1)

                image, results = mediapipe_detection(frame, holistic)

                # Check if the frame has all the landmarks
                has_pose = True if results.pose_landmarks else False
                has_face = True if results.face_landmarks else False
                has_left_hand = True if results.left_hand_landmarks else False
                has_right_hand = True if results.right_hand_landmarks else False
                current_gesture = (
                    "gesture"
                    if all([has_pose, has_face, has_left_hand or has_right_hand])
                    else "idle"
                )
                frame_gesture_types.append(current_gesture)

                keypoints = extract_keypoints_v3(results)
                example_keypoints.append(keypoints)

            try:
                first_gesture_idx = frame_gesture_types.index("gesture")
                last_gesture_idx = len(frame_gesture_types) - frame_gesture_types[
                    ::-1
                ].index("gesture")
            except ValueError:
                print("Unable to find gesture in", file)
                first_gesture_idx = 0
                last_gesture_idx = len(frame_gesture_types)

            if first_gesture_idx > 0:
                idle_keypoints = example_keypoints[:first_gesture_idx]
                new_examples["id"].append(-1)
                new_examples["label"].append("IDLE")
                new_examples["category"].append("IDLE")
                new_examples["file"].append(file)
                new_examples["fps"].append(example_fps / frame_num_skip)
                new_examples["keypoints_length"].append(len(idle_keypoints))

                idle_keypoints_dict = {k: [] for k in idle_keypoints[0].keys()}
                for kp in idle_keypoints:
                    for key in kp.keys():
                        idle_keypoints_dict[key].append(kp[key])

                for key in idle_keypoints_dict.keys():
                    new_examples[key].append(idle_keypoints_dict[key])

            gesture_keypoints = example_keypoints[first_gesture_idx:last_gesture_idx]
            if len(gesture_keypoints) <= 10:
                print(f"Warning: {file} has less than 10 frames of gesture")

            new_examples["id"].append(examples["id"][idx])
            new_examples["label"].append(examples["label"][idx])
            new_examples["category"].append(examples["category"][idx])
            new_examples["file"].append(file)
            new_examples["fps"].append(example_fps / frame_num_skip)
            new_examples["keypoints_length"].append(len(gesture_keypoints))

            gesture_keypoints_dict = {k: [] for k in gesture_keypoints[0].keys()}
            for kp in gesture_keypoints:
                for key in kp.keys():
                    gesture_keypoints_dict[key].append(kp[key])

            for key in gesture_keypoints_dict.keys():
                new_examples[key].append(gesture_keypoints_dict[key])

    return new_examples


count_by_label = {}


def filter_max_count(examples, max_samples):
    result = []

    for label in examples["label"]:
        if label not in count_by_label:
            count_by_label[label] = 0

        if count_by_label[label] > max_samples:
            result.append(False)
        else:
            count_by_label[label] += 1
            result.append(True)

    return result


if __name__ == "__main__":
    LOAD_FROM_CACHE = False
    output_name = "-".join(
        [
            DATASET_NAME,
            DATASET_VERSION,
            f"{TARGET_FPS}fps",
            "with-flipped" if WITH_FLIP else "orig",
        ]
    )

    if LOAD_FROM_CACHE:
        os.path.join(
            DATA_PATH,
        )
        ds = load_from_disk(f"../datasets_cache/v4-{output_name}-raw")
    else:
        ds = Dataset.from_csv(
            f"../Dataset/{DATASET_NAME}/labels.csv",
            cache_dir="../datasets_cache",
        )

        ds = ds.map(get_files, batched=True, batch_size=100)
        ds_non_flipped = ds.map(
            get_keypoints,
            batched=True,
            batch_size=15,
            num_proc=NUM_PROC,
        )

        if WITH_FLIP:
            ds_flipped = ds.map(
                get_keypoints,
                batched=True,
                batch_size=15,
                fn_kwargs={"flip_horizontal": True},
                num_proc=NUM_PROC,
            )
            ds = concatenate_datasets([ds_non_flipped, ds_flipped])
        else:
            ds = ds_non_flipped

        ds = ds.cast_column("pose", Array2D(shape=(None, 33 * 4), dtype="float32"))
        ds = ds.cast_column("face", Array2D(shape=(None, 468 * 3), dtype="float32"))
        ds = ds.cast_column("lh", Array2D(shape=(None, 21 * 3), dtype="float32"))
        ds = ds.cast_column("rh", Array2D(shape=(None, 21 * 3), dtype="float32"))
        ds = ds.class_encode_column("label")
        ds = ds.class_encode_column("category")

        # Limit the number of samples per label to average label count
        label_ctr = Counter(ds["label"]).most_common()[1:]
        # label_ctr_avg = sum([x[1] for x in label_ctr]) // len(label_ctr)
        label_ctr_max = max([x[1] for x in label_ctr])
        ds = ds.filter(
            filter_max_count,
            batched=True,
            fn_kwargs={"max_samples": label_ctr_max * 2},
        )

        ds.save_to_disk(f"../datasets_cache/v4-{output_name}-raw")

    datasets = ds.train_test_split(
        test_size=0.20,
        seed=10293812098,
        stratify_by_column="label",
    )

    datasets = datasets.sort("keypoints_length")
    datasets = datasets.flatten()

    ds = DatasetDict(train=datasets["train"], test=datasets["test"])

    ds.save_to_disk(f"../datasets_cache/v4-{output_name}")
    print("Dataset saved to: ", f"v4-{output_name}")
