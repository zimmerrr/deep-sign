from collections import Counter
import time
import cv2
from datasets import Dataset
import os
import glob
from datasets import load_from_disk, Array2D
from datasets import DatasetDict
import mediapipe as mp
from utils import mediapipe_detection, extract_keypoints

mp_holistic = mp.solutions.holistic

DATASET_NAME = "fsl-105"
DATASET_VERSION = "v2"
DATA_PATH = os.path.join(f"../Dataset/{DATASET_NAME}")
NUM_PROC = max(int(os.cpu_count() * 0.45), 1)


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


def get_keypoints(examples):
    new_examples = {
        "id": [],
        "label": [],
        "category": [],
        "file": [],
        "keypoints": [],
        "fps": [],
    }

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for idx, file in enumerate(examples["file"]):
            cap = cv2.VideoCapture(file)
            example_fps = cap.get(cv2.CAP_PROP_FPS)
            num_split = int(example_fps // 30)
            example_fps /= num_split

            example_keypoints = {}
            frame_idx = 0
            split_idx = 0
            last_gesture = None

            while cap.isOpened():
                ret, frame = cap.read()
                frame_idx += 1

                # Discard frames other than the first split
                if frame_idx % num_split != 0:
                    continue
                if not ret:
                    break

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

                # Split the keypoints by gesture or idle
                if last_gesture != None and last_gesture != current_gesture:
                    split_idx += 1
                last_gesture = current_gesture

                if split_idx not in example_keypoints:
                    example_keypoints[split_idx] = (current_gesture, [])

                example_keypoints[split_idx][1].append(extract_keypoints(results))

            for gesture, keypoints in example_keypoints.values():
                # Discard keypoints with less than 10 frames
                if len(keypoints) <= 10:
                    continue

                if gesture == "idle":
                    new_examples["id"].append(-1)
                    new_examples["label"].append("IDLE")
                    new_examples["category"].append("IDLE")
                else:
                    new_examples["id"].append(examples["id"][idx])
                    new_examples["label"].append(examples["label"][idx])
                    new_examples["category"].append(examples["category"][idx])
                new_examples["file"].append(file)
                new_examples["keypoints"].append(keypoints)
                new_examples["fps"].append(example_fps)

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


def compute_keypoints_length(examples):
    lengths = []

    for keypoint in examples["keypoints"]:
        lengths.append(len(keypoint))

    return {"keypoints_length": lengths}


if __name__ == "__main__":
    LOAD_FROM_CACHE = False
    output_name = "-".join([DATASET_NAME, DATASET_VERSION])

    if LOAD_FROM_CACHE:
        os.path.join(
            DATA_PATH,
        )
        ds = load_from_disk(f"../datasets_cache/{output_name}-raw")
    else:
        ds = Dataset.from_csv(
            f"../Dataset/{DATASET_NAME}/labels.csv",
            cache_dir="../datasets_cache",
        )

        # PROCESSING
        ds = ds.map(get_files, batched=True, batch_size=100)
        ds = ds.map(
            get_keypoints,
            batched=True,
            batch_size=15,
            num_proc=NUM_PROC,
        )
        ds = ds.cast_column("keypoints", Array2D(shape=(None, 1662), dtype="float32"))
        ds = ds.map(compute_keypoints_length, batched=True, num_proc=4)
        ds = ds.class_encode_column("label")
        ds = ds.class_encode_column("category")

        # Limit the number of samples per label to average label count
        label_ctr = Counter(ds["label"]).most_common()
        label_ctr_avg = sum([x[1] for x in label_ctr]) // len(label_ctr)
        ds = ds.filter(
            filter_max_count,
            batched=True,
            fn_kwargs={"max_samples": label_ctr_avg},
        )

        ds.save_to_disk(f"../datasets_cache/{output_name}-raw")

    datasets = ds.train_test_split(
        test_size=0.10,
        seed=10293812098,
        stratify_by_column="label",
    )

    datasets = datasets.sort("keypoints_length")
    datasets = datasets.flatten()

    ds = DatasetDict(train=datasets["train"], test=datasets["test"])

    ds.save_to_disk(f"../datasets_cache/{output_name}")
