import time
import cv2
from datasets import Dataset
import os
import numpy as np
import glob
import tqdm
from datasets import load_from_disk
from datasets import DatasetDict
import mediapipe as mp
from utils import mediapipe_detection, extract_keypoints

mp_holistic = mp.solutions.holistic

DATASET_NAME = "fsl-143-v1"
DATA_PATH = os.path.join(f"../Dataset/{DATASET_NAME}")
NUM_PROC = max(int(os.cpu_count() * 0.2), 1)  # Use all available processors except 2


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
            while cap.isOpened():
                ret, frame = cap.read()
                split_idx = frame_idx % num_split
                frame_idx += 1
                if not ret:
                    break

                image, results = mediapipe_detection(frame, holistic)

                if split_idx not in example_keypoints:
                    example_keypoints[split_idx] = []
                example_keypoints[split_idx].append(extract_keypoints(results))

            for keypoint in example_keypoints.values():
                new_examples["id"].append(examples["id"][idx])
                new_examples["label"].append(examples["label"][idx])
                new_examples["category"].append(examples["category"][idx])
                new_examples["file"].append(file)
                new_examples["keypoints"].append(keypoint)
                new_examples["fps"].append(example_fps)

    return new_examples


if __name__ == "__main__":
    LOAD_FROM_CACHE = False

    if LOAD_FROM_CACHE:
        os.path.join(DATA_PATH, )
        ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}-raw")
    else:
        ds = Dataset.from_csv(
            f"../Dataset/{DATASET_NAME}/labels.csv",
            cache_dir="../datasets_cache",
        )

        # PROCESSING
        ds = ds.map(get_files, batched=True, batch_size=100)
        ds = ds.class_encode_column("label")
        ds = ds.class_encode_column("category")
        ds = ds.map(
            get_keypoints,
            batched=True,
            batch_size=15,
            num_proc=NUM_PROC,
        )

        ds.save_to_disk(f"../datasets_cache/{DATASET_NAME}-raw")

    datasets = ds.train_test_split(
        test_size=0.10,
        shuffle=True,
        seed=10293812098,
        stratify_by_column="label",
    )

    ds = DatasetDict(train=datasets["train"], test=datasets["test"])

    ds.save_to_disk(f"../datasets_cache/{DATASET_NAME}")
