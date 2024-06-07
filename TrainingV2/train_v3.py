import time
from model.deepsign_v3 import DeepSignV3, DeepSignConfigV3
from augmentations.augmentation_v2 import AugmentationV2, Transform, Flip, Scale, Rotate
from utils import get_hand_angles, get_pose_angles
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
import wandb
import os
import numpy as np

os.environ["WANDB_API_KEY"] = "7204bcf714a2bd747d4d973bc999fbc86df91649"

torch.manual_seed(182731928)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "v4-fsl-143-v1-v2-20fps-with-flipped"
NUM_EPOCH = 2000
LEARNING_RATE = 0.001
GESTURE_LENGTH = 4
BATCH_SIZE = 32
MINIBATCH_SIZE = 32
# SEQUENCE_LENGTH = 30
ENABLE_AUGMENTATION = False
NUM_SEQ_ACCURACY_CHECK = 1

# Make sure the batch size is divisible by the mini-batch size
assert BATCH_SIZE / MINIBATCH_SIZE == BATCH_SIZE // MINIBATCH_SIZE


@torch.no_grad()
def get_loss_and_accuracy(model, dl):
    model.eval()
    total_accuracy = []
    total_loss = []
    for sample in dl:
        input_pose = sample["pose"].to(DEVICE)
        # input_face = sample["face"].to(DEVICE)
        input_lh = sample["lh"].to(DEVICE)
        input_rh = sample["rh"].to(DEVICE)
        input_pose_angles = sample["pose_angles"].to(DEVICE)
        input_lh_angles = sample["lh_angles"].to(DEVICE)
        input_rh_angles = sample["rh_angles"].to(DEVICE)
        input_pose_mean = sample["pose_mean"].to(DEVICE)
        input_lh_mean = sample["lh_mean"].to(DEVICE)
        input_rh_mean = sample["rh_mean"].to(DEVICE)
        input = torch.cat(
            [
                input_pose,
                input_pose_mean,
                input_pose_angles,
                input_lh,
                input_lh_mean,
                input_lh_angles,
                input_rh,
                input_rh_mean,
                input_rh_angles,
            ],
            dim=-1,
        )
        label = sample["label"].to(DEVICE)

        output, loss = model(input, target=label)
        total_loss.append(loss.item())

        output = F.softmax(output, dim=-1)
        output = output.argmax(-1)

        # only compare the last NUM_SEQ_ACCURACY_CHECK sequences
        output = output[:, -NUM_SEQ_ACCURACY_CHECK:]
        label = label[:, -NUM_SEQ_ACCURACY_CHECK:]

        total_correct = torch.sum(output == label)
        total_samples = label.numel()
        accuracy = total_correct / total_samples
        total_accuracy.append(accuracy.item())

    avg_loss = sum(total_loss) / len(total_loss)
    avg_accuracy = sum(total_accuracy) / len(total_accuracy)
    return avg_loss, avg_accuracy


# Jitter is applied as [-jitter / 2, jitter / 2]
augmentation = AugmentationV2(
    [
        Rotate(10),
        Flip(0.5, 0.2, (0, 0)),
        Scale((0.5, 0.5, 0.25)),
        Transform((0.5, 0.5, 0.25)),
    ]
)

fields_to_pad = [
    "pose",
    "face",
    "lh",
    "rh",
    "pose_mean",
    "face_mean",
    "lh_mean",
    "rh_mean",
    "pose_angles",
    "lh_angles",
    "rh_angles",
    "label",
]


def concat_gestures_transform(examples):
    new_examples = {k: [] for k in fields_to_pad}
    new_examples["keypoints_length"] = []
    new_examples["label"] = []
    new_examples["pose_angles"] = []
    new_examples["lh_angles"] = []
    new_examples["rh_angles"] = []

    for sample_idx in range(0, len(examples["keypoints_length"]), GESTURE_LENGTH):
        keypoints_length = 0
        label = []

        for field in fields_to_pad:
            if field == "label":
                continue

            field_value = []
            for i in range(GESTURE_LENGTH):
                field_value.extend(examples[field][sample_idx + i])
            new_examples[field].append(field_value)

        for i in range(GESTURE_LENGTH):
            current_length = examples["keypoints_length"][sample_idx + i]
            keypoints_length += current_length
            label.extend([examples["label"][sample_idx + i]] * current_length)

        new_examples["keypoints_length"].append(keypoints_length)
        new_examples["label"].append(label)

    return new_examples


# Make sure all examples have the same number of keypoints
# Pad the keypoints with the last frame until it reaches the max length
def pad_transform(examples):
    examples = concat_gestures_transform(examples)

    max_len = max(examples["keypoints_length"])
    examples["pad_count"] = []

    for sample_idx in range(len(examples["keypoints_length"])):
        curr_len = examples["keypoints_length"][sample_idx]
        missing = max_len - curr_len
        # start_idx = 0

        pad_count = max(missing, 0)
        examples["pad_count"].append(pad_count)

        # if missing < 0:
        #     max_idx = curr_len - max_len
        #     start_idx = torch.randint(0, max_idx, (1,)).item()

        for field in fields_to_pad:
            field_value = torch.tensor(examples[field][sample_idx])

            if missing > 0:
                if field == "label":
                    field_value = torch.concat(
                        [torch.empty(missing, dtype=torch.int).fill_(-100), field_value]
                    )
                else:
                    field_value = torch.concat(
                        [torch.zeros(missing, len(field_value[0])), field_value]
                    )
            # elif missing < 0:
            #     field_value = field_value[start_idx : start_idx + max_len]

            examples[field][sample_idx] = field_value

        # label = torch.tensor([examples["label"][sample_idx]]).repeat(5)
        # examples["label"][sample_idx] = label

    return examples


def augment_and_pad_transform(examples):
    examples = pad_transform(examples)

    for sample_idx in range(len(examples["keypoints_length"])):
        pose = examples["pose"][sample_idx]
        face = examples["face"][sample_idx]
        lh = examples["lh"][sample_idx]
        rh = examples["rh"][sample_idx]
        pad_count = examples["pad_count"][sample_idx]

        pose[pad_count:], face[pad_count:], lh[pad_count:], rh[pad_count:] = (
            augmentation(
                pose[pad_count:],
                face[pad_count:],
                lh[pad_count:],
                rh[pad_count:],
            )
        )

        examples["pose"][sample_idx] = pose
        examples["face"][sample_idx] = face
        examples["lh"][sample_idx] = lh
        examples["rh"][sample_idx] = rh

    return examples


def compute_angles(examples):
    pose_angles = []
    lh_angles = []
    rh_angles = []

    for pose in examples["pose"]:
        pose_angles.append(np.array([get_pose_angles(kp) for kp in pose]))

    for lh in examples["lh"]:
        lh_angles.append(np.array([get_hand_angles(kp) for kp in lh]))

    for rh in examples["rh"]:
        rh_angles.append(np.array([get_hand_angles(kp) for kp in rh]))

    return dict(
        pose_angles=pose_angles,
        lh_angles=lh_angles,
        rh_angles=rh_angles,
    )


if __name__ == "__main__":
    ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}")
    ds = ds.map(compute_angles, batched=True)
    label_feature = ds["train"].features["label"]
    # idle_idx = label_feature.str2int("IDLE")
    # ds = ds.filter(lambda example: example["label"] != idle_idx)

    print(ds)

    if ENABLE_AUGMENTATION:
        print(augmentation)

    ds = ds.with_format("torch")
    ds["train"].set_transform(
        augment_and_pad_transform if ENABLE_AUGMENTATION else pad_transform
    )
    ds["test"].set_transform(pad_transform)

    model_config = DeepSignConfigV3(
        input_size=(33 * 4 + 28 + 4) + (21 * 3 + 15 + 3) + (21 * 3 + 15 + 3),
        num_label=len(label_feature.names),
        lstm_size=96,
        lstm_layers=2,
        linear_size=128,
        bidirectional=True,
        loss_whole_sequence=False,
    )
    model = DeepSignV3(model_config).to(DEVICE)
    print("Number of parameters:", model.get_num_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    dl_params = dict(
        batch_size=MINIBATCH_SIZE * GESTURE_LENGTH,
        persistent_workers=True,
        shuffle=True,
        drop_last=True,
    )
    train_dl = DataLoader(ds["train"], num_workers=10, **dl_params)
    test_dl = DataLoader(ds["test"], num_workers=2, **dl_params)

    tags = [DATASET_NAME, "deepsign_v3", "fp32", "no_face"]
    if ENABLE_AUGMENTATION:
        tags.append("augmentation_v2")

    wandb.init(
        mode="disabled",
        project="deep-sign-v2",
        notes=f"golden-oath-121 loss&acc check last seq",
        config={
            "dataset": "v2",
            "batch_size": BATCH_SIZE,
            "num_epoch": NUM_EPOCH,
            "lr": LEARNING_RATE,
            "model_config": model_config,
            "loss_fn": model.criterion.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "train_count": ds["train"],
            "test_count": ds["test"],
            "train_batch_count": len(train_dl),
            "test_batch_count": len(test_dl),
            "augmentation": str(augmentation) if ENABLE_AUGMENTATION else "None",
            "num_params": model.get_num_parameters(),
            "gesture_len": GESTURE_LENGTH,
            "num_seq_accuracy_check": NUM_SEQ_ACCURACY_CHECK,
        },
        tags=tags,
    )

    best_acc = 0
    for epoch in (pbar := tqdm.tqdm(range(NUM_EPOCH), "[EPOCH]")):

        model.train()

        # TRAIN LOOP
        for sample in train_dl:
            input_pose = sample["pose"].to(DEVICE)
            # input_face = sample["face"].to(DEVICE)
            input_lh = sample["lh"].to(DEVICE)
            input_rh = sample["rh"].to(DEVICE)
            input_pose_angles = sample["pose_angles"].to(DEVICE)
            input_lh_angles = sample["lh_angles"].to(DEVICE)
            input_rh_angles = sample["rh_angles"].to(DEVICE)
            input_pose_mean = sample["pose_mean"].to(DEVICE)
            input_lh_mean = sample["lh_mean"].to(DEVICE)
            input_rh_mean = sample["rh_mean"].to(DEVICE)
            input = torch.cat(
                [
                    input_pose,
                    input_pose_mean,
                    input_pose_angles,
                    input_lh,
                    input_lh_mean,
                    input_lh_angles,
                    input_rh,
                    input_rh_mean,
                    input_rh_angles,
                ],
                dim=-1,
            )
            label = sample["label"].to(DEVICE)

            output, loss = model(input, target=label)

            # Accomodate mini-batch size in the loss
            loss = loss / (BATCH_SIZE / MINIBATCH_SIZE)

            # BACK PROPAGATION
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # TEST LOOP
        if (epoch + 1) % 10 == 0:
            train_loss, train_acc = get_loss_and_accuracy(model, train_dl)
            test_loss, test_acc = get_loss_and_accuracy(model, test_dl)

            data = {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
            pbar.set_postfix(**data)
            wandb.log(data, step=epoch)

        # SAVE MODEL IF ACCURACY INCREASED
        if (epoch + 1) % 10 == 0 and test_acc > best_acc:
            best_acc = test_acc
            base_path = f"checkpoints/{wandb.run.name}/"
            os.makedirs(base_path, exist_ok=True)

            # filename = f"{epoch}_{loss:.3f}.pt"
            filename = "checkpoint.pt"
            path = os.path.join(base_path, filename)

            torch.save(
                {
                    "config": model_config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": data["train_loss"],
                    "test_loss": data["test_loss"],
                    "epoch": epoch,
                    "test_acc": best_acc,
                    "label_names": label_feature.names,
                },
                path,
            )

    wandb.finish()
