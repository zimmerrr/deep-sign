import time
from model.deepsign_v5 import DeepSignV5, DeepSignConfigV5
from augmentations.augmentation_v2 import AugmentationV2, Transform, Flip, Scale, Rotate
from utils import get_hand_angles, get_pose_angles, get_directions
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
DATASET_NAME = "v4-fsl-143-v2-v3-20fps-with-flipped"
NUM_EPOCH = 400
LEARNING_RATE = 0.001
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 3
GESTURE_LENGTH = 1
BATCH_SIZE = 32
MINIBATCH_SIZE = 32
INPUT_SEQ_LENGTH = -1  # -1 for whole sequence
ENABLE_AUGMENTATION = False
# NUM_SEQ_ACCURACY_CHECK = 1

# Make sure the batch size is divisible by the mini-batch size
assert BATCH_SIZE / MINIBATCH_SIZE == BATCH_SIZE // MINIBATCH_SIZE


@torch.no_grad()
def get_loss_and_accuracy(model, dl):
    model.eval()
    total_accuracy = []
    total_loss = []
    total_indv_loss = []

    for sample in dl:
        input_pose = sample["pose"].to(DEVICE)
        input_lh = sample["lh"].to(DEVICE)
        input_rh = sample["rh"].to(DEVICE)
        input_pose_angles = sample["pose_angles"].to(DEVICE)
        input_lh_angles = sample["lh_angles"].to(DEVICE)
        input_rh_angles = sample["rh_angles"].to(DEVICE)
        input_lh_dir = sample["lh_dir"].to(DEVICE)
        input_rh_dir = sample["rh_dir"].to(DEVICE)
        input = torch.cat(
            [
                input_pose,
                input_pose_angles,
                input_lh,
                input_lh_angles,
                input_lh_dir,
                input_rh,
                input_rh_angles,
                input_rh_dir,
            ],
            dim=-1,
        )

        target_label = sample["label"].to(DEVICE)
        target_handshape = sample["handshape"].to(DEVICE)
        target_orientation = sample["orientation"].to(DEVICE)
        target_movement = sample["movement"].to(DEVICE)
        target_location = sample["location"].to(DEVICE)
        target_hands = sample["hands"].to(DEVICE)
        output, loss, indv_loss = model(
            input,
            target_label,
            target_handshape,
            target_orientation,
            target_movement,
            target_location,
            target_hands,
        )
        total_loss.append(loss.item())
        total_indv_loss.append({k: v.item() for k, v in indv_loss.items()})

        output = F.softmax(output, dim=-1)
        output = output.argmax(-1)

        # only compare the last NUM_SEQ_ACCURACY_CHECK sequences
        # output = output[:, -NUM_SEQ_ACCURACY_CHECK:]
        # target_label = target_label[:, -NUM_SEQ_ACCURACY_CHECK:]
        target_label = target_label.view(-1)

        total_correct = torch.sum(output == target_label)
        total_samples = target_label.numel()
        accuracy = total_correct / total_samples
        total_accuracy.append(accuracy.item())

    avg_loss = sum(total_loss) / len(total_loss)
    avg_accuracy = sum(total_accuracy) / len(total_accuracy)

    avg_indv_loss = {k: [] for k in total_indv_loss[0].keys()}
    for item in total_indv_loss:
        for k, v in item.items():
            avg_indv_loss[k].append(v)
    avg_indv_loss = {k: sum(v) / len(v) for k, v in avg_indv_loss.items()}
    return avg_loss, avg_accuracy, indv_loss


# Jitter is applied as [-jitter / 2, jitter / 2]
augmentation = AugmentationV2(
    [
        Rotate(15),
        # Flip(0.5, 0.2, (0, 0)),
        Scale((0.5, 0.5, 0)),
        Transform((0.5, 0.5, 0)),
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
    "lh_dir",
    "rh_dir",
]


def concat_gestures_transform(examples, gesture_length):
    new_examples = {k: [] for k in examples.keys()}

    for sample_idx in range(0, len(examples["keypoints_length"]), gesture_length):
        example = {k: [] for k in examples.keys()}
        for field in examples.keys():
            for i in range(gesture_length):
                if isinstance(examples[field][sample_idx + i], list):
                    example[field].extend(examples[field][sample_idx + i])
                else:
                    example[field].append(examples[field][sample_idx + i])

        example["keypoints_length"] = sum(example["keypoints_length"])

        for k, v in example.items():
            if type(v) == int or type(v[0]) == str or k in fields_to_pad:
                new_examples[k].append(v)
            else:
                new_examples[k].append(torch.tensor(v))

    return new_examples


# Make sure all examples have the same number of keypoints
# Pad the keypoints with the last frame until it reaches the max length
def pad_transform(examples):
    examples["pad_count"] = []

    if INPUT_SEQ_LENGTH == -1:
        max_len = max(examples["keypoints_length"])
    else:
        max_len = INPUT_SEQ_LENGTH

    for sample_idx in range(len(examples["keypoints_length"])):
        curr_len = examples["keypoints_length"][sample_idx]
        missing = max_len - curr_len
        start_idx = 0

        pad_count = max(missing, 0)
        examples["pad_count"].append(pad_count)

        if missing < 0:
            max_idx = curr_len - max_len
            start_idx = torch.randint(0, max_idx, (1,)).item()

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
            elif missing < 0:
                field_value = field_value[start_idx : start_idx + max_len]

            examples[field][sample_idx] = field_value

    return examples


def augment_transform(examples):
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


def unnormalized_keypoints(example):
    for field in ["pose", "lh", "rh"]:
        field_value = np.array(example[field])
        field_mean = np.array(example[f"{field}_mean"])

        seq_len = len(field_value)
        interleave = field_mean.shape[1]
        field_value = field_value.reshape(seq_len, -1, interleave)
        field_mean = np.expand_dims(field_mean, 1)

        field_value = field_value + field_mean
        example[field] = field_value.reshape(seq_len, -1)

    return example


def train_transform(examples):
    examples = concat_gestures_transform(examples, GESTURE_LENGTH)
    examples = pad_transform(examples)
    if ENABLE_AUGMENTATION:
        examples = augment_transform(examples)
    return examples


def test_transform(examples):
    examples = concat_gestures_transform(examples, 1)
    examples = pad_transform(examples)
    return examples


# def test_transform(examples):

if __name__ == "__main__":
    ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}")
    ds = ds.remove_columns(["file", "fps"])
    ds = ds.map(unnormalized_keypoints)
    label_feature = ds["train"].features["label"]
    handshape_feature = ds["train"].features["handshape"]
    orientation_feature = ds["train"].features["orientation"]
    movement_feature = ds["train"].features["movement"]
    location_feature = ds["train"].features["location"]
    hands_feature = ds["train"].features["hands"]

    print(ds)

    if ENABLE_AUGMENTATION:
        print(augmentation)

    ds = ds.with_format("torch")
    ds["train"].set_transform(train_transform)
    ds["test"].set_transform(test_transform)

    model_config = DeepSignConfigV5(
        num_label=len(label_feature.names),
        num_handshape=len(handshape_feature.names),
        num_orientation=len(orientation_feature.names),
        num_movement=len(movement_feature.names),
        num_location=len(location_feature.names),
        num_hands=len(hands_feature.names),
        input_size=(33 * 4 + 28)
        + (21 * 3 + 15 + 6 * 3)
        + (21 * 3 + 15 + 6 * 3),  # unnormalized input w/ directions
        label_lstm_size=64,
        label_lstm_layers=2,
        feature_lstm_size=24,
        feature_lstm_layers=2,
        label_linear_size=64,
        handshape_linear_size=24,
        orientation_linear_size=24,
        movement_linear_size=24,
        location_linear_size=24,
        hands_linear_size=24,
        bidirectional=True,
        label_smoothing=0.0,
        dropout=0.5,
    )
    model = DeepSignV5(model_config).to(DEVICE)
    print("Number of parameters:", model.get_num_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        LR_REDUCE_FACTOR,
        LR_REDUCE_PATIENCE,
    )

    dl_params = dict(
        persistent_workers=True,
        shuffle=True,
        drop_last=True,
    )
    train_dl = DataLoader(
        ds["train"],
        num_workers=12,
        batch_size=MINIBATCH_SIZE * GESTURE_LENGTH,
        **dl_params,
    )
    test_dl = DataLoader(
        ds["test"],
        num_workers=2,
        batch_size=MINIBATCH_SIZE,
        **dl_params,
    )

    tags = [DATASET_NAME, "deepsign_v5", "train_v5", "fp32", "no_face"]
    if ENABLE_AUGMENTATION:
        tags.append("augmentation_v2")

    wandb.init(
        # mode="disabled",
        project="deep-sign-v2",
        notes=f"New model with asl structures, flipped dataset",
        config={
            "dataset": "v2",
            "batch_size": BATCH_SIZE,
            "num_epoch": NUM_EPOCH,
            "lr": LEARNING_RATE,
            "lr_reduce_factor": LR_REDUCE_FACTOR,
            "lr_reduce_patience": LR_REDUCE_PATIENCE,
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
            # "num_seq_accuracy_check": NUM_SEQ_ACCURACY_CHECK,
            "input_seq_length": INPUT_SEQ_LENGTH,
        },
        tags=tags,
    )

    best_acc = 0
    for epoch in (pbar := tqdm.tqdm(range(NUM_EPOCH), "[EPOCH]")):

        model.train()

        # TRAIN LOOP
        for sample in train_dl:
            input_pose = sample["pose"].to(DEVICE)
            input_lh = sample["lh"].to(DEVICE)
            input_rh = sample["rh"].to(DEVICE)
            input_pose_angles = sample["pose_angles"].to(DEVICE)
            input_lh_angles = sample["lh_angles"].to(DEVICE)
            input_rh_angles = sample["rh_angles"].to(DEVICE)
            input_lh_dir = sample["lh_dir"].to(DEVICE)
            input_rh_dir = sample["rh_dir"].to(DEVICE)
            input = torch.cat(
                [
                    input_pose,
                    input_pose_angles,
                    input_lh,
                    input_lh_angles,
                    input_lh_dir,
                    input_rh,
                    input_rh_angles,
                    input_rh_dir,
                ],
                dim=-1,
            )

            target_label = sample["label"].to(DEVICE)
            target_handshape = sample["handshape"].to(DEVICE)
            target_orientation = sample["orientation"].to(DEVICE)
            target_movement = sample["movement"].to(DEVICE)
            target_location = sample["location"].to(DEVICE)
            target_hands = sample["hands"].to(DEVICE)
            output, loss, indv_loss = model(
                input,
                target_label,
                target_handshape,
                target_orientation,
                target_movement,
                target_location,
                target_hands,
            )

            # Accomodate mini-batch size in the loss
            loss = loss / (BATCH_SIZE / MINIBATCH_SIZE)

            # BACK PROPAGATION
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # TEST LOOP
        if (epoch + 1) % 10 == 0:
            train_loss, train_acc, train_indv_loss = get_loss_and_accuracy(
                model, train_dl
            )
            test_loss, test_acc, test_indv_loss = get_loss_and_accuracy(model, test_dl)
            scheduler.step(test_loss)

            data = {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            pbar.set_postfix(**data)
            data.update({f"train_{k}": v for k, v in train_indv_loss.items()})
            data.update({f"test_{k}": v for k, v in test_indv_loss.items()})
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
                    "handshape_names": handshape_feature.names,
                    "orientation_names": orientation_feature.names,
                    "movement_names": movement_feature.names,
                    "location_names": location_feature.names,
                    "hands_names": hands_feature.names,
                },
                path,
            )

    wandb.finish()
