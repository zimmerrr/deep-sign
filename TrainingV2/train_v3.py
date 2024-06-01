import time
from model.deepsign_v2 import DeepSignV2, DeepSignConfigV2
from augmentations.augmentation_v2 import AugmentationV2, Transform, Flip, Scale
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
import wandb
import os

os.environ["WANDB_API_KEY"] = "7204bcf714a2bd747d4d973bc999fbc86df91649"

torch.manual_seed(182731928)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "v3-fsl-105-v3"
NUM_EPOCH = 2000
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MINIBATCH_SIZE = 32
SEQUENCE_LENGTH = 15

# Make sure the batch size is divisible by the mini-batch size
assert BATCH_SIZE / MINIBATCH_SIZE == BATCH_SIZE // MINIBATCH_SIZE


@torch.no_grad()
def get_loss_and_accuracy(model, dl, criterion=None):
    model.eval()
    total_accuracy = []
    total_loss = []
    for sample in dl:
        input_pose = sample["pose"].to(DEVICE)
        input_face = sample["face"].to(DEVICE)
        input_lh = sample["lh"].to(DEVICE)
        input_rh = sample["rh"].to(DEVICE)
        input = torch.cat([input_pose, input_face, input_lh, input_rh], dim=-1)
        label = sample["label"].to(DEVICE)

        output = model(input)

        if criterion:
            loss = criterion(output, label)
            total_loss.append(loss.item())

        output = F.softmax(output)
        output = output.argmax(-1)

        total_correct = torch.sum(output == label)
        total_samples = len(label)
        accuracy = total_correct / total_samples
        total_accuracy.append(accuracy.item())

    avg_loss = sum(total_loss) / len(total_loss) if criterion else 0
    avg_accuracy = sum(total_accuracy) / len(total_accuracy)
    return avg_loss, avg_accuracy


# Jitter is applied as [-jitter / 2, jitter / 2]
augmentation = AugmentationV2(
    [
        Transform((0.5, 0.5, 0.25)),
        Flip(0.5, 0.2),
        Scale((0.5, 0.5, 0.25)),
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
]


# Make sure all examples have the same number of keypoints
# Pad the keypoints with the last frame until it reaches the max length
def pad_transform(examples):
    for sample_idx in range(len(examples["keypoints_length"])):
        curr_len = examples["keypoints_length"][sample_idx]
        missing = SEQUENCE_LENGTH - curr_len
        start_idx = 0

        if missing < 0:
            max_idx = curr_len - SEQUENCE_LENGTH
            start_idx = torch.randint(0, max_idx, (1,)).item()

        for field in fields_to_pad:
            field_value = torch.tensor(examples[field][sample_idx])

            if missing > 0:
                field_value = torch.concat(
                    [torch.zeros(missing, len(field_value[0])), field_value]
                )
            elif missing < 0:
                field_value = field_value[start_idx : start_idx + SEQUENCE_LENGTH]

            examples[field][sample_idx] = field_value

    return examples


def augment_and_pad_transform(examples):
    examples = pad_transform(examples)

    for sample_idx in range(len(examples["keypoints_length"])):
        pose = examples["pose"][sample_idx]
        face = examples["face"][sample_idx]
        lh = examples["lh"][sample_idx]
        rh = examples["rh"][sample_idx]

        pose, face, lh, rh = augmentation(pose, face, lh, rh)

        examples["pose"][sample_idx] = pose
        examples["face"][sample_idx] = face
        examples["lh"][sample_idx] = lh
        examples["rh"][sample_idx] = rh

    return examples


if __name__ == "__main__":
    ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}")
    # label_feature = ds["train"].features["label"]
    # idle_idx = label_feature.str2int("IDLE")
    # ds = ds.filter(lambda example: example["label"] != idle_idx)

    print(ds)
    ds = ds.with_format("torch")
    ds["train"].set_transform(augment_and_pad_transform)
    ds["test"].set_transform(pad_transform)

    model_config = DeepSignConfigV2(
        num_label=len(ds["train"].features["label"].names),
        lstm_size=256,
        lstm_layers=1,
        linear_size=128,
    )
    model = DeepSignV2(model_config).to(DEVICE)
    print("Number of parameters:", model.get_num_parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    dl_params = dict(
        batch_size=MINIBATCH_SIZE,
        persistent_workers=True,
    )
    train_dl = DataLoader(ds["train"], num_workers=10, **dl_params)
    test_dl = DataLoader(ds["test"], num_workers=2, **dl_params)

    wandb.init(
        # mode="disabled",
        project="deep-sign-v2",
        notes="fresh-sky-43 with generate_v3 with sequence length of 15 and augmentation",
        config={
            "dataset": "v2",
            "batch_size": BATCH_SIZE,
            "num_epoch": NUM_EPOCH,
            "lr": LEARNING_RATE,
            "model_config": model_config,
            "loss_fn": criterion.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "train_count": ds["train"],
            "test_count": ds["test"],
            "train_batch_count": len(train_dl),
            "test_batch_count": len(test_dl),
            "augmentation": str(augmentation),
        },
        tags=[DATASET_NAME, "deepsign_v2", "fp32", "augmentation_v1"],
    )

    best_acc = 0
    for epoch in (pbar := tqdm.tqdm(range(NUM_EPOCH), "[EPOCH]")):

        model.train()

        # TRAIN LOOP
        for sample in train_dl:
            input_pose = sample["pose"].to(DEVICE)
            input_face = sample["face"].to(DEVICE)
            input_lh = sample["lh"].to(DEVICE)
            input_rh = sample["rh"].to(DEVICE)
            input = torch.cat([input_pose, input_face, input_lh, input_rh], dim=-1)
            label = sample["label"].to(DEVICE)

            output = model(input)
            loss = criterion(output, label)

            # Accomodate mini-batch size in the loss
            loss = loss / (BATCH_SIZE / MINIBATCH_SIZE)

            # BACK PROPAGATION
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # TEST LOOP
        if (epoch + 1) % 10 == 0:
            train_loss, train_acc = get_loss_and_accuracy(model, train_dl, criterion)
            test_loss, test_acc = get_loss_and_accuracy(model, test_dl, criterion)

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
                },
                path,
            )

    wandb.finish()
