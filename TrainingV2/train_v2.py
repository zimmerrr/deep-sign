import time
from model.deepsign_v2 import DeepSignV2, DeepSignConfigV2
from augmentations.augmentation_v1 import AugmentationV1, Transform, Flip, Scale
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
DATASET_NAME = "fsl-105-v2"
NUM_EPOCH = 2000
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MINIBATCH_SIZE = 32

# Make sure the batch size is divisible by the mini-batch size
assert BATCH_SIZE / MINIBATCH_SIZE == BATCH_SIZE // MINIBATCH_SIZE


@torch.no_grad()
def get_loss_and_accuracy(model, dl, criterion=None):
    model.eval()
    total_accuracy = []
    total_loss = []
    for sample in dl:
        input = sample["keypoints"].to(DEVICE)
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
augmentation = AugmentationV1(
    [
        Transform((0.5, 0.5, 0.25)),
        Flip(0.5, 0.2),
        Scale((0.5, 0.5, 0.25)),
    ]
)


# Make sure all examples have the same number of keypoints
# Pad the keypoints with the last frame until it reaches the max length
def pad_transform(examples):
    max_len = max([len(kp) for kp in examples["keypoints"]])

    pad_value = torch.zeros(1, 1662)
    for idx, frame in enumerate(examples["keypoints"]):
        missing = max_len - len(frame)
        frame = torch.tensor(frame)
        if missing > 0:
            frame = torch.concat(
                [
                    pad_value.repeat(missing, 1),
                    frame,
                ]
            )

        examples["keypoints"][idx] = frame

    return examples


def augment_and_pad_transform(examples):
    max_len = max([len(kp) for kp in examples["keypoints"]])

    pad_value = torch.zeros(1, 1662)
    for idx, frame in enumerate(examples["keypoints"]):
        # Apply augmentation
        frame = augmentation(torch.tensor(frame))

        # Pad the frame with zeros at the beginning
        missing = max_len - len(frame)
        if missing > 0:
            frame = torch.concat(
                [
                    pad_value.repeat(missing, 1),
                    frame,
                ]
            )

        examples["keypoints"][idx] = frame

    return examples


if __name__ == "__main__":
    ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}")
    print(ds)
    ds = ds.with_format("torch")
    ds["train"].set_transform(augment_and_pad_transform)
    ds["test"].set_transform(pad_transform)

    model_config = DeepSignConfigV2(
        num_label=len(ds["train"].features["label"].names),
        lstm_size=256,
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
    train_dl = DataLoader(ds["train"], num_workers=8, **dl_params)
    test_dl = DataLoader(ds["test"], num_workers=2, **dl_params)

    wandb.init(
        # mode="disabled",
        project="deep-sign-v2",
        notes="fresh-sky-43 with augmentation",
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
            input = sample["keypoints"].to(DEVICE)
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
