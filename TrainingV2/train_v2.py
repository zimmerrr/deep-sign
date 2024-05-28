import time
from model.deepsign_v2 import DeepSignV2, DeepSignConfigV2
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
import wandb
import os


torch.manual_seed(182731928)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCH = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 128


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


# Make sure all examples have the same number of keypoints
# Pad the keypoints with the last frame until it reaches the max length
def padding_transform(examples):
    max_len = max([len(kp) for kp in examples["keypoints"]])

    for keypoint in examples["keypoints"]:
        missing = max_len - len(keypoint)
        for _ in range(missing):
            keypoint.append(keypoint[-1])

    examples["keypoints"] = torch.tensor(examples["keypoints"], dtype=torch.float32)
    return examples


if __name__ == "__main__":
    ds = load_from_disk("../datasets_cache/fsl-143-v1")
    print(ds)
    ds = ds.with_format("torch")
    ds.set_transform(padding_transform)

    model_config = DeepSignConfigV2(
        num_label=len(ds["train"].features["label"].names),
        lstm1_size=256,
        lstm2_size=256,
        lstm3_size=256,
        linear_size=128,
    )
    model = DeepSignV2(model_config).to(DEVICE)
    print("Number of parameters:", model.get_num_parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dl_params = dict(
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=6,
    )
    train_dl = DataLoader(ds["train"], **dl_params)
    test_dl = DataLoader(ds["test"], **dl_params)

    wandb.init(
        # mode="disabled",
        project="deep-sign-v2",
        notes="",
        config={
            "dataset": "fsl-143-v1",
            "batch_size": BATCH_SIZE,
            "num_epoch": NUM_EPOCH,
            "lr": LEARNING_RATE,
            "model_config": model_config,
            "loss_fn": criterion.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "train_count": len(train_dl),
            "test_count": len(test_dl),
        },
        tags=["fsl-143-v1", "deepsign_v2"],
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

            # BACK PROPAGATION
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # TEST LOOP
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
