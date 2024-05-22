from model.deepsign import DeepSign, DeepSignConfig
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
import wandb
import os


torch.manual_seed(182731928)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCH = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 128

@torch.no_grad()
def get_accuracy(model, dl):
    model.eval()
    total_accuracy = []
    for sample in dl:
        input = sample['sequence'].to(DEVICE)
        label = sample['label'].to(DEVICE)
        output = model(input)
        output = F.softmax(output)
        output = output.argmax(-1)

        total_correct = torch.sum(output == label)
        total_samples = len(label)
        accuracy = total_correct / total_samples
        total_accuracy.append(accuracy.item())

    return sum(total_accuracy) / len(total_accuracy)
    

if __name__ == "__main__":
    ds = load_from_disk("../cache2")
    ds = ds.with_format('torch')

    model_config = DeepSignConfig()
    model = DeepSign(model_config).to(DEVICE)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_dl = DataLoader(ds['train'], batch_size=BATCH_SIZE)
    test_dl = DataLoader(ds['test'], batch_size=BATCH_SIZE)

    wandb.init(
        project="deep-sign-v2", 
        notes="",
        config={
            "dataset": "v1",
            "batch_size": BATCH_SIZE,
            "num_epoch": NUM_EPOCH,
            "lr": LEARNING_RATE,
            "model_config": model_config,
            "loss_fn": criterion.__class__.__name__, 
            "optimizer": optimizer.__class__.__name__, 
            "train_count": len(train_dl),
            "test_count": len(test_dl),
        },
        tags=[
            "deepsign-dataset-v1"
        ],
    )  

    best_acc = 0
    for epoch in (pbar := tqdm.tqdm(range(NUM_EPOCH), '[EPOCH]')):
        
        model.train()

        #TEST LOOP
        train_loss = []
        for sample in train_dl:
            input = sample['sequence'].to(DEVICE)
            label = sample['label'].to(DEVICE)
           
            output = model(input)
            loss = criterion(output, label)
            train_loss.append(loss.item())
            
            #BACK PROPAGATION
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #TEST LOOP
        model.eval()
        test_loss = []
        for sample in test_dl:
            input = sample['sequence'].to(DEVICE)
            label = sample['label'].to(DEVICE)
           
            output = model(input)
            loss = criterion(output, label)
            test_loss.append(loss.item())
       
        test_acc = get_accuracy(model, test_dl)

        data = {
            "train_loss": round(sum(train_loss)/ len(train_loss), 3),
            "test_loss": round(sum(test_loss)/ len(test_loss), 3),
            "train_acc": get_accuracy(model, train_dl),
            "test_acc": test_acc
        }
        pbar.set_postfix(**data)
        wandb.log(data, step=epoch)
        
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
                    "train_loss": data['train_loss'],
                    "test_loss": data['test_loss'],
                    "epoch": epoch,
                },
                path
            )


    wandb.finish()
    