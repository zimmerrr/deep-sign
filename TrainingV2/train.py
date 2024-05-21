from model.deepsign import DeepSign
from datasets import load_from_disk
import torch.nn.functional as F



if __name__ == "__main__":
    ds = load_from_disk("../cache")
    ds = ds.with_format('torch')
    # print(ds[0]['sequence'].shape)
    model = DeepSign()
    model.train()
    preds = model(ds[0]['sequence'])
    preds = F.softmax(preds, -1)
    print(preds.argmax(-1))
