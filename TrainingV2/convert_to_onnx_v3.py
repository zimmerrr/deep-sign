import torch
from model.deepsign_v3 import DeepSignV3, DeepSignConfigV3
import onnxruntime.backend as backend
import onnx
import json
from datasets import load_from_disk

RUN_NAME = "summer-thunder-101"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = f"./checkpoints/{RUN_NAME}/checkpoint.pt"
onnx_path = f"./checkpoints/{RUN_NAME}/deepsign.onnx"
label_path = f"./checkpoints/{RUN_NAME}/metadata.json"

ds = load_from_disk("../datasets_cache/v3-fsl-105-v3")
label_feature = ds["train"].features["label"]

# Load model
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
config: DeepSignConfigV3 = checkpoint["config"]
model = DeepSignV3(config).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

batch_size = 1
sequence_length = 10
input_size = (batch_size, sequence_length, config.input_size)
hn_size = (config.lstm_layers, batch_size, config.lstm_size)
# cn_size = (config.lstm_layers, batch_size, config.lstm_size)

# Convert model to ONNX
input_tensor = torch.rand(input_size, dtype=torch.float32).to(DEVICE)
hn_tensor = torch.rand(hn_size, dtype=torch.float32).to(DEVICE)
# cn_tensor = torch.rand(cn_size, dtype=torch.float32).to(DEVICE)
onnx_program = torch.onnx.export(
    model,
    (input_tensor, hn_tensor),
    onnx_path,
    verbose=False,
    opset_version=17,
    input_names=["input", "hn"],
    output_names=["output"],
    dynamic_axes={
        "input": {1: "sequence"},
    },
)
print("Model succesfully converted at ", onnx_path)

# Test the model input
input_tensor = torch.rand(input_size, dtype=torch.float32).to(DEVICE)
hn_tensor = torch.rand(hn_size, dtype=torch.float32).to(DEVICE)
# cn_tensor = torch.rand(cn_size, dtype=torch.float32).to(DEVICE)
onnx_model = onnx.load(onnx_path)
output = backend.run(
    onnx_model,
    [
        input_tensor.cpu().numpy(),
        hn_tensor.cpu().numpy(),
        # cn_tensor.cpu().numpy(),
    ],
)

# Save metadata.json
with open(label_path, "w") as f:
    json.dump(
        {
            "labels": label_feature.names,
            "input_size": config.input_size,
            "hn_size": hn_size,
            # "cn_size": cn_size,
        },
        f,
    )
print("Labels succesfully saved at ", label_path)
