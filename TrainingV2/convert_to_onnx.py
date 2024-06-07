import torch
from model.deepsign_v6 import DeepSignV6, DeepSignConfigV6
import onnxruntime.backend as backend
import onnx
import json

RUN_NAME = "earnest-cloud-155"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = f"./checkpoints/{RUN_NAME}/checkpoint.pt"
onnx_path = f"./checkpoints/{RUN_NAME}/deepsign.onnx"
label_path = f"./checkpoints/{RUN_NAME}/labels.json"

# Load model
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
config: DeepSignConfigV6 = checkpoint["config"]
model = DeepSignV6(config).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

batch_size = 1
sequence_length = 10
input_size = (batch_size, sequence_length, config.input_size)
hidden_feature_size = (
    (
        config.feature_lstm_layers * 2
        if config.bidirectional
        else config.feature_lstm_layers
    ),
    batch_size,
    config.feature_lstm_size,
)
hidden_label_size = (
    config.label_lstm_layers * 2 if config.bidirectional else config.label_lstm_layers,
    batch_size,
    config.label_lstm_size,
)

# Convert model to ONNX
input_tensor = torch.rand(input_size, dtype=torch.float32)
input_tensor = input_tensor.to(DEVICE)
hidden_feature_tensor = torch.rand(hidden_feature_size, dtype=torch.float32).to(DEVICE)
hidden_label_tensor = torch.rand(hidden_label_size, dtype=torch.float32).to(DEVICE)
onnx_program = torch.onnx.export(
    model,
    (input_tensor, hidden_feature_tensor, hidden_label_tensor),
    onnx_path,
    verbose=False,
    opset_version=17,
    input_names=["input", "hidden_features", "hidden_label"],
    output_names=["output"],
    dynamic_axes={
        "input": {1: "sequence"},
    },
)
print("Model succesfully converted at ", onnx_path)

# Test the model input
input_tensor = torch.rand(input_size, dtype=torch.float32)
hidden_feature_tensor = torch.rand(hidden_feature_size, dtype=torch.float32)
hidden_label_tensor = torch.rand(hidden_label_size, dtype=torch.float32)
onnx_model = onnx.load(onnx_path)
output = backend.run(
    onnx_model,
    [
        input_tensor.cpu().numpy(),
        hidden_feature_tensor.cpu().numpy(),
        hidden_label_tensor.cpu().numpy(),
    ],
)

# Save labels.json
with open(label_path, "w") as f:
    labels = checkpoint["label_names"]
    json.dump(
        {
            "labels": labels,
            "input_size": config.input_size,
            "hidden_feature_size": hidden_feature_size,
            "hidden_label_size": hidden_label_size,
        },
        f,
    )
print("Labels succesfully saved at ", label_path)
