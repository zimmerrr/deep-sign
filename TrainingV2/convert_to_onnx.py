import torch
from model.deepsign_v5 import DeepSignV5, DeepSignConfigV5
import onnxruntime.backend as backend
import onnx
import json

RUN_NAME = "silver-dragon-146"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = f"./checkpoints/{RUN_NAME}/checkpoint.pt"
onnx_path = f"./checkpoints/{RUN_NAME}/deepsign.onnx"
label_path = f"./checkpoints/{RUN_NAME}/labels.json"

# Load model
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model_config: DeepSignConfigV5 = checkpoint["config"]
model = DeepSignV5(model_config).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Convert model to ONNX
input_tensor = torch.rand((1, 1, model_config.input_size), dtype=torch.float32)
input_tensor = input_tensor.to(DEVICE)
onnx_program = torch.onnx.export(
    model,
    input_tensor,
    onnx_path,
    verbose=False,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {1: "sequence"}},
)
print("Model succesfully converted at ", onnx_path)

# Test the model input
input_tensor = torch.rand((1, 10, model_config.input_size), dtype=torch.float32)
onnx_model = onnx.load(onnx_path)
output = backend.run(onnx_model, input_tensor.cpu().numpy())

# Save labels.json
with open(label_path, "w") as f:
    labels = checkpoint["label_names"]
    json.dump(
        {
            "labels": labels,
            "input_size": model_config.input_size,
        },
        f,
    )
print("Labels succesfully saved at ", label_path)
