import time
import cv2
import numpy as np
import mediapipe as mp
import torch.nn.functional as F
from model.deepsign_v2 import DeepSignV2
import torch
from datasets import load_from_disk, ClassLabel
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints


def prob_viz(prediction: torch.Tensor, labels: ClassLabel, input_frame):
    row = 1
    probs, label_indices = prediction.topk(5)
    for prob, label_idx in zip(probs.numpy(), label_indices.numpy()):
        label = f"{labels.int2str(int(label_idx))} - {round(prob * 100, 3)}%"
        cv2.rectangle(
            input_frame,
            (0, 60 + row * 40),
            (int(np.round(prob * 900)), 90 + row * 40),
            (245, 117, 16),
            -1,
        )
        cv2.putText(
            input_frame,
            label,
            (0, 85 + row * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        row += 1

    return input_frame


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NAME = "royal-bird-48"
DATASET_NAME = "fsl-105-v2"

checkpoint_path = f"./checkpoints/{RUN_NAME}/checkpoint.pt"
sequence = []
sentence = []
predictions = []
threshold = 0.5
max_sequence = 30

if __name__ == "__main__":
    ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}")
    print("Labels:", ds["train"].features["label"].names)

    checkpoint = torch.load(checkpoint_path)
    model = DeepSignV2(checkpoint["config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            start_time = time.time()
            # print(predictions)
            # Read feed

            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-max_sequence:]

            if len(sequence) == max_sequence:
                input = torch.tensor([sequence], dtype=torch.float32).to(DEVICE)
                output = model(input)
                output = F.softmax(output).cpu().detach()

                # 3. Viz logic
                image = prob_viz(output[0], ds["train"].features["label"], image)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image,
                " ".join(sentence),
                (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Display FPS
            duration = round((time.time() - start_time) * 1000, 2)
            cv2.putText(
                image,
                f"Inference: {duration}ms",
                (0, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
            )

            # Show to screen
            cv2.imshow("OpenCV Feed", image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
