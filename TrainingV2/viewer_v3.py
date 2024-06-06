import time
import cv2
import numpy as np
import mediapipe as mp
import torch.nn.functional as F
from model.deepsign_v3 import DeepSignV3
import torch
from datasets import load_from_disk, ClassLabel
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints_v3


def prob_viz(prediction: torch.Tensor, labels: ClassLabel, input_frame):
    labels.names
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
RUN_NAME = "cosmic-sound-125"
# DATASET_NAME = "v4-fsl-105-v4-20fps-orig"

checkpoint_path = f"./checkpoints/{RUN_NAME}/checkpoint.pt"
max_sequence = 60
# input_size=(33 * 4 + 28 + 4) + (21 * 3 + 15 + 3) + (21 * 3 + 15 + 3)  # normalized input
input_size = (33 * 4 + 28) + (21 * 3 + 15) + (21 * 3 + 15)  # unnormalized input
sequence = [np.zeros(input_size) for _ in range(max_sequence)]
sentence = []
predictions = []
threshold = 0.5

if __name__ == "__main__":
    # ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}")

    checkpoint = torch.load(checkpoint_path)
    label_feature = ClassLabel(names=checkpoint["label_names"])
    print("Labels:", checkpoint["label_names"])
    print(
        "epoch:",
        checkpoint["epoch"],
        "test_acc:",
        checkpoint["test_acc"],
    )

    model = DeepSignV3(checkpoint["config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # Set mediapipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:
        while cap.isOpened():
            # print(predictions)
            # Read feed

            overall_start_time = time.time()
            start_time = time.time()
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            holistic_duration = round((time.time() - start_time) * 1000, 2)

            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints_v3(results)
            sequence.append(
                np.concatenate(
                    [
                        keypoints["pose"],
                        # keypoints["pose_mean"],
                        keypoints["pose_angles"],
                        # keypoints["face"],
                        keypoints["lh"],
                        # keypoints["lh_mean"],
                        keypoints["lh_angles"],
                        keypoints["rh"],
                        # keypoints["rh_mean"],
                        keypoints["rh_angles"],
                    ]
                )
            )
            sequence = sequence[-max_sequence:]

            start_time = time.time()
            if len(sequence) == max_sequence:
                input = torch.tensor([sequence], dtype=torch.float32).to(DEVICE)
                output, _ = model(input)
                output = F.softmax(output[:, -1, :]).cpu().detach()

                # 3. Viz logic
                # image = prob_viz(output[0], ds["train"].features["label"], image)
                # image = prob_viz(output[0], label_feature, image)
            deepsign_duration = round((time.time() - start_time) * 1000, 2)
            overall_duration = round((time.time() - overall_start_time) * 1000, 2)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            # cv2.putText(
            #     image,
            #     " ".join(sentence),
            #     (3, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (255, 255, 255),
            #     2,
            #     cv2.LINE_AA,
            # )

            # Display FPS
            cv2.putText(
                image,
                f"Hol: {holistic_duration}ms, DSign: {deepsign_duration}ms, All: {overall_duration}ms",
                (0, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Show to screen
            cv2.imshow("OpenCV Feed", image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
