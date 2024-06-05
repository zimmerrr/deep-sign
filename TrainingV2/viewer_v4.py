import time
import cv2
import numpy as np
import mediapipe as mp
import torch.nn.functional as F
from model.deepsign_v3 import DeepSignV3
import torch
from datasets import load_from_disk, ClassLabel
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints_v3
from multiprocessing import Pool


mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NAME = "youthful-feather-135"
# DATASET_NAME = "v4-fsl-105-v4-20fps-orig"

checkpoint_path = f"./checkpoints/{RUN_NAME}/checkpoint.pt"
# input_size=(33 * 4 + 28 + 4) + (21 * 3 + 15 + 3) + (21 * 3 + 15 + 3)  # normalized input
input_size = (33 * 4 + 28) + (21 * 3 + 15) + (21 * 3 + 15)  # unnormalized input
sequence_async = []
sentence = []
num_frames_to_idle = 10
target_fps = 20


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


def unnormalized_keypoints(example):
    for field in ["pose", "lh", "rh"]:
        field_value = np.array(example[field])
        field_mean = np.array(example[f"{field}_mean"])

        interleave = len(field_mean)
        field_value = field_value.reshape(-1, interleave)

        field_value = field_value + field_mean
        example[field] = field_value.reshape(-1)

    return example


holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)


def process_frame(frame):
    _, results = mediapipe_detection(frame, holistic)
    frame_kp = unnormalized_keypoints(extract_keypoints_v3(results))
    return np.concatenate(
        [
            frame_kp["pose"],
            # frame_kp["pose_mean"],
            frame_kp["pose_angles"],
            # frame_kp["face"],
            frame_kp["lh"],
            # frame_kp["lh_mean"],
            frame_kp["lh_angles"],
            frame_kp["rh"],
            # frame_kp["rh_mean"],
            frame_kp["rh_angles"],
        ]
    )


if __name__ == "__main__":
    pool = Pool(processes=1)
    # ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}")

    checkpoint = torch. load (checkpoint_path, map_location = torch.device(DEVICE))
    label_feature = ClassLabel(names=checkpoint["label_names"])
    print("Checkpoint Name:", RUN_NAME)
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

    recording = False
    num_frames_no_hand = 0
    last_frame_time = None

    # Set mediapipe model
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as hands:
        while cap.isOpened():
            # print(predictions)
            # Read feed

            start_time = time.time()
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, hands)

            # draw_styled_landmarks(image, results)

            has_hands = results.multi_hand_landmarks

            if not recording and has_hands:
                # User is starting to sign
                recording = True
                sequence_async = []

                result = pool.apply_async(process_frame, [frame])
                sequence_async.append(result)
                last_frame_time = time.time()
            elif (
                recording and not has_hands and num_frames_no_hand > num_frames_to_idle
            ):
                recording = False

                if len(sequence_async):
                    start_time = time.time()
                    keypoints = []
                    for results in sequence_async:
                        keypoints.append(results.get())

                    holistic_duration = round((time.time() - start_time) * 1000, 2)

                    # User has gone idle, make a prediction
                    start_time = time.time()
                    input = torch.tensor([keypoints], dtype=torch.float32).to(DEVICE)
                    output, _ = model(input)
                    output = F.softmax(output[:, -1, :]).cpu().detach()

                    probs, label_indices = output[0].topk(1)
                    prob, label_idx = probs.numpy(), label_indices.numpy()

                    label = label_feature.int2str(int(label_idx))
                    if label != "IDLE":
                        sentence.append(label)
                        sentence = sentence[-5:]

                    deepsign_duration = round((time.time() - start_time) * 1000, 2)
                    print(
                        f"Gesture: {label} @ {len(input[0])} frames,",
                        f"DSign: {deepsign_duration}ms,",
                        f"Hol: {holistic_duration}ms",
                    )

                    # image = prob_viz(output[0], label_feature, image)
            elif recording:
                if not has_hands:
                    num_frames_no_hand += 1

                now = time.time()
                if (now - last_frame_time) >= 1 / target_fps:
                    result = pool.apply_async(process_frame, [frame])
                    sequence_async.append(result)
                    last_frame_time += 1 / target_fps

            frame_duration = round((time.time() - start_time) * 1000, 2)
            cv2.rectangle(image, (0, 0), (640, 20), (245, 117, 16), -1)
            cv2.putText(
                image,
                " ".join(sentence),
                (3, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Display FPS
            cv2.putText(
                image,
                f"Frame: {frame_duration}ms",
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
