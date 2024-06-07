import time
import cv2
import numpy as np
import mediapipe as mp
import torch.nn.functional as F
from model.deepsign_v6 import DeepSignV6
import torch
from datasets import load_from_disk, ClassLabel
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints_v3
from multiprocessing import Pool


mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NAME = "earnest-cloud-155"
# DATASET_NAME = "v4-fsl-105-v4-20fps-orig"

checkpoint_path = f"./checkpoints/{RUN_NAME}/checkpoint.pt"
num_frames_to_idle = 5
target_fps = 20
input_seq_len = 60


WIDTH = 640
HEIGHT = 360


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


def kp_viz(frame, keypoints):
    pose = keypoints["pose"]
    lh = keypoints["lh"]
    rh = keypoints["rh"]

    pose_mean = keypoints["pose_mean"]
    lh_mean = keypoints["lh_mean"]
    rh_mean = keypoints["rh_mean"]

    # Pose Playback
    for idx in range(0, len(pose), 4):
        x = pose[idx] + pose_mean[0]
        y = pose[idx + 1] + pose_mean[1]
        z = pose[idx + 2] + pose_mean[2]
        visibility = pose[idx + 3]
        center = (int(x * WIDTH), int(y * HEIGHT))

        cv2.circle(frame, center, 3, (255, 255, 255))

    # Left Hand Playback
    for idx in range(0, len(lh), 3):
        x = lh[idx] + lh_mean[0]
        y = lh[idx + 1] + lh_mean[1]
        z = lh[idx + 2] + lh_mean[2]
        center = (int(x * WIDTH), int(y * HEIGHT))

        cv2.circle(frame, center, 3, (0, 0, 255))

    # Right Hand Playback
    for idx in range(0, len(rh), 3):
        x = rh[idx] + rh_mean[0]
        y = rh[idx + 1] + rh_mean[1]
        z = rh[idx + 2] + rh_mean[2]
        center = (int(x * WIDTH), int(y * HEIGHT))

        cv2.circle(frame, center, 3, (0, 255, 0))

    return frame


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
    image, results = mediapipe_detection(frame, holistic)
    frame_kp = extract_keypoints_v3(results)
    # frame_kp = unnormalized_keypoints(extract_keypoints_v3(results))
    keypoints = np.concatenate(
        [
            frame_kp["pose"],
            # frame_kp["pose_mean"],
            frame_kp["pose_angles"],
            # frame_kp["face"],
            frame_kp["lh"],
            # frame_kp["lh_mean"],
            frame_kp["lh_angles"],
            frame_kp["lh_dir"],
            frame_kp["rh"],
            # frame_kp["rh_mean"],
            frame_kp["rh_angles"],
            frame_kp["rh_dir"],
        ]
    )
    return image, keypoints, frame_kp


if __name__ == "__main__":
    pool = Pool(processes=1)
    # ds = load_from_disk(f"../datasets_cache/{DATASET_NAME}")

    checkpoint = torch.load(checkpoint_path)
    label_feature = ClassLabel(names=checkpoint["label_names"])
    print("Checkpoint Name:", RUN_NAME)
    print("Labels:", checkpoint["label_names"])
    print(
        "epoch:",
        checkpoint["epoch"],
        "test_acc:",
        checkpoint["test_acc"],
    )

    model = DeepSignV6(checkpoint["config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    recording = False
    num_frames_no_hand = 0
    last_frame_time = None
    last_preview_time = None
    last_output = None
    last_input_frames = []
    last_input_frames_idx = 0
    sequence_async = []
    sentence = []

    # Set mediapipe model
    with mp_hands.Hands(
        min_detection_confidence=0.65,
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
                num_frames_no_hand = 0

                result = pool.apply_async(process_frame, [frame])
                sequence_async.append(result)
                last_frame_time = time.time()
            elif (
                recording and not has_hands and num_frames_no_hand > num_frames_to_idle
            ):
                recording = False

                if len(sequence_async) > 10:
                    start_time = time.time()
                    keypoints = []
                    frames = []
                    for results in sequence_async[:-num_frames_to_idle]:
                        frame, kp, frame_kp = results.get()
                        keypoints.append(kp)
                        frames.append((frame, frame_kp))

                    last_input_frames = frames
                    last_input_frames_idx = 0
                    last_preview_time = time.time()
                    holistic_duration = round((time.time() - start_time) * 1000, 2)

                    # User has gone idle, make a prediction
                    start_time = time.time()
                    input = torch.tensor([keypoints], dtype=torch.float32)

                    # Trim or pad sequence
                    missing = input_seq_len - input.size(1)
                    if missing > 0:
                        pad = torch.zeros(1, missing, len(input[0][0]))
                        input = torch.concat([pad, input], dim=1)
                    else:
                        input = input[:, -input_seq_len:, :]

                    input = input.to(DEVICE)
                    output, _, _ = model(input)
                    output = F.softmax(output).cpu().detach()

                    last_output = output[0]
                    probs, label_indices = output[0].topk(1)
                    prob, label_idx = probs.numpy(), label_indices.numpy()

                    label = label_feature.int2str(int(label_idx))
                    if label != "IDLE":
                        sentence.append(label)
                        sentence = sentence[-5:]

                    deepsign_duration = round((time.time() - start_time) * 1000, 2)
                    print(
                        f"Gesture: {label} @ {len(input[0]) - missing} frames,",
                        f"DSign: {deepsign_duration}ms,",
                        f"Hol: {holistic_duration}ms",
                    )

            elif recording:
                if not has_hands:
                    num_frames_no_hand += 1

                now = time.time()
                if (now - last_frame_time) >= 1 / target_fps:
                    result = pool.apply_async(process_frame, [frame])
                    sequence_async.append(result)
                    sequence_async = sequence_async[-input_seq_len:]
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

            if last_output is not None:
                image = prob_viz(last_output, label_feature, image)

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

            # Display replay
            if len(last_input_frames) > 0:
                preview_width = int(WIDTH * 0.35)
                preview_height = int(HEIGHT * 0.35)
                preview, keypoints = last_input_frames[last_input_frames_idx]
                kp_viz(preview, keypoints)
                preview = cv2.resize(preview, (preview_width, preview_height))
                image[-preview_height:, -preview_width:] = preview

                now = time.time()
                if (now - last_preview_time) >= 1 / target_fps:
                    last_input_frames_idx = (last_input_frames_idx + 1) % len(
                        last_input_frames
                    )
                    last_preview_time = now

            # Show to screen
            cv2.imshow("OpenCV Feed", image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
