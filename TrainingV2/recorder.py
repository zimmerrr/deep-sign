import cv2
import numpy as np
import mediapipe as mp
import torch.nn.functional as F
from model.deepsign import DeepSign
import torch
import os
from datasets import load_from_disk, ClassLabel
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "./checkpoints/prime-lake-22/checkpoint.pt"


# TODO: add timestamp to output folder for continous recording

DATA_PATH = os.path.join('../Data') 
NO_SEQUNCE = 30
SEQUENCE_LENGTH = 30
START_FOLDER = 1

actions = np.array([
    'hello', 'thanks', 'iloveyou', 'idle',
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y',
    'Z',
    ])


def generate_folder(): 
    for action in actions: 
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            os.makedirs(action_path)
        dirmax = np.max([0] + [int(d) for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))])
        for sequence in range(1, NO_SEQUNCE+1):
            try: 
                os.makedirs(os.path.join(action_path, str(dirmax+sequence)))
            except:
                pass

if __name__ == "__main__":
    generate_folder()
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Loop through actions
            for action in actions:
                # Loop through sequences aka videos
                for sequence in range(START_FOLDER, START_FOLDER+NO_SEQUNCE):
                    # Loop through video length aka sequence length
                    for frame_num in range(SEQUENCE_LENGTH):

                        ret, frame = cap.read()

                        image, results = mediapipe_detection(frame, holistic)

                        draw_styled_landmarks(image, results)
                        
                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(500)
                        else: 
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                        
                        # NEW Export keypoints
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                            
        cap.release()
        cv2.destroyAllWindows()