import cv2
import os
import glob
import numpy as np
from time import sleep

WIDTH = 640
HEIGHT = 480
NUM_SAMPLES = 120

ACTIONS = [
    'hello', 'thanks', 'iloveyou', 'idle',
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y',
    'Z',
    ]
def play(path):
    files = glob.glob(os.path.join(path, "*.npy"))
    last_key = -1
    for file in files:
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        data = np.load(file)
        
        # Reverse exteract keypoints
        pose = data[0: 132]
        face = data[132: 1404+132]
        rh = data[1404+132: 63+1404+132]
        lh = data[63+1404+132: 63+1404+132+63+1]

        #Pose Playback
        for idx in range(0, len(pose), 4):
            x = pose[idx]
            y = pose[idx + 1]
            z = pose[idx + 2]
            visibility = pose[idx + 3]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 5, (255, 255, 255))

        #Face Playback
        for idx in range(0, len(face), 3):
            x = face[idx]
            y = face[idx + 1]
            z = face[idx + 2]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 5, (255, 0, 0))

        #Right Hand Playback
        for idx in range(0, len(rh), 3):
            x = rh[idx]
            y = rh[idx + 1]
            z = rh[idx + 2]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 5, (0, 255, 0))

        #Left Hand Playback
        for idx in range(0, len(lh), 3):
            x = lh[idx]
            y = lh[idx + 1]
            z = lh[idx + 2]
            center = (int(x * WIDTH), int(y * HEIGHT))

            cv2.circle(image, center, 10, (0, 0, 255))
        cv2.imshow('Deep Sign Playback', image)

        last_key = cv2.waitKey(200)
        if last_key != -1:
            break

    return last_key

if __name__ == "__main__":
    runing = True
    sample_idx = 0
    gesture_idx = 0

    while runing:
        path = f"../Data/{ACTIONS[gesture_idx + 1]}/{sample_idx + 1}"
        if not os.path.exists(path):
            last_key = cv2.waitKey(1000)
            print(f"Unable to find: {path}")
        else: 
            last_key = play(path)
        print(path)
        # print(index+1)

        # NEXT SAMPLE
        if last_key == 100:
            sample_idx = (sample_idx + 1) % NUM_SAMPLES
            continue
        # PREVIOUS SAMPLE
        elif last_key == 97:
            sample_idx = (sample_idx + NUM_SAMPLES - 1 ) % NUM_SAMPLES
            continue

        # NEXT GESTURE
        elif last_key == 119:
            gesture_idx = (gesture_idx + len(ACTIONS) + 1 ) % len(ACTIONS)
            continue
        # PREVIOUS GESTURE
        elif last_key == 155:
            gesture_idx = (gesture_idx + len(ACTIONS) - 1 ) % len(ACTIONS)
            continue
        elif last_key == 32:
            runing = False
            break
        # TODO: ADD LEFT RIGHT ARROW KEY TO NEXT