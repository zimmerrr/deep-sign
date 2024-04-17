from typing import List
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# print(landmark_pb2.NormalizedLandmark([0, 0, 0))
p1 = mp.tasks.components.containers.NormalizedLandmark(0, 1, 2, 3)
# print(p1)
print(landmark_pb2.NormalizedLandmarkList(List(p1)))

#TODO: REVERSE ENGINEER NORMALIZED LANDMARK TO VIEW STORED DATA