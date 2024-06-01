import mediapipe as mp
import cv2
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)    
    image.flags.writeable = True                  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh], dtype=np.float32)

def extract_keypoints_v2(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    pose_mean_x = np.mean(pose[0::4])
    pose_mean_y = np.mean(pose[1::4])
    pose_mean_z = np.mean(pose[2::4])
    pose_mean_visibility = np.mean(pose[3::4])
    face_mean_x = np.mean(face[0::3])
    face_mean_y = np.mean(face[1::3])
    face_mean_z = np.mean(face[2::3])
    lh_mean_x = np.mean(lh[0::3])
    lh_mean_y = np.mean(lh[1::3])
    lh_mean_z = np.mean(lh[2::3])
    rh_mean_x = np.mean(rh[0::3])
    rh_mean_y = np.mean(rh[1::3])
    rh_mean_z = np.mean(rh[2::3])

    pose[0::4] = pose[0::4] - pose_mean_x
    pose[1::4] = pose[1::4] - pose_mean_y
    pose[2::4] = pose[2::4] - pose_mean_z
    pose[3::4] = pose[3::4] - pose_mean_visibility
    face[0::3] = face[0::3] - face_mean_x
    face[1::3] = face[1::3] - face_mean_y
    face[2::3] = face[2::3] - face_mean_z
    lh[0::3] = lh[0::3] - lh_mean_x
    lh[1::3] = lh[1::3] - lh_mean_y
    lh[2::3] = lh[2::3] - lh_mean_z
    rh[0::3] = rh[0::3] - rh_mean_x
    rh[1::3] = rh[1::3] - rh_mean_y
    rh[2::3] = rh[2::3] - rh_mean_z
    
     
    return (
        pose.astype(np.float32),
        face.astype(np.float32),
        lh.astype(np.float32),
        rh.astype(np.float32),
        np.array([pose_mean_x, pose_mean_y, pose_mean_z, pose_mean_visibility], dtype=np.float32),
        np.array([face_mean_x, face_mean_y, face_mean_z], dtype=np.float32),
        np.array([lh_mean_x, lh_mean_y, lh_mean_z], dtype=np.float32),
        np.array([rh_mean_x, rh_mean_y, rh_mean_z], dtype=np.float32),
    )
