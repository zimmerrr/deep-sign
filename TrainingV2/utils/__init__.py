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
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
    )  # Draw face connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )  # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )  # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh], dtype=np.float32)


def extract_keypoints_v2(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

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
        np.array(
            [pose_mean_x, pose_mean_y, pose_mean_z, pose_mean_visibility],
            dtype=np.float32,
        ),
        np.array([face_mean_x, face_mean_y, face_mean_z], dtype=np.float32),
        np.array([lh_mean_x, lh_mean_y, lh_mean_z], dtype=np.float32),
        np.array([rh_mean_x, rh_mean_y, rh_mean_z], dtype=np.float32),
    )


def normalize_keypoints(data, interleave):
    means = []
    for i in range(interleave):
        mean = np.mean(data[i::interleave])
        means.append(mean)
        data[i::interleave] = data[i::interleave] - mean
        
    return data.astype(np.float32), np.array(means, dtype=np.float32)


# Reference: https://github.com/kairess/gesture-recognition/blob/master/create_dataset.py
def get_angles(data, interleave, parent_indices, child_indices, vec_a, vec_b):
    data = np.array(data, dtype=np.float32).reshape(-1, interleave)
    if not data.any():
        return np.zeros(len(vec_a), dtype=np.float32)

    v1 = data[parent_indices, :interleave]
    v2 = data[child_indices, :interleave]
    v = v2 - v1

    # Normalize
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    v_dot = np.einsum("nt,nt->n", v[vec_a, :], v[vec_b, :])
    v_dot = np.clip(v_dot, -1.0, 1.0)
    angles = np.arccos(v_dot)
    angles = np.degrees(angles)

    if np.isnan(angles).any():
        print("NaN detected", v_dot)
        return np.zeros(len(vec_a), dtype=np.float32)

    return angles


def get_hand_angles(data):
    parent = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    child = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    vec_a = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
    vec_b = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    return get_angles(data, 3, parent, child, vec_a, vec_b)


def get_pose_angles(data):
    parent = [
        [0, 1, 2, 3, 0, 4, 5, 6],  # nose, eyes
        [9],  # mouth
        [11, 13, 15, 15, 15, 11, 23, 25, 27, 27],  # left body
        [12, 14, 16, 16, 16, 12, 24, 26, 28, 28],  # right body
        [11, 23],  # body
        # [17, 18],  # fingers
        # [29, 30],  # foot
    ]
    child = [
        [1, 2, 3, 7, 4, 5, 6, 8],  # nose, eyes
        [10],  # mouth
        [13, 15, 17, 19, 21, 23, 25, 27, 29, 31],  # left body
        [14, 16, 18, 20, 22, 24, 26, 28, 30, 32],  # right body
        [12, 24],  # body
        # [19, 20],  # fingers
        # [31, 32],  # foot
    ]
    vec_a = [
        [0, 1, 2, 4, 5, 6],  # nose, eyes
        [9, 10, 10, 10, 14, 15, 16, 16, 9],  # left body
        [19, 20, 20, 20, 24, 25, 26, 26, 19],  # right body
        [29, 29, 30, 30],  # body
    ]
    vec_b = [
        [1, 2, 3, 5, 6, 7],  # nose, eyes
        [10, 11, 12, 13, 15, 16, 17, 18, 14],  # left body
        [20, 21, 22, 23, 25, 26, 27, 28, 24],  # right body
        [14, 24, 15, 25],  # body
    ]

    # Flatten the list
    parent = sum(parent, [])
    child = sum(child, [])
    vec_a = sum(vec_a, [])
    vec_b = sum(vec_b, [])

    return get_angles(data, 4, parent, child, vec_a, vec_b)


def extract_keypoints_v3(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    pose, pose_mean = normalize_keypoints(pose, 4)
    face, face_mean = normalize_keypoints(face, 3)
    lh, lh_mean = normalize_keypoints(lh, 3)
    rh, rh_mean = normalize_keypoints(rh, 3)

    return dict(
        pose=pose,
        pose_angles=get_pose_angles(pose),
        face=face,
        lh=lh,
        rh=rh,
        lh_angles=get_hand_angles(lh),
        rh_angles=get_hand_angles(rh),
        pose_mean=pose_mean,
        face_mean=face_mean,
        lh_mean=lh_mean,
        rh_mean=rh_mean,
    )
