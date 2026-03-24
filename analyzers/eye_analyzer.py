import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# MediaPipe Face Mesh landmark indices for each eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

EAR_THRESHOLD  = 0.25   # below this = eye closed
CONSEC_FRAMES  = 15     # frames closed before "drowsy" alert

def compute_ear(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C + 1e-6)

class EyeAnalyzer:
    def __init__(self):
        self.mp_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.blink_count  = 0
        self.consec_count = 0
        self.ear_history  = deque(maxlen=10)

    def analyze(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        data = {
            'detected':    False,
            'ear':         0.0,
            'blink_count': self.blink_count,
            'drowsy':      False,
            'status':      'No face detected'
        }

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            left_ear  = compute_ear(lm, LEFT_EYE,  w, h)
            right_ear = compute_ear(lm, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(ear)
            avg_ear = float(np.mean(self.ear_history))

            data['detected'] = True
            data['ear']      = round(avg_ear, 3)

            if avg_ear < EAR_THRESHOLD:
                self.consec_count += 1
                if self.consec_count >= CONSEC_FRAMES:
                    data['drowsy'] = True
                    data['status'] = 'DROWSY - ALERT!'
                else:
                    data['status'] = 'Eyes closing...'
            else:
                if 3 <= self.consec_count < CONSEC_FRAMES:
                    self.blink_count += 1
                self.consec_count  = 0
                data['status']     = 'Alert'
                data['blink_count'] = self.blink_count

        return data