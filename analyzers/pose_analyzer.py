import cv2
import mediapipe as mp

class PoseAnalyzer:
    def __init__(self):
        # This will now work once 'Has solutions' is True
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def analyze(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        data = {'detected': False, 'landmarks': None}
        if results.pose_landmarks:
            data['detected'] = True
            data['landmarks'] = results.pose_landmarks
        return data

    def draw(self, frame, data):
        if data['detected']:
            self.mp_draw.draw_landmarks(
                frame, data['landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2)
            )
        return frame