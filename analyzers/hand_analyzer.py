import cv2
import mediapipe as mp

class HandAnalyzer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def analyze(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        data = {'count': 0, 'hands': [], 'handedness': []}
        if results.multi_hand_landmarks:
            data['count'] = len(results.multi_hand_landmarks)
            data['hands'] = results.multi_hand_landmarks
            data['handedness'] = results.multi_handedness or []
        return data

    def draw(self, frame, data):
        for hand_lm in data['hands']:
            self.mp_draw.draw_landmarks(
                frame, hand_lm,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255, 200, 0), thickness=2)
            )
        return frame