import cv2
from analyzers.face_analyzer import FaceAnalyzer
from analyzers.pose_analyzer import PoseAnalyzer
from analyzers.hand_analyzer import HandAnalyzer
from analyzers.eye_analyzer  import EyeAnalyzer
from utils.overlay           import draw_dashboard

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_analyzer = FaceAnalyzer()
    pose_analyzer = PoseAnalyzer()
    hand_analyzer = HandAnalyzer()
    eye_analyzer  = EyeAnalyzer()

    frame_count = 0
    print("Starting AI Vision Dashboard... press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read from webcam.")
            break

        frame_count += 1

        # Fast analyzers — every frame
        pose_data = pose_analyzer.analyze(frame)
        hand_data = hand_analyzer.analyze(frame)
        eye_data  = eye_analyzer.analyze(frame)

        # Slow analyzer (DeepFace) — every 10th frame, non-blocking
        if frame_count % 10 == 0:
            face_analyzer.analyze_async(frame)

        face_data = face_analyzer.get_latest()

        # Draw skeleton + landmarks
        frame = pose_analyzer.draw(frame, pose_data)
        frame = hand_analyzer.draw(frame, hand_data)

        # Draw HUD panels
        frame = draw_dashboard(frame, face_data, eye_data, hand_data, pose_data)

        cv2.imshow('AI Vision Dashboard', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()