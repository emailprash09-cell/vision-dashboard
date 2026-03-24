import cv2

def draw_panel(frame, x, y, w, h, title, lines, accent):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + 3), accent, -1)
    cv2.putText(frame, title, (x + 8, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, accent, 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 8, y + 36 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
    return frame

def draw_dashboard(frame, face_data, eye_data, hand_data, pose_data):
    h, w = frame.shape[:2]

    # ── Face panel (top-left) ──────────────────────────────────
    if face_data:
        region = face_data.get('region', {})
        if region:
            rx, ry = region.get('x', 0), region.get('y', 0)
            rw, rh = region.get('w', 0), region.get('h', 0)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (100, 255, 100), 2)
        face_lines = [
            f"Age     : ~{face_data.get('age', '?')}",
            f"Gender  : {face_data.get('gender', '?')}",
            f"Emotion : {face_data.get('emotion', '?')}",
            f"Ethnicity: {str(face_data.get('race', '?'))[:12]}",
        ]
        draw_panel(frame, 10, 10, 210, 115, 'FACE ANALYSIS', face_lines, (100, 255, 100))
    else:
        draw_panel(frame, 10, 10, 210, 52, 'FACE ANALYSIS', ['Detecting...'], (100, 255, 100))

    # ── Eye / drowsiness panel (top-right) ────────────────────
    status_color = (0, 80, 255) if eye_data.get('drowsy') else (100, 255, 200)
    eye_lines = [
        f"Status : {eye_data.get('status', 'N/A')}",
        f"EAR    : {eye_data.get('ear', 0):.3f}  (thresh 0.25)",
        f"Blinks : {eye_data.get('blink_count', 0)}",
    ]
    draw_panel(frame, w - 240, 10, 230, 90, 'EYE / DROWSINESS', eye_lines, status_color)

    # ── Hand panel (bottom-left) ──────────────────────────────
    hand_lines = [f"Hands detected : {hand_data.get('count', 0)}"]
    for hd in hand_data.get('handedness', []):
        hand_lines.append(f"  -> {hd.classification[0].label} hand")
    draw_panel(frame, 10, h - 110, 210, 90, 'HAND TRACKER', hand_lines, (255, 160, 50))

    # ── Pose panel (bottom-right) ─────────────────────────────
    pose_status = 'Body detected' if pose_data.get('detected') else 'No body found'
    draw_panel(frame, w - 240, h - 70, 230, 55, 'POSE TRACKER', [pose_status], (80, 180, 255))

    # ── Footer ────────────────────────────────────────────────
    cv2.putText(frame, 'AI Vision Dashboard  |  Press Q to quit',
                (w // 2 - 165, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1, cv2.LINE_AA)

    return frame