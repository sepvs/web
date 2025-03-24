
from flask import Flask, render_template, Response, stream_with_context
import cv2
import mediapipe as mp
import time

def crear_app():

    app = Flask(__name__)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    def is_gesture_a(landmarks):
        return (landmarks[8].y > landmarks[5].y and
                landmarks[12].y > landmarks[9].y and
                landmarks[16].y > landmarks[13].y and
                landmarks[20].y > landmarks[17].y)

    def is_gesture_e(landmarks):
        return (landmarks[8].y > landmarks[6].y and landmarks[8].y < landmarks[5].y and
                landmarks[12].y > landmarks[10].y and landmarks[12].y < landmarks[9].y and
                landmarks[16].y > landmarks[14].y and landmarks[16].y < landmarks[13].y and
                landmarks[20].y > landmarks[18].y and landmarks[20].y < landmarks[17].y)

    def is_gesture_i(landmarks):
        return (landmarks[8].y > landmarks[5].y and landmarks[12].y > landmarks[9].y and
                landmarks[16].y > landmarks[13].y and landmarks[20].y < landmarks[19].y)

    def is_gesture_o(landmarks):
        return (landmarks[8].y > landmarks[4].y and landmarks[12].y > landmarks[4].y and
                landmarks[16].y > landmarks[4].y and landmarks[20].y > landmarks[4].y)

    def is_gesture_u(landmarks):
        return (landmarks[8].y < landmarks[5].y and landmarks[12].y > landmarks[9].y and
                landmarks[16].y > landmarks[13].y and landmarks[20].y < landmarks[17].y)

    def generate_frames():
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                detected_letter = ""
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        if is_gesture_a(hand_landmarks.landmark):
                            detected_letter = "A"
                        elif is_gesture_e(hand_landmarks.landmark):
                            detected_letter = "E"
                        elif is_gesture_i(hand_landmarks.landmark):
                            detected_letter = "I"
                        elif is_gesture_o(hand_landmarks.landmark):
                            detected_letter = "O"
                        elif is_gesture_u(hand_landmarks.landmark):
                            detected_letter = "U"

                if detected_letter:
                    cv2.putText(frame, detected_letter, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    def detect_gestures():
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                detected_letter = ""
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if is_gesture_a(hand_landmarks.landmark):
                            detected_letter = "A"
                        elif is_gesture_e(hand_landmarks.landmark):
                            detected_letter = "E"
                        elif is_gesture_i(hand_landmarks.landmark):
                            detected_letter = "I"
                        elif is_gesture_o(hand_landmarks.landmark):
                            detected_letter = "O"
                        elif is_gesture_u(hand_landmarks.landmark):
                            detected_letter = "U"

                yield f"data: {detected_letter}\n\n"

        cap.release()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/letter_feed')
    def letter_feed():
        return Response(stream_with_context(detect_gestures()), mimetype='text/event-stream')

    return app


if __name__ == '_main_':
    app = crear_app()
    app.run()