import cv2
import mediapipe as mp
import os
import csv

# MediaPipe hands model for detecting hand landmarks
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Directory to store processed data
# output_dir = 'data/processed/'
output_dir = 'E:\Projects\Sign Language Project\SignSpeak\data\processed'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to extract and save hand landmarks from a video
def extract_hand_landmarks(video_source=0, save_file='hand_landmarks.csv'):
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Open CSV file for writing the landmarks
    with open(os.path.join(output_dir, save_file), mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        # Write the header (landmark indices)
        csv_writer.writerow([f"x{i}, y{i}, z{i}" for i in range(21)])

        # Initialize the MediaPipe hands model
        with mp_hands.Hands(
            static_image_mode=False,       # Enable continuous video mode
            max_num_hands=2,               # We assume one hand at a time
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Could not read frame.")
                    break

                # Convert the BGR frame to RGB (MediaPipe expects RGB input)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame to detect hands
                result = hands.process(frame_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Collect landmarks and write them to the CSV
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.append([landmark.x, landmark.y, landmark.z])

                        csv_writer.writerow(landmarks)

                        # Draw hand landmarks on the original frame for visualization
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the frame with landmarks
                cv2.imshow('Hand Landmark Extraction', frame)

                # Press 'q' to stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Function to extract and save hand and body landmarks from a video
def extract_holistic_landmarks(video_source=0, save_file='holistic_landmarks.csv'):
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Open CSV file for writing the landmarks
    with open(os.path.join(output_dir, save_file), mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        # Write the header (hand + body landmarks)
        header = []
        # Hand landmarks
        for i in range(21):
            header.extend([f"hand_x{i}", f"hand_y{i}", f"hand_z{i}"])
        # Body landmarks (e.g., 33 keypoints in MediaPipe)
        for i in range(33):
            header.extend([f"body_x{i}", f"body_y{i}", f"body_z{i}"])
        csv_writer.writerow(header)

        # Initialize the MediaPipe Holistic model
        with mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Could not read frame.")
                    break

                # Convert the BGR frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame to detect landmarks
                result = holistic.process(frame_rgb)

                # Initialize list to store all landmarks
                landmarks = []

                # Extract hand landmarks
                if result.left_hand_landmarks:
                    for lm in result.left_hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    # Placeholder for missing landmarks
                    landmarks.extend([0, 0, 0] * 21)

                if result.right_hand_landmarks:
                    for lm in result.right_hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    landmarks.extend([0, 0, 0] * 21)

                # Extract body landmarks
                if result.pose_landmarks:
                    for lm in result.pose_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                else:
                    # Placeholder for missing landmarks
                    landmarks.extend([0, 0, 0] * 33)

                # Write landmarks to CSV
                csv_writer.writerow(landmarks)

                # Draw landmarks on the frame for visualization
                mp_drawing.draw_landmarks(
                    frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Display the frame with landmarks
                cv2.imshow('Holistic Landmark Extraction', frame)

                # Press 'q' to stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_hand_landmarks(video_source=0, save_file='hand_landmarks.csv')
    # extract_holistic_landmarks(video_source=0, save_file='holistic_landmarks.csv')
