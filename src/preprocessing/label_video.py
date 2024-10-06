import cv2
import mediapipe as mp
import os
import csv


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Directory to store processed data
output_dir = 'E:\Projects\Sign Language Project\SignSpeak\data\processed'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of signs to label
# Extend this list as needed
signs = ['hello', 'thank_you', 'please', 'yes', 'no']


def label_frames(video_source=0, save_file='labeled_hand_landmarks.csv'):
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Open CSV file for writing the landmarks and labels
    with open(os.path.join(output_dir, save_file), mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        # Write the header (landmarks + label)
        header = [f"x{i}, y{i}, z{i}" for i in range(21)]
        header.append("label")
        csv_writer.writerow(header)

        # Initialize the MediaPipe hands model
        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            current_sign = None
            print("Press the corresponding key to assign a label to the current frame:")
            for idx, sign in enumerate(signs):
                print(f"Press '{idx}' for '{sign}'")

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Could not read frame.")
                    break

                # Convert the BGR frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame to detect hands
                result = hands.process(frame_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Collect landmarks
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend(
                                [landmark.x, landmark.y, landmark.z])

                        # Write landmarks and label to CSV
                        if current_sign is not None:
                            csv_writer.writerow(landmarks + [current_sign])
                            current_sign = None  # Reset after labeling

                        # Draw landmarks on the frame
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the frame
                cv2.imshow('Frame Labeling - Press keys to label', frame)

                # Capture key press
                key = cv2.waitKey(1) & 0xFF

                # Assign label based on key press
                if key >= ord('0') and key < ord(str(len(signs))):
                    label_idx = key - ord('0')
                    if label_idx < len(signs):
                        current_sign = signs[label_idx]
                        print(
                            f"Label '{current_sign}' assigned to the next detected frame.")
                elif key == ord('q'):
                    break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    label_frames(video_source=0, save_file='labeled_hand_landmarks.csv')
