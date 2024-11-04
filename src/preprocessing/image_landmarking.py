# write a program that takes a series of images and add hand landmarks to it
import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def process_images(image_paths):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}")
            continue

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(image_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save or display the image
        output_path = os.path.join('output', os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        print(f"Processed image saved to {output_path}")

# Example usage
image_folder = 'path_to_images'
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
process_images(image_paths)