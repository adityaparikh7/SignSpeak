# import cv2
# import mediapipe as mp
# import os

# # Initialize MediaPipe Hands and OpenCV
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False,
#                        max_num_hands=1, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # Create a directory to save captured images
# save_dir = 'E:\Projects\Sign Language Project\SignSpeak\data/raw'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # Open webcam
# cap = cv2.VideoCapture(0)
# image_counter = 0

# print("Press 'q' to quit and 's' to save an image with a label.")

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

#     # Flip the image horizontally for a later selfie-view display
#     image = cv2.flip(image, 1)

#     # Convert the image to RGB as MediaPipe uses RGB format
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Process the image to detect hands
#     results = hands.process(image_rgb)

#     # Draw hand annotations on the image
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # Display the image with hand landmarks
#     cv2.imshow('MediaPipe Hands', image)

#     # Capture key press
#     key = cv2.waitKey(1)

#     if key & 0xFF == ord('s'):
#         # Save the image when 's' is pressed
#         image_counter += 1
#         image_path = os.path.join(save_dir, f'image_{image_counter}.png')
#         cv2.imwrite(image_path, image)
#         print(f"Saved: {image_path}")

#     if key & 0xFF == ord('q'):
#         # Quit the loop when 'q' is pressed
#         break

# # Release the webcam and close windows
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create a directory to save captured images
# save_dir = 'captured_images'
save_dir = 'E:\Projects\Sign Language Project\SignSpeak\data/raw/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Open webcam
cap = cv2.VideoCapture(0)
image_counter = 0

print("Press 'q' to quit and 's' to save a 300x300 image with a label.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Define the 300x300 window at the center
    height, width, _ = image.shape
    x_center = width // 2
    y_center = height // 2

    # Define the top-left and bottom-right coordinates for the 300x300 window
    top_left_x = x_center - 150
    top_left_y = y_center - 150
    bottom_right_x = x_center + 150
    bottom_right_y = y_center + 150

    # Crop the 300x300 area from the center
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Convert the image to RGB as MediaPipe uses RGB format
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Process the cropped image to detect hands
    results = hands.process(cropped_image_rgb)

    # Draw hand landmarks on the cropped image if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                cropped_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the cropped 300x300 image
    cv2.imshow('300x300 Window - MediaPipe Hands', cropped_image)

    # Capture key press
    key = cv2.waitKey(1)

    if key & 0xFF == ord('s'):
        # Save the cropped 300x300 image when 's' is pressed
        image_counter += 1
        image_path = os.path.join(save_dir, f'image_{image_counter}.png')
        cv2.imwrite(image_path, cropped_image)
        print(f"Saved: {image_path}")

    if key & 0xFF == ord('q'):
        # Quit the loop when 'q' is pressed
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
