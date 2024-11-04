# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model('src/recognition/models/image models/ASL_reduced_100.h5')
# # model = load_model('E:\Projects\Sign Language Project\SignSpeak\src/recognition\models\image models\keras_model.h5')


# # Define parameters for frame capture
# NUM_FRAMES = 10  # Number of frames to consider for each sequence
# IMAGE_SIZE = (64, 64)  # Size used during model training
# sequence = []

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Example: Reshape or pad input_sequence to match the expected shape
# def pad_sequence(input_sequence, target_length=30):
#     current_length = input_sequence.shape[1]
#     if current_length < target_length:
#         padding = np.zeros(
#             (input_sequence.shape[0], target_length - current_length, 64, 64, 3))
#         input_sequence = np.concatenate((input_sequence, padding), axis=1)
#     return input_sequence


# def preprocess_frame(frame):
#     """
#     Preprocesses a single frame for the model input.
#     Resizes, normalizes, and converts to a suitable format.
#     """
#     # Resize the frame to the target size
#     frame = cv2.resize(frame, IMAGE_SIZE)
#     # Normalize the pixel values
#     frame = frame.astype('float32') / 255.0
#     # Convert frame to the correct shape (64, 64, 3)
#     return frame


# # Define a dictionary to map predicted class indices to actual sign language signs
# class_labels = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
#                 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N",
#                 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U",
#                 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: "0", 27: "1",
#                 28: "2"}

# # class_labels = {0: "A", 1: "B", 2: "C"}

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocess the frame
#     processed_frame = preprocess_frame(frame)
#     sequence.append(processed_frame)

#     # Keep only the last NUM_FRAMES frames in the sequence
#     if len(sequence) == NUM_FRAMES:
#         # Convert to numpy array with shape (1, NUM_FRAMES, 64, 64, 3)
#         input_sequence = np.array(sequence).reshape(
#             1, NUM_FRAMES, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

#         # Make predictions
#         input_sequence = pad_sequence(input_sequence)
#         predictions = model.predict(input_sequence)
#         predicted_class = np.argmax(predictions[0])

#         # Get the corresponding label
#         predicted_label = class_labels[predicted_class]

#         # Display the label on the video
#         cv2.putText(frame, predicted_label, (10, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # Remove the oldest frame from the sequence
#         sequence.pop(0)

#     # Display the resulting frame
#     cv2.imshow('Sign Language Recognition', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = 'E:\\Projects\\Sign Language Project\\SignSpeak\\src\\recognition\\models\\image models\\ASL_reduced_500_30.h5'
model = load_model(model_path)

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy')

# Preprocess input images


def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)  # For LSTM input shape
    return image

# Predict gesture


def predict_gesture(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]


# Capture images from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict gesture
    gesture = predict_gesture(frame)

    # Display the result
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
