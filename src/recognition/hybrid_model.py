# import tensorflow as tf
# from tensorflow.keras import layers, models

# # CNN Model for extracting spatial features from images
# def build_cnn_model(input_shape):
#     cnn_model = models.Sequential()

#     # Convolutional layers to extract image features
#     cnn_model.add(layers.Conv2D(
#         32, (3, 3), activation='relu', input_shape=input_shape))
#     cnn_model.add(layers.MaxPooling2D((2, 2)))

#     cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     cnn_model.add(layers.MaxPooling2D((2, 2)))

#     cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     cnn_model.add(layers.MaxPooling2D((2, 2)))

#     # Flatten the feature maps into a 1D vector
#     cnn_model.add(layers.Flatten())
#     cnn_model.add(layers.Dense(128, activation='relu'))

#     return cnn_model

# # LSTM-CNN Hybrid Model
# def build_hybrid_lstm_cnn_model(input_shape, timesteps, num_classes):
#     # CNN for feature extraction (applied on each frame in the sequence)
#     cnn_model = build_cnn_model(input_shape)

#     # LSTM input shape is (timesteps, feature_vector_size)
#     lstm_model = models.Sequential()

#     # Repeat CNN model output for LSTM (one set of features per timestep)
#     lstm_model.add(layers.TimeDistributed(
#         cnn_model, input_shape=(timesteps, *input_shape)))

#     # LSTM layer for sequence learning
#     lstm_model.add(layers.LSTM(64, return_sequences=True))
#     lstm_model.add(layers.LSTM(64))

#     # Fully connected output layer
#     lstm_model.add(layers.Dense(num_classes, activation='softmax'))

#     return lstm_model


# # Parameters (example)
# # input_shape = (64, 64, 3)  # Image size (64x64 RGB images)
# input_shape = (300, 300, 3)  
# timesteps = 30  # Number of image frames in a sequence
# num_classes = 3  # Number of possible sign language gestures

# # Build the hybrid model
# hybrid_model = build_hybrid_lstm_cnn_model(input_shape, timesteps, num_classes)

# # Compile the model
# hybrid_model.compile(
#     optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Display model summary
# hybrid_model.summary()

# train_dataset = "E:\Projects\Sign Language Project\SignSpeak\data/raw"
# test_dataset = "E:\Projects\Sign Language Project\ASL/test"
# # train the model
# hybrid_model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# # Evaluate the model
# loss, accuracy = hybrid_model.evaluate(test_dataset)

# print(f'Test accuracy: {accuracy * 100:.2f}%')

# # Save the model
# hybrid_model.save('hybrid_model.h5')




import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


# Load and preprocess images
def load_images(image_folder):
    images = []
    labels = []
    for label in os.listdir(image_folder):
        label_folder = os.path.join(image_folder, label)
        if os.path.isdir(label_folder):
            for image_name in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (64, 64))
                    images.append(image)
                    labels.append(label)
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)
    return images, labels

# Encode labels
def encode_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    one_hot_encoded = to_categorical(integer_encoded)
    return one_hot_encoded, label_encoder


# Load images and labels
# image_folder = 'E:\Projects\Sign Language Project\SignSpeak\data/raw'
image_folder = 'E:\Projects\Sign Language Project\ASL/train_reduced500'
images, labels = load_images(image_folder)
labels, label_encoder = encode_labels(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape data for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, 64, 64, 3))
X_val = X_val.reshape((X_val.shape[0], 1, 64, 64, 3))

# Define the hybrid CNN-LSTM model
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 64, 64, 3)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32)

# Save the trained model
# model.save('sign_language_recognition_model.h5')
model.save('E:\Projects\Sign Language Project\SignSpeak\src/recognition\models\image models/ASL_reduced_500_30.h5')
