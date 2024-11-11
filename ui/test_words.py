import math
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import cv2

# Initialize Video, Detector, and Classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "E:/Projects/Sign Language Project/SignSpeak/src/recognition/models/final models/ASL_BW_200_0.00001.h5",
    "E:/Projects/Sign Language Project/SignSpeak/src/recognition/models/final models/labels.txt")

# Parameters
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
recognized_word = []  # Buffer for word formation
confidence_threshold = 0.8  # Set confidence threshold for predictions
current_letter = None  # Hold the current recognized letter

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resizing and centering the image in white canvas
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        confidence = max(prediction)  # Get confidence of prediction

        # If confidence meets threshold, set current recognized letter
        if confidence > confidence_threshold:
            current_letter = labels[index]
            cv2.putText(imgOutput, f"Letter: {current_letter}", (10, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 255), 2)

        # Display the cropped hand images only when hands are detected
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the current word being formed
    word_display = ''.join(recognized_word)
    cv2.putText(imgOutput, f"Word: {word_display}", (10, 450),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # Show Images
    cv2.imshow("Image", imgOutput)

    # Handle key controls
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):  # Clear word buffer
        recognized_word = []
        print("Word buffer cleared.")
    # Add current letter to the word
    elif key & 0xFF == ord('a') and current_letter:
        recognized_word.append(current_letter)
        print(f"Added letter: {current_letter}")
    elif key & 0xFF == ord('s'):  # Save the current recognized word
        with open("recognized_words.txt", "a") as file:
            file.write(word_display + "\n")
        print(f"Saved word: {word_display}")

cap.release()
cv2.destroyAllWindows()
