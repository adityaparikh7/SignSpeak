import time
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import platform

# Setup camera capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize Hand Detector
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300
counter = 0

# Define folder path based on operating system
if platform.system() == "Windows":
    folder = "E:/Projects/Sign Language Project/SignSpeak/data/raheel_raw/X"
else:
    folder = "/Users/raheel/Developer/LYPROJECT/SignSpeak/data/raheel_raw/B"

# Ensure folder exists
os.makedirs(folder, exist_ok=True)

print("Press 's' to save an image, or 'q' to quit.")
# Function for preprocessing image to reduce lighting dependence
def preprocess_image(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgEqualized = clahe.apply(imgGray)
    imgBlurred = cv2.GaussianBlur(imgEqualized, (5, 5), 0)
    imgProcessed = cv2.cvtColor(imgBlurred, cv2.COLOR_GRAY2BGR)
    return imgProcessed

# Main loop
try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image. Check camera connection.")
            time.sleep(1)
            continue

        # Preprocess the image to reduce lighting dependence
        img = preprocess_image(img)

        # Detect hands
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]),
                          max(0, x - offset):min(x + w + offset, img.shape[1])]
            
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
            
            # Display the processed images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", img)

        # Capture key presses
        key = cv2.waitKey(10)
        if key == ord("s"):
            counter += 1
            # Use both counter and time.time() to ensure unique filenames
            image_path = os.path.join(folder, f'Image_{counter}_{int(time.time() * 1000)}.jpg')
            try:
                cv2.imwrite(image_path, imgWhite)
                print(f"Image saved: {image_path} (Total: {counter})")
            except Exception as e:
                print(f"Failed to save image: {e}")
            time.sleep(0.1)  # Small delay to ensure saving completes
        elif key == ord("q"):
            print("Exiting program.")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Program ended.")