import time
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cv2

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Configuration
offset = 20
imgSize = 300
folder = "/Users/raheel/Developer/LYPROJECT/SignSpeak/data/raw/U"
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    hands, img = detector.findHands(img)

    if hands:
        # Get the first detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image for resizing
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Calculate aspect ratio and resize accordingly
        aspectRatio = h / w
        try:
            if aspectRatio > 1:
                # Height is greater, fit to height
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                # Width is greater, fit to width
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Show cropped and resized images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print("Error processing image:", e)

    # Show original image with bounding box
    cv2.imshow("Image", img)
    
    # Key press events
    key = cv2.waitKey(1)
    if key == ord("s"):
        # Save image when 's' key is pressed
        counter += 1
        imgPath = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(imgPath, imgWhite)
        print(f"Saved {imgPath}, Count: {counter}")
    elif key == ord('q'):
        # Exit loop when 'q' key is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()