# Jeremy Pretty
# CSC 515 Module 3 Critical Thinking Assignment
import cv2
import os

# Fine the path to teh photos
img_path_1 = os.path.join(os.path.dirname(__file__), 'image1.jpeg')
img_path_2 = os.path.join(os.path.dirname(__file__), 'image2.jpeg')

# The path to the frontal face needed
haras_frontal_face = os.path.join(os.path.dirname(__file__), '/Users/jeremypretty/opt/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Load the images
img_1 = cv2.imread(img_path_1)
img_2 = cv2.imread(img_path_2)

# Convert the images to grayscale
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# Adjust the illumination of the images
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_1 = clahe.apply(gray_1)
gray_2 = clahe.apply(gray_2)

# Detect the faces in the images
face_cascade = cv2.CascadeClassifier(haras_frontal_face)
faces_1 = face_cascade.detectMultiScale(gray_1, scaleFactor=1.1, minNeighbors=5)
faces_2 = face_cascade.detectMultiScale(gray_2, scaleFactor=1.1, minNeighbors=5)

# Process each face
for (x, y, w, h) in faces_1:
    # Rotate the image based on the angle between the eye centers
    eyes_1 = cv2.HoughCircles(gray_1[y:y+h, x:x+w], cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if eyes_1 is not None:
        eyes_1 = eyes_1[0, :]
        if len(eyes_1) == 2:
            (x1, y1, r1), (x2, y2, r2) = eyes_1
            angle = (180 / 3.14159) * (-(y2 - y1) / (x2 - x1))
            M = cv2.getRotationMatrix2D((x + w/2, y + h/2), angle, 1)
            gray_1 = cv2.warpAffine(gray_1, M, (img_1.shape[1], img_1.shape[0]))
    
    # Crop and scale the image to create a new bounding box for the face
    roi_gray = gray_1[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (100, 100))
    
    # Save the processed image
    cv2.imwrite("subject_1_processed.jpg", roi_gray)


for (x, y, w, h) in faces_2:
    # Rotate the image based on the angle between the eye centers
    eyes_2 = cv2.HoughCircles(gray_2[y:y+h, x:x+w], cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if eyes_2 is not None:
        eyes_2 = eyes_2[0, :]
        if len(eyes_2) == 2:
            (x1, y1, r1), (x2, y2, r2) = eyes_2
            angle = (180 / 3.14159) * (-(y2 - y1) / (x2 - x1))
            M = cv2.getRotationMatrix2D((x + w/2, y+ h/2), angle, 1)
            gray_2 = cv2.warpAffine(gray_2, M, (img_2.shape[1], img_2.shape[0]))

    # Crop and scale the image to create a new bounding box for the face
    roi_gray = gray_2[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (100, 100))

    # Save the processed image
    cv2.imwrite("subject_2_processed.jpg", roi_gray)

