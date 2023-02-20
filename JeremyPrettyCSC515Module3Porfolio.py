# Jeremy Pretty
# CSC 515 Module 3 Critical Thinking Assignment
import cv2
import dlib
import numpy as np
import os

# Getting the file path for both images
subject1 = os.path.join(os.path.dirname(__file__), 'image1.jpeg')
subject2 = os.path.join(os.path.dirname(__file__), 'image2.jpeg')

# Load the original images for subject 1 and subject 2
img1 = cv2.imread(subject1)
img2 = cv2.imread(subject2)

# Convert the original images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Apply illumination correction
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray1 = clahe.apply(gray1)
gray2 = clahe.apply(gray2)

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Detect the face landmarks for subject 1 and subject 2
faces1 = detector(gray1, 1)
faces2 = detector(gray2, 1)

# Define a fixed size for the cropped face image
face_width = 200
face_height = 200

# Process the first image
if len(faces1) > 0:
    face = faces1[0]

    # Get the facial landmarks
    landmarks = predictor(gray1, face)

    # Compute the angle between the eye centers
    eye_left = (landmarks.part(36).x, landmarks.part(36).y)
    eye_right = (landmarks.part(45).x, landmarks.part(45).y)
    angle = np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]) * 180 / np.pi

    # Rotate the image based on the angle between the eye centers
    rows, cols = gray1.shape
    rot_mat = cv2.getRotationMatrix2D(eye_left, angle, 1.0)
    rotated = cv2.warpAffine(gray1, rot_mat, (cols, rows), flags=cv2.INTER_LINEAR)

    # Define a new bounding box around the face region
    left = max(0, landmarks.part(0).x - 50)
    top = max(0, landmarks.part(19).y - 50)
    right = min(cols, landmarks.part(16).x + 50)
    bottom = min(rows, landmarks.part(8).y + 50)

    # Crop the image to the new bounding box
    cropped = rotated[top:bottom, left:right]

    # Resize the cropped image to a fixed size
    resized = cv2.resize(cropped, (face_width, face_height))

    # Save the processed image to a file
    cv2.imwrite("subject1_processed.jpg", resized)

# Process the second image
if len(faces2) > 0:
    face = faces2[0]

    # Get the facial landmarks
    landmarks = predictor(gray2, face)

    # Compute the angle between the eye centers
    eye_left = (landmarks.part(36).x, landmarks.part(36).y)
    eye_right = (landmarks.part(45).x, landmarks.part(45).y)
    angle = np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]) * 180 / np.pi

    # Rotate the image based on the angle between the eye centers
    rows, cols = gray2.shape
    rot_mat = cv2.getRotationMatrix2D(eye_left, angle, 1.0)
    rotated = cv2.warpAffine(gray2, rot_mat, (cols, rows), flags=cv2.INTER_LINEAR)

    # Define a new bounding box around the face region
    left = max(0, landmarks.part(0).x - 50)
    top = max(0, landmarks.part(19).y - 50)
    right = min(cols, landmarks.part(16).x + 50)
    bottom = min(rows, landmarks.part(8).y + 50)

    # Crop the image to the new bounding box
    cropped = rotated[top:bottom, left:right]

    # Resize the cropped image to a fixed size
    resized = cv2.resize(cropped, (face_width, face_height))

    # Save the processed image to a file
    cv2.imwrite("subject2_processed.jpg", resized)
