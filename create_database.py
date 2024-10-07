import cv2
import sys
import numpy
import os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  # All the faces data will be present in this folder

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)  # '0' is used for the default webcam

# Function to capture images for a person
def capture_images_for_person(person_name):
    path = os.path.join(datasets, person_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    
    (width, height) = (130, 100)  # defining the size of images 
    count = 1  # Initialize count for number of images
    
    print(f"Capturing images for {person_name}. Please look at the camera...")
    while count < 31: 
        _, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite(f'{path}/{count}.jpg', face_resize)  # Save the face image
            count += 1
            
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(150)
        if key == 27:  # Exit if 'ESC' is pressed
            break

# Main loop to input multiple names
while True:
    person_name = input("Enter the name of the person to capture images (or type 'exit' to quit): ")
    if person_name.lower() == 'exit':
        break
    capture_images_for_person(person_name)

# Release the webcam and destroy all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
