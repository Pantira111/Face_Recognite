import cv2
import numpy
import os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
video_path = 'D:\PROJECT\Face_recognite\jaijai.mp4'  # Replace with the path to your video file

print('Training...')
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            img = cv2.imread(path, 0)
            if img is not None:
                face_resize = cv2.resize(img, (130, 100))
                images.append(face_resize)
                labels.append(int(label))
        id += 1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
video_capture = cv2.VideoCapture(video_path)

while True:
    ret, im = video_capture.read()
    if not ret:
        break  # Exit if the video ends or if there is an error reading the frame

    im = cv2.flip(im, 1)  # Flip image
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))  # Resize for prediction
        prediction = model.predict(face_resize)

        if prediction[1] < 100:
            name = names[prediction[0]]
            accuracy = 100 - prediction[1]
            text = f'{name} {accuracy:.2f}%'
            # Initialize font scale
            font_scale = 0.8
            # Calculate text size to ensure it fits in the rectangle
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]

            # Adjust position to center the text above the rectangle
            text_x = x + (w - text_size[0]) // 2
            text_y = y - 10

            # Ensure the text does not exceed the rectangle width
            while text_size[0] > w and font_scale > 0.4:
                font_scale -= 0.1
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                text_x = x + (w - text_size[0]) // 2  # Update x position based on new text size

            # Draw a filled rectangle for the text background
            cv2.rectangle(im, (x, y - 40), (x + w, y), (0, 255, 0), -1)  # Green rectangle
            # Put the text on the image
            cv2.putText(im, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)  # Black text
        else:
            unknown_text = 'Unknown'
            # Initialize font scale for unknown text
            font_scale = 0.8
            text_size = cv2.getTextSize(unknown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y - 10

            # Draw a filled rectangle for the text background
            cv2.rectangle(im, (x, y - 40), (x + w, y), (0, 255, 0), -1)  # Green rectangle
            cv2.putText(im, unknown_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)  # Black text

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:  # Escape key
        break

video_capture.release()
cv2.destroyAllWindows()
