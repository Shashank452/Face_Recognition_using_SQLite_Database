import cv2
import numpy as np
import sqlite3

# Load the Haar cascade classifier for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if faceDetect.empty():
    raise IOError("Unable to load the face cascade classifier XML file")

# Initialize the camera
cam = cv2.VideoCapture(0)

# Function to insert or update records in the SQLite database
def insert_or_update(Id, Name, age):
    # Connect to the SQLite database
    conn = sqlite3.connect("sqlite.db")
    # Check if the record already exists
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE ID=?", (Id,))
    isRecordExist = cursor.fetchone() is not None

    if isRecordExist:
        # Update the record if it already exists
        conn.execute("UPDATE STUDENTS SET Name=?, age=? WHERE Id=?", (Name, age, Id))
    else:
        # Insert a new record if it doesn't exist
        conn.execute("INSERT INTO STUDENTS (Id, Name, age) VALUES (?, ?, ?)", (Id, Name, age))

    # Commit changes and close connection
    conn.commit()
    conn.close()

# Take user input for ID, Name, and Age
Id = input('Enter User Id:')
Name = input('Enter User Name:')
age = input('Enter User Age:')

# Check if ID and age are integers
try:
    Id = int(Id)
    age = int(age)
except ValueError:
    print("ID and age must be integers")
    exit(1)

# Insert or update the record in the database
insert_or_update(Id, Name, age)

# Detect faces in the camera feed and save samples
sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(f"dataset/user.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum > 20:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
