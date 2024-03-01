import os
import cv2
import numpy as np

# Create LBPH face recognizer directly
recognizer=cv2.face.LBPHFaceRecognizer_create()

# Define the path where the dataset is located
path = "dataset"

# Function to get images and corresponding IDs from the dataset
def get_images_with_id(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for image_path in image_paths:
        # Open the image and convert it to grayscale
        face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Extract the ID from the image filename
        id = int(os.path.split(image_path)[-1].split(".")[1])
        # Print the ID (for debugging purposes)
        print(id)
        # Append the face image and its corresponding ID to the lists
        faces.append(face_img)
        ids.append(id)
        # Display the image (for debugging purposes)
        cv2.imshow("Training", face_img)
        cv2.waitKey(100)

    return np.array(ids), faces

# Get IDs and corresponding faces from the dataset
ids, faces = get_images_with_id(path)

# Train the recognizer with the images and IDs
recognizer.train(faces, ids)

# Save the trained model
recognizer.save("recognizer/trainingdata.yml")

# Close the OpenCV windows
cv2.destroyAllWindows()
