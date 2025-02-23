import numpy as np
import os
import cv2

def read_images(dataset_path):
    images, labels = [], []
    person_id = 0

    for person_name in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

            if image is None:
                print(f"Could not read {image_path}, skipping.")
                continue

            image = cv2.resize(image, (200, 200))  
            images.append(np.asarray(image, dtype=np.uint8))
            labels.append(person_id)

        person_id += 1

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    dataset_directory = r'C:\Users\Keith\Documents\Activity 7. Performing Face Recognition\images'
    face_images, face_labels = read_images(dataset_directory)