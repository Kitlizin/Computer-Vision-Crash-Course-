import numpy as np
import cv2
import sys
from read_images import read_images

def face_rec():
    names = ['Keith', 'Marc', 'Monic']  

    data_path = r'C:\Users\Keith\Documents\Activity 7. Performing Face Recognition\images'  
    face_images, face_labels = read_images(data_path)
    face_labels = np.asarray(face_labels, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_images, face_labels)

    cam = cv2.VideoCapture(0) 
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector.detectMultiScale(grayscale_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_region = grayscale_frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_region, (200, 200))

            try:
                predicted_label, confidence = recognizer.predict(resized_face)
                recognized_name = names[predicted_label] if predicted_label < len(names) else "Unknown"
                cv2.putText(frame, f"{recognized_name}, {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except:
                continue

        cv2.imshow("LBPH Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()
