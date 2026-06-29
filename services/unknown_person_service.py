import cv2
import uuid
import os

def save_unknown_face(face_image, folder):
    filename = (str(uuid.uuid4()) + ".jpg")
    path = os.path.join(folder,filename)
    cv2.imwrite(path, face_image)

    return filename