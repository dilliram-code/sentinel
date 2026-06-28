import cv2 

def initialize_camera(source):
  
  cap = cv2.VideoCapture(source)
  
  if not cap.isOpened():
    raise Exception("Camera cannot be opened 😕")
  
  return cap