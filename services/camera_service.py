import cv2 


# initialize camera
def initialize_camera(source):
  
  cap = cv2.VideoCapture(source)
  
  if not cap.isOpened():
    raise Exception("Camera cannot be opened 😕")
  
  return cap

# get frame
def get_frame(cap):
  
  success, frame = cap.read()
  
  if not success:
    return None 
  return frame 

# get release camera
def release_camera(cap):
  cap.release()