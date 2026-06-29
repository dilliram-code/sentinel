from insightface.app import FaceAnalysis
import numpy as np

app = FaceAnalysis()
app.prepare(ctx_id=0)


# get the face embedding if it exists
def get_embedding(face_image):
  faces = app.get(face_image)
  if len(faces) == 0:
    return None
  
  return faces[0].embedding