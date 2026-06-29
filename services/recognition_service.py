import json
import numpy as np 

def cosine_similarity(a,b):
  
  # return the cosine value
  return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

