import json
import numpy as np 

# calculate and get the cosine value
def cosine_similarity(a,b):
  return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

# identify person
def identify_person(embedding, database_path,threshold):
  with open(database_path) as file:
    database = json.load(file)
    
  best_score = 0
  best_person = None 
  
  for person in database:
    score = cosine_similarity(embedding, np.array(person['embedding']))
    if score > best_score:
      best_score = score 
      best_person = person
  if best_score > threshold:
    return best_person  
  
  return None