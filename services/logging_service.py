import json
from datetime import datetime

def save_visit(person):
    record = {
        "id": person["id"],
        "name": person["name"],
        "department": person["department"],
        "position": person["position"],
        "time": str(datetime.now())
    }
    
    with open("database/visits.json", "a") as file:
        file.write(json.dumps(record))
        file.write("\n")