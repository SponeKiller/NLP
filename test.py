import json
data = []
with open('data.jsonl', 'r') as file:
    for line in file:
        data.append(json.loads(line))
        
        


print(data[1]["messages"][0]["content"])


