import requests

headers = { "content-type": "application/json" }

response = requests.post("http://localhost:5000/predict", headers=headers, data={ "filename": "0001.wav", "word": "have" })

print(response.text)