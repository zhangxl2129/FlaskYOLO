import requests

url = 'http://127.0.0.1:5000/task/e2a1611b-8673-41ee-a1f5-4f6d18d89d87'

response = requests.get(url)

print(response.json())
