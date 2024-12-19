import requests

url = 'http://127.0.0.1:5000/train/'
file_path = 'E:/Projects/YOLOV10/NEU-DET/NEU-DTC.zip'  # 压缩包路径

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print(response.json())
