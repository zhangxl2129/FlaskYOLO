import requests
import base64

# 读取图片并进行Base64编码
def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 发送POST请求
url = 'http://192.168.31.166:5000/predict/'

# json
images = [
    {"image": image_to_base64('E:/我的比赛/第六届校园人工智能大赛/国赛/DataB/Img/000001.jpg')},
    {"image": image_to_base64('E:/我的比赛/第六届校园人工智能大赛/国赛/DataB/Img/000061.jpg')}
]

response = requests.post(url, json={'images': images})

# 输出响应结果
print(response.status_code)
print(response.json())
