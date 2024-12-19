import base64
import os
from flask import Flask, request, jsonify
from ultralytics.YOLOTest import YOLOTest
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # 启用 CORS

# 设置保存上传文件和预测图像的目录
UPLOAD_FOLDER = r'data/test/images'
PREDICT_FOLDER = r'data/test/predicts'

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

# 输入json： { images{image:}, detectType="string" }
@app.route('/predict/', methods=['POST'])
def predict():
    # 获取请求的 JSON 数据
    data = request.get_json()
    # 检查是否提供了图像
    if not data or 'images' not in data:
        return jsonify({'error': 'No images provided'}), 400

    # 获取 detectType 参数
    detect_type = data.get('detectType', 'default')  # 默认值为 'default'

    # 构造模型文件路径
    model_path = os.path.join('data/mode', f'{detect_type}.pt')

    # 如果指定的模型文件不存在，使用默认模型路径
    if not os.path.exists(model_path):
        model_path = 'data/mode/default.pt'  # 默认模型路径

    # 初始化YOLOTest类
    yolo_test = YOLOTest(model_path)

    images_data = data['images']  # 获取图像数据
    result_images = []

    images_bytes = []  # 存储所有图像的字节流
    for idx, img_data in enumerate(images_data):
        try:
            # 解码Base64图像数据
            image_bytes = base64.b64decode(img_data['image'])

            # 保存上传的原始图像到指定目录
            upload_filename = os.path.join(UPLOAD_FOLDER, f'{idx:06d}.jpg')  # 使用 6 位数的文件名
            with open(upload_filename, 'wb') as f:
                f.write(image_bytes)

            images_bytes.append(image_bytes)

        except Exception as e:
            result_images.append({'error': f"Error decoding image: {str(e)}"})
            continue

    # 处理所有图像
    detected_images = yolo_test.process_image(images_bytes)

    for idx, detected_image in enumerate(detected_images):
        if detected_image is not None:
            try:
                # 保存预测后的图像到指定目录
                predict_filename = os.path.join(PREDICT_FOLDER, f'predicted_{idx:06d}.png')  # 使用 6 位数的文件名
                detected_image.save(predict_filename)

                # 将带框的图像转换为Base64编码
                detected_image_base64 = yolo_test.image_to_base64(detected_image)
                result_images.append({'image': detected_image_base64})
            except Exception as e:
                result_images.append({'error': f"Error processing predicted image: {str(e)}"})
        else:
            result_images.append({'error': 'Unable to process image'})

    return jsonify({'images': result_images})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

