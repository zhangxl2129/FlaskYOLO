# FlaskYOLO

FlaskYOLO 是一个基于 Flask 的轻量级 API，用于集成 YOLO（You Only Look Once）目标检测模型。它提供了训练、测试和检测的接口，方便快速构建和部署机器学习应用。

功能展示
--
![show](https://github.com/user-attachments/assets/40dc5b9e-765a-4f7f-9488-eb98befdc095)



## 功能特点

- **训练 YOLO 模型：** 通过 API 接口对自定义数据集进行训练。
- **测试 YOLO 模型：** 评估 YOLO 模型性能，获取详细的测试指标。
- **目标检测：** 对上传的图片进行目标检测，返回 JSON 格式的检测结果。
- **模块化设计：** 将 YOLO 的训练、测试和推理功能独立封装，便于扩展。
- **轻量级框架：** 基于 Flask 实现，简单易用。

---

## 先决条件

在安装 FlaskYOLO 之前，请确保已安装以下环境：

- **Python 3.8+**
- **Pip**
- **虚拟环境（可选）**
- **YOLOv11**

另外，通过以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

---

## 安装步骤

1. **克隆仓库：**
   ```bash
   git clone https://github.com/yourusername/flaskyolo.git
   cd flaskyolo
   ```

2. **创建虚拟环境（可选）：**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

4. **运行应用程序：**
   ```bash
   python app.py
   ```

默认情况下，Flask 应用程序将在 `http://127.0.0.1:5000` 上运行。

---

## API 接口

### 1. **目标检测**

- **接口地址：** `POST /predict/`
- **描述：** 对指定图片进行目标检测。
- **请求参数：**
  ```json
  {
    "images": [
      {"image": "<Base64编码的图片数据>"},
      {"image": "<Base64编码的图片数据>"}
    ],
    "detectType": "string" // 可选，指定检测类型，默认为 'default'
  }
  ```
- **返回示例：**
  ```json
  {
    "images": [
      {"image": "<带框的Base64编码图片数据>"},
      {"image": "<带框的Base64编码图片数据>"}
    ]
  }
  ```

#### 使用示例（Python 客户端代码）：

```python
import requests
import base64

def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

url = 'http://127.0.0.1:5000/predict/'

images = [
    {"image": image_to_base64('path/to/image1.jpg')},
    {"image": image_to_base64('path/to/image2.jpg')}
]

response = requests.post(url, json={'images': images, 'detectType': 'default'})

print(response.status_code)
print(response.json())
```

### 2. **训练 YOLO 模型**

- **接口地址：** `POST /train/`
- **描述：** 开始使用指定数据集训练 YOLO 模型。
- **请求参数：** 上传 `.zip` 格式的数据集，其中需包含 `data.yaml` 文件。
- **返回示例：**
  ```json
  {
    "message": "Training started",
    "task_id": "unique_task_id"
  }
  ```

#### 使用示例：

使用 Postman 或其他工具上传 `.zip` 文件：
- URL: `http://127.0.0.1:5000/train/`
- 方法: `POST`
- Body: 选择 `form-data`，键名为 `file`，值为 `.zip` 文件。

### 3. **查询训练任务状态**

- **接口地址：** `GET /task/<task_id>`
- **描述：** 获取训练任务的状态信息。
- **返回示例：**
  ```json
  {
    "task_id": "unique_task_id",
    "status": "running",
    "result": null,
    "finished_at": null,
    "created_at": "2024-12-18T12:00:00"
  }
  ```

---

## 项目结构

```plaintext
flaskyolo/
|-- app.py                # Flask 应用程序入口
|-- requirements.txt      # Python 依赖文件
|-- data/                 # 数据目录
    |-- train/            # 训练数据集
    |-- test/             # 测试数据集
    |-- model/            # 保存的模型文件
|-- utils/                # 工具模块
    |-- yolo_train.py     # 训练模块
    |-- yolo_test.py      # 测试模块
    |-- yolo_detect.py    # 检测模块
```

---

## 自定义

您可以根据需求修改以下模块：

- **`utils/yolo_train.py`：** 调整训练逻辑和超参数。
- **`utils/yolo_test.py`：** 优化模型评估逻辑。
- **`utils/yolo_detect.py`：** 修改目标检测的具体实现。

如果对这些模块进行了修改，请确保同步更新 Flask 的接口逻辑。

---

## 测试

您可以使用以下工具测试 API 接口：

- **Postman**
- **curl**
- **Swagger UI（如已集成）**

使用 `curl` 示例：

```bash
curl -X POST http://127.0.0.1:5000/predict/ \
-H "Content-Type: application/json" \
-d '{
  "images": [
    {"image": "<Base64编码的图片数据>"}
  ],
  "detectType": "default"
}'
```

---

## 未来改进

- 完善前端界面，提升用户体验。
- 扩展 API 功能，支持模型转换（如 TensorFlow、ONNX）。
- 增加 API 接口的认证与授权功能。

---

## 开源协议

此项目基于 MIT 协议开源。详见 `LICENSE` 文件。

---

## 联系方式

如有问题或需要支持，请联系：

- **作者：** Zhang XiaoLong
- **邮箱：** zhangxl2129@gmail.com
- **GitHub：** [(https://github.com/zhangxl2129)]((https://github.com/zhangxl2129)e)

