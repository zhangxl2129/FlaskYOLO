import base64
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics.mode import YOLO


class YOLOTest:
    def __init__(self, model_path):
        # 加载YOLO模型
        self.model = YOLO(model_path)

    def perform_inference(self, image_path):
        """
        对单张图片执行YOLO推理，返回带检测框的图像以及相关的检测框数据
        """
        results = self.model.predict(source=image_path, save=False)

        if results and len(results) > 0:
            result = results[0]  # 获取第一个检测结果
            boxes = result.boxes  # 获取检测框
            boxes_xywh = boxes.xywh.cpu().numpy()  # 获取框的坐标

            # 获取类ID
            class_ids = boxes.cls.cpu().numpy()
            class_ids = class_ids.astype(np.int32)

            # 获取带框图像（这里返回的是一个 numpy.ndarray）
            detected_image = result.plot()

            # 将 numpy.ndarray 转换为 PIL.Image
            pil_image = Image.fromarray(detected_image)

            return boxes_xywh, class_ids, pil_image

        return None, None, None

    def process_image(self, images_bytes):
        """
        从字节流中处理多张图像，进行推理，并返回带框的图像
        """
        detected_images = []  # 存储检测后的图像

        for image_bytes in images_bytes:
            # 读取图像
            image = Image.open(BytesIO(image_bytes))

            # 执行推理
            boxes_xywh, class_ids, detected_image = self.perform_inference(image)

            if detected_image is not None:
                detected_images.append(detected_image)
            else:
                detected_images.append(None)

        return detected_images

    def image_to_base64(self, image):
        """
        将图像对象转换为Base64编码的字符串
        """
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return base64.b64encode(img_byte_arr.read()).decode('utf-8')


