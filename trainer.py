import sys

from flask import Flask, request, jsonify
import os
import zipfile
import traceback
import yaml

from ultralytics.YOLOTrainService import YOLOTrainService
from flask_cors import CORS
# 添加环境变量，允许 OpenMP 多次加载
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


app = Flask(__name__)
CORS(app)  # 启用 CORS
# 定义路径并存入配置
app.config["UPLOAD_FOLDER"] = os.path.join("data", "train")
app.config["MODEL_SAVE_FOLDER"] = os.path.join("data", "model")

# 确保文件夹存在
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_SAVE_FOLDER"], exist_ok=True)

# 实例化训练服务类
train_service = YOLOTrainService()


@app.route("/train/", methods=["POST"])
def start_training():
    """
    接收上传文件，启动训练任务
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".zip"):
        return jsonify({"error": "Only .zip files are allowed"}), 400

    try:
        # 保存并解压文件
        dataset_name = os.path.splitext(file.filename)[0]
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset_name)
        zip_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

        file.save(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_path)

        # 读取 data.yaml 文件（验证上传的数据结构）
        data_yaml = os.path.join(save_path, "data.yaml")
        if not os.path.exists(data_yaml):
            raise FileNotFoundError("data.yaml not found in uploaded dataset")

        with open(data_yaml, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)
            print("Train Path:", data_config.get("train"))
            print("Validation Path:", data_config.get("val"))

        # 启动训练任务
        model_save_path = os.path.join(app.config["MODEL_SAVE_FOLDER"], dataset_name)
        task_id = train_service.start_training(dataset_path=save_path, model_save_path=model_save_path)

        return jsonify({"message": "Training started", "task_id": task_id})

    except Exception as e:
        print(f"[ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """
    查询任务状态
    """
    status = train_service.get_task_status(task_id)
    return jsonify({
        "task_id": task_id,
        "status": status["status"],
        "result": status["result"],
        "finished_at": status.get("finished_at"),
        "created_at": status.get("created_at")
    })


if __name__ == "__main__":
    app.run(debug=True)
