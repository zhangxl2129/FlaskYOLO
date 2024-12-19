import os
import torch
import uuid
import shutil
import threading
from datetime import datetime
from ultralytics import YOLO

class YOLOTrainService:
    """
    训练服务管理器：提供任务管理和线程支持
    """
    def __init__(self):
        self.tasks = {}  # 存储任务状态和结果 {task_id: {"status": "...", "result": "..."}}
        self.lock = threading.Lock()

    def _run_training(self, task_id, dataset_path, model_save_path):
        """
        实际执行YOLO训练的后台任务
        """
        try:
            with self.lock:
                self.tasks[task_id]["status"] = "RUNNING"

            # 执行YOLO训练
            trainer = YOLOTrain(dataset_path=dataset_path, model_save_path=model_save_path)
            best_model_path = trainer.train()

            with self.lock:
                self.tasks[task_id]["status"] = "SUCCESS"
                self.tasks[task_id]["result"] = best_model_path
                self.tasks[task_id]["finished_at"] = datetime.now().isoformat()

        except Exception as e:
            with self.lock:
                self.tasks[task_id]["status"] = "FAILED"
                self.tasks[task_id]["result"] = str(e)
                self.tasks[task_id]["finished_at"] = datetime.now().isoformat()

    def start_training(self, dataset_path, model_save_path):
        """
        启动YOLO训练任务
        """
        task_id = str(uuid.uuid4())
        with self.lock:
            self.tasks[task_id] = {
                "status": "PENDING",
                "result": None,
                "created_at": datetime.now().isoformat(),
                "finished_at": None
            }

        thread = threading.Thread(
            target=self._run_training, args=(task_id, dataset_path, model_save_path)
        )
        thread.start()
        return task_id

    def get_task_status(self, task_id):
        """
        查询任务状态
        """
        with self.lock:
            return self.tasks.get(task_id, {"status": "NOT FOUND", "result": None})


class YOLOTrain:
    """
    YOLO训练类：封装YOLO训练逻辑
    """
    def __init__(self, dataset_path, model_save_path):
        self.dataset_path = os.path.abspath(dataset_path)
        self.model_save_path = os.path.abspath(model_save_path)
        self.data_yaml_path = os.path.join(self.dataset_path, "data.yaml")

        # 确定YOLO配置文件和设备
        self.model_config = "cfg/models/11/yolo11-WTConv-SPDConv.yaml"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def validate_paths(self):
        """
        验证数据路径和配置文件
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"data.yaml not found in {self.dataset_path}")

    def train(self):
        """
        执行YOLO训练并返回最佳模型路径
        """
        self.validate_paths()

        print(f"Starting training with data: {self.data_yaml_path}")
        print(f"Using device: {self.device}")

        # 初始化模型
        model = YOLO(self.model_config)
        model.train(
            data="data/train/NEU-DTC/data.yaml",
            #data=self.data_yaml_path,
            imgsz=512,
            epochs=200,
            batch=32,
            optimizer="SGD",
            lr0=0.01,
            lrf=0.2,
            weight_decay=5e-4,
            device=self.device,
            cache=False,
        )

        # 保存最佳模型
        best_model_src = "runs/detect/train/weights/best.pt"
        best_model_dst = os.path.join(self.model_save_path, "best.pt")
        os.makedirs(self.model_save_path, exist_ok=True)

        if os.path.exists(best_model_src):
            shutil.copy(best_model_src, best_model_dst)
            print(f"Best model saved to: {best_model_dst}")
            return best_model_dst
        else:
            raise FileNotFoundError("Best model file not found after training.")
