import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 原yolov8s
    yaml_yolov8s = 'ultralytics/cfg/models/v8/cls_self/yolov8s-cls.yaml'
    # SE 注意力机制
    yaml_yolov8_SE = 'ultralytics/cfg/models/v8/cls_self/yolov8s-cls-atten-SE.yaml'

    # 替换一下变量名即可
    model_yaml = yaml_yolov8s
    # 模型加载
    model = YOLO(model_yaml)
    data_path = r'E:\develop\bysj\data\traindata'
    # 以yaml文件的名字进行命名
    name = os.path.basename(model_yaml).split('.')[0]
    # 文档中对参数有详细的说明
    model.train(data=data_path,             # 数据集路径
                imgsz=300,                  # 训练图片大小
                epochs=200,                 # 训练的轮次
                batch=4,                    # 训练batch
                workers=2,                  # 加载数据线程数
                device='cpu',               # 使用CPU
                optimizer='SGD',            # 优化器
                project='runs/train',       # 模型保存路径
                name=name,                  # 模型保存命名
                )
