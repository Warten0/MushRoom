from ultralytics import YOLO

if __name__ == '__main__':
    # 模型路径
    model_pt = 'weights/yolov8s-cls/weights/best.pt'
    model = YOLO(model_pt)
    # 数据集路径
    data_path = r'E:\develop\bysj\data\traindata'
    # 文档中对参数有详细的说明
    model.val(data=data_path,           # 数据集路径
              imgsz=300,                # 图片大小，要和训练时一样
              batch=4,                  # batch
              workers=2,                # 加载数据线程数
              conf=0.001,               # 设置检测的最小置信度阈值。置信度低于此阈值的检测将被丢弃。
              iou=0.6,                  # 设置非最大抑制 (NMS) 的交叉重叠 (IoU) 阈值。有助于减少重复检测。
              device='cpu',             # 使用cpu
              project='runs/val',       # 保存路径
              name='exp',               # 保存命名
              )
