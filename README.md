DINOv3 目标检测框架使用预训练的 DINOv3 视觉 transformer 进行生产就绪的目标检测。无需训练 - 使用自然语言描述立即检测对象！🎯 无需训练，即刻使用 | 🖱️ GUI 支持 | 🖥️ 本地运行 | 🐍 Python API📋 目录安装方法快速入门详细用法实战示例性能优化🚀 Key FeaturesZero-Shot Detection: 使用文本描述检测任何对象 - 无需训练Pretrained Models: 使用 DINOv3 + CLIP 进行即时目标检测Natural Language: 描述您想查找的内容 (例如："红色的车", "戴帽子的人")Production Ready: 专为实际应用设计的简单 APIFast Inference: 支持 FP16，推理速度优化Flexible: 适用于图像和视频🛠 安装方法1. 使用 UV (推荐)Bash# 克隆仓库
git clone https://github.com/euisuk-chung/DINOv3-Object-Detection.git
cd Dino_v3_OD

# 使用 UV 安装依赖
uv sync
2. 使用 pipBash# 克隆仓库后
cd Dino_v3_OD

# 创建虚拟环境 (可选)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install torch torchvision
pip install transformers opencv-python pillow numpy scipy
🚀 快速入门 (5分钟上手!)🖥️ 本地运行 (3种方式)方式 1: 🖱️ 运行 GUI (推荐新手)Bash# 运行 GUI
python detect_gui.py

# Windows: 双击运行
run_detector.bat
支持鼠标操作支持图像预览实时查看结果方式 2: 💻 命令行 (CLI)Bash# 处理单个图像
python detect.py photo.jpg "人" "汽车" "狗"

# 保存结果
python detect.py photo.jpg --targets "person,car,dog" --output detected.jpg

# 处理视频
python detect.py video.mp4 --targets "人,汽车" --output result.mp4

# 高级选项
python detect.py image.jpg --targets "红色的车" --threshold 0.2 --model large --gpu

# 帮助
python detect.py --help
方式 3: 📁 批量处理文件夹Bash# 处理文件夹内的所有图像
python detect_folder.py ./images "人" "汽车"

# 将结果保存为 CSV
python detect_folder.py ./photos --targets "person,car,dog" --csv results.csv

# 包含子文件夹, 并保存结果图像
python detect_folder.py ./dataset --targets "vehicle" --recursive --output ./results

# 仅查看摘要
python detect_folder.py ./images --targets "person" --summary
🐍 Python 代码调用 - 3行搞定!Pythonfrom production_api import detect_objects

# 从图像中查找所需对象 - 无需训练!
results = detect_objects("photo.jpg", ["人", "汽车", "狗"])
print(f"找到的对象: {len(results)}个")
🎯 第一个示例: 从图像中查找对象Pythonfrom production_api import DINOv3DetectorAPI
import cv2

# 1. 初始化检测器 (首次运行时会下载模型)
detector = DINOv3DetectorAPI(model_size="small")  # 用于快速测试

# 2. 加载图像
image = cv2.imread("my_photo.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. 查找所需对象 - 支持中文!
results = detector.detect(
    image,
    ["人", "红色的车", "黄色的巴士", "狗"],
    confidence_threshold=0.3  # 置信度阈值
)

# 4. 检查结果
for detection in results:
    print(f"发现: {detection.class_name} (置信度: {detection.confidence:.2f})")
    print(f"位置: {detection.bbox}")

# 5. 可视化 (可选)
vis_image = detector.visualize(image, results, "result.jpg")
📖 详细用法1️⃣ 基本配置Pythonfrom production_api import DINOv3DetectorAPI

# 选择模型大小
detector = DINOv3DetectorAPI(
    model_size="small",    # 速度快, 适合简单对象
    # model_size="base",     # 性能均衡
    # model_size="large",    # 准确度最高, 速度慢
    device="cuda",         # 使用 GPU (如果没有则为 "cpu")
    use_fp16=True          # 在 GPU 上推理速度快 2 倍
)
2️⃣ 多种输入方式Python# 方式 1: 文件路径
results = detector.detect("image.jpg", ["person", "car"])

# 方式 2: numpy 数组
import numpy as np
image_array = np.array(...)  # 你的图像数据
results = detector.detect(image_array, ["person", "car"])

# 方式 3: PIL 图像
from PIL import Image
pil_image = Image.open("image.jpg")
image_array = np.array(pil_image)
results = detector.detect(image_array, ["person", "car"])
3️⃣ 用自然语言描述对象Python# 简单的对象名称
simple_targets = ["人", "汽车", "狗", "猫"]

# 具体的描述
detailed_targets = [
    "红色的跑车",
    "穿蓝色衬衫的人",
    "戴眼镜的老人",
    "背着背包的学生",
    "黄色的校车"
]

# 复杂的场景
complex_targets = [
    "正在过马路的人",
    "停在红灯前的车",
    "叼着球的狗",
    "使用笔记本电脑的人"
]

results = detector.detect(image, detailed_targets)
4️⃣ 结果筛选Python# 所有检测结果
all_results = detector.detect(image, targets, confidence_threshold=0.2)

# 仅保留高置信度
high_confidence = detector.filter_detections(
    all_results,
    min_confidence=0.7
)

# 仅保留特定类别
only_people = detector.filter_detections(
    all_results,
    class_names=["人", "person"]
)

# 仅保留大对象 (基于像素面积)
large_objects = detector.filter_detections(
    all_results,
    min_area=10000  # 100x100 像素以上
)
5️⃣ 批量处理多个图像Python# 准备图像列表
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
targets = ["person", "car", "dog"]

# 批量处理
all_results = detector.detect_batch(images, targets)

# 每个图像的结果
for idx, results in enumerate(all_results):
    print(f"\n图像 {idx+1}: 发现 {len(results)} 个对象")
    for det in results:
        print(f"  - {det.class_name}: {det.confidence:.2f}")
6️⃣ 视频处理Pythonfrom production_api import process_video

# 简单的视频处理
process_video(
    "input.mp4",        # 输入视频
    "output.mp4",       # 输出视频
    ["人", "汽车"],      # 要查找的对象
    threshold=0.3       # 置信度阈值
)

# 高级视频处理 (手动控制)
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 逐帧处理
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect(frame_rgb, ["person", "car"])
    
    # 执行你想要的处理...
💡 实战示例示例 1: 安防摄像头监控Python# 检测特定情况
security_targets = [
    "背着背包的人",
    "被遗弃的包",
    "奔跑的人",
    "倒下的人",
    "戴口罩的人"
]

results = detector.detect(cctv_frame, security_targets)

# 危险情况警报
for det in results:
    if det.class_name == "倒下的人" and det.confidence > 0.7:
        send_alert("检测到紧急情况!")
示例 2: 库存管理Python# 查找产品
products = [
    "可口可乐罐",
    "百事可乐瓶",
    "橙汁",
    "空货架"
]

results = detector.detect(shelf_image, products)

# 检查库存
product_count = {}
for det in results:
    if det.class_name in product_count:
        product_count[det.class_name] += 1
    else:
        product_count[det.class_name] = 1
示例 3: 交通监控Python# 分析交通状况
traffic_targets = [
    "红灯",
    "绿灯",
    "停止的车辆",
    "行驶中的车辆",
    "人行横道上的行人"
]

results = detector.detect(traffic_cam, traffic_targets)

# 检测违规
for det in results:
    if det.class_name == "红灯":
        red_light_box = det.bbox
        # 检查闯红灯的车辆...
示例 4: 质量检查Python# 查找产品缺陷
defect_targets = [
    "划伤的表面",
    "破损的部分",
    "变色的区域",
    "缺失的部件"
]

results = detector.detect(product_image, defect_targets)

# 质量判定
if len(results) > 0:
    print("检测到不良品!")
    for det in results:
        print(f"问题: {det.class_name} at {det.bbox}")
else:
    print("正常产品")
🔧 利用结果数据保存为 JSONPython# 将检测结果保存为 JSON
import json

results = detector.detect(image, targets)
results_dict = detector.to_dict(results)

with open("detection_results.json", "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=2, ensure_ascii=False)
保存到数据库Python# 将结果保存到数据库
for det in results:
    db.insert({
        "timestamp": datetime.now(),
        "object_type": det.class_name,
        "confidence": det.confidence,
        "location": det.bbox,
        "image_id": image_id
    })
导出为 CSVPythonimport csv

# 导出为 CSV
with open("detections.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Confidence", "X1", "Y1", "X2", "Y2"])
    
    for det in results:
        writer.writerow([
            det.class_name,
            det.confidence,
            *det.bbox
        ])
⚡ 性能优化GPU 加速Python# 检查 GPU 是否可用
import torch
print(f"GPU 是否可用: {torch.cuda.is_available()}")

# 在 GPU 上使用 FP16 (速度快 2 倍)
detector = DINOv3DetectorAPI(
    model_size="base",
    device="cuda",
    use_fp16=True  # 重要!
)
处理速度比较模型大小CPU (秒)GPU (秒)GPU+FP16 (秒)Small0.50.10.05Base1.00.20.1Large2.00.40.2减少内存使用Python# 使用小批量进行处理
for i in range(0, len(images), 4):
    batch = images[i:i+4]
    results = detector.detect_batch(batch, targets)
    
# 使用后清理内存
import gc
del detector
torch.cuda.empty_cache()
gc.collect()
🆘 问题排查首次运行模型下载缓慢Python# 解决方案: 提前下载模型
from transformers import AutoModel, AutoImageProcessor

# 运行一次后即会缓存
AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
CUDA out of memory 错误Python# 解决方案 1: 使用更小的模型
detector = DINOv3DetectorAPI(model_size="small")

# 解决方案 2: 减小图像尺寸
image = cv2.resize(image, (640, 480))

# 解决方案 3: 减小批量大小
results = detector.detect_batch(images[:2], targets)
检测率低Python# 解决方案 1: 降低阈值
results = detector.detect(image, targets, confidence_threshold=0.1)

# 解决方案 2: 使用更具体的描述
targets = ["大的红色卡车", "小的白色轿车"]  # 好
# targets = ["车"]  # 太笼统

# 解决方案 3: 使用更大的模型
detector = DINOv3DetectorAPI(model_size="large")
📊 结果示例检测结果的格式如下:PythonDetection(
    class_name="红色的车",
    confidence=0.85,
    bbox=(120, 230, 450, 380)  # [x1, y1, x2, y2]
)
JSON 输出:JSON{
    "class_name": "红色的车",
    "confidence": 0.85,
    "bbox": [120, 230, 450, 380]
}
🚀 开始使用🎯 使用方法总结方式 1: 命令行 (终端/CMD)Bash# 最简单的用法
python detect.py 图像.jpg "人" "汽车"

# 查看帮助
python detect.py --help
方式 2: GUI (推荐新手)Bashpython detect_gui.py
# 使用鼠标点击操作!
方式 3: Python 脚本Pythonfrom production_api import DINOv3DetectorAPI

detector = DINOv3DetectorAPI()
results = detector.detect("image.jpg", ["person", "car"])
📝 全部文件说明文件用途适用对象detect.py命令行单次处理终端用户detect_folder.py文件夹批量处理需要批量处理时detect_gui.pyGUI 界面新手, 偏好鼠标操作production_api.pyPython API开发者test_simple.py简单测试初次使用时💻 运行示例合集Bash# 1. 测试 (初次使用时)
python test_simple.py

# 2. 单张图像
python detect.py my_photo.jpg "人" "汽车"

# 3. 整个文件夹
python detect_folder.py ./photos --targets "person,car" --output ./results

# 4. 运行 GUI
python detect_gui.py

# 5. 视频
python detect.py video.mp4 --targets "人" --output detected_video.mp4
📞 支持问题: 提交到 GitHub Issues改进建议: 欢迎 Pull Request许可证: MIT (可商用)现在就开始吧! 只需几行代码，即可实现强大的目标检测 🎯
