import os
import json
import cv2
import random
import torch
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# 🟢 **تحديد المسارات**
json_file = r"D:\python\instances_default.json"  # ملف JSON المستخرج من CVAT
image_dir = r"D:\python\LV-MHP-v2\selected_images"  # الصور المصفاة
output_dir = r"D:\python\output"  # مجلد لحفظ النتائج
os.makedirs(output_dir, exist_ok=True)

# 🟢 **تحميل البيانات من JSON**
with open(json_file, "r") as f:
    dataset = json.load(f)

# 🟢 **تحديد الفئات الصحيحة مع إضافة عنصر فارغ في البداية** ✅
thing_classes = ["", "man", "woman", "child" ]  # ✅ إضافة عنصر فارغ للحفاظ على تناسق الفهرس
MetadataCatalog.get("LVMHPV2_train").set(thing_classes=thing_classes)
MetadataCatalog.get("LVMHPV2_val").set(thing_classes=thing_classes)

# 🟢 **توزيع الصور بين التدريب والتقييم**
all_images = dataset["images"]
random.shuffle(all_images)
train_images = all_images[:500]  # 500 صورة للتدريب
val_images = all_images[500:700]  # 200 صورة للتقييم

# 🟢 **دالة لتحويل البيانات إلى تنسيق Detectron2**
def get_LVMHPV2_dicts(subset):
    dataset_dicts = []
    for img in subset:
        record = {
            "file_name": os.path.join(image_dir, img["file_name"]),
            "image_id": img["id"],
            "height": img["height"],
            "width": img["width"],
            "annotations": []
        }

        # 🟢 استخراج التعليقات التوضيحية المرتبطة بهذه الصورة
        annotations = [ann for ann in dataset["annotations"] if ann["image_id"] == img["id"]]
        for ann in annotations:
            category_id = ann["category_id"]  # ✅ استخدم الفهرس كما هو
            if 1 <= category_id <= 4:  # ✅ فقط الفئات المسموح بها (man, woman, child)
                record["annotations"].append({
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": category_id  # ✅ لا حاجة لتعديل الفهرس لأنه متطابق مع thing_classes
                })

        dataset_dicts.append(record)
    return dataset_dicts

# 🟢 **تسجيل بيانات التدريب والتقييم**
DatasetCatalog.register("LVMHPV2_train", lambda: get_LVMHPV2_dicts(train_images))
DatasetCatalog.register("LVMHPV2_val", lambda: get_LVMHPV2_dicts(val_images))

MetadataCatalog.get("LVMHPV2_train").set(thing_classes=thing_classes)
MetadataCatalog.get("LVMHPV2_val").set(thing_classes=thing_classes)

# 🟢 **عرض 3 عينات قبل التدريب للتأكد من صحة البيانات**
def visualize_samples():
    sample_images = random.sample(train_images, 3)
    for idx, img_data in enumerate(sample_images):
        img_path = os.path.join(image_dir, img_data["file_name"])
        img = cv2.imread(img_path)

        annotations = [ann for ann in dataset["annotations"] if ann["image_id"] == img_data["id"]]

        instances = []
        for ann in annotations:
            category_id = ann["category_id"]
            instances.append({"bbox": ann["bbox"], "bbox_mode": BoxMode.XYWH_ABS, "category_id": category_id})

        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("LVMHPV2_train"), scale=0.5)
        out = visualizer.draw_dataset_dict({"annotations": instances})

        output_path = os.path.join(output_dir, f"sample_{idx}.jpg")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        print(f"✅ تم حفظ العينة في: {output_path}")

# 🟢 **عرض العينات قبل التدريب**
visualize_samples()

if __name__ == '__main__':
    # 🟢 **إعداد النموذج**
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("LVMHPV2_train",)
    cfg.DATASETS.TEST = ("LVMHPV2_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # استخدام أوزان مسبقة
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 20000  # عدد التكرارات
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)  # ✅ ضبط عدد الفئات

    # 🟢 **بدء التدريب**
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 🟢 **حفظ النموذج المدرب**
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # عتبة الثقة
    predictor = DefaultPredictor(cfg)
    val_loader = build_detection_test_loader(cfg, "LVMHPV2_val")

    # 🟢 **التقييم باستخدام الـ 200 صورة من `selected_images` فقط**
    evaluator = COCOEvaluator("LVMHPV2_val", cfg, False, output_dir=output_dir)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    print("✅ 🎉 التدريب انتهى بنجاح! تحقق من النتائج في:", output_dir)
    results_file = os.path.join(output_dir, "coco_instances_results.json")

    # 🟢 تحقق مما إذا تم إنشاء ملف النتائج
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            predictions = json.load(f)

        print(f"✅ عدد التوقعات المحفوظة: {len(predictions)}")
        print("🔍 أول 5 نتائج من التقييم:")
        for pred in predictions[:5]:
            print(pred)
    else:
        print("❌ لم يتم العثور على ملف التوقعات! تحقق من تنفيذ التقييم بشكل صحيح.")

