import os
import torch
import json
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2 import model_zoo
SELECTED_IMAGES_DIR = r"D:\\python\\LV-MHP-v2\\selected_images"
def get_dicts(json_file):
    import cv2
    import pycocotools.mask as mask_util
    from detectron2.structures import BoxMode

    with open(json_file, "r") as file:
        coco = json.load(file)
    print(f"\n📂 تحميل {len(coco['images'])} صورة من {json_file}...")

    CATEGORY_MAPPING = {
        4: 0,  # person
        1: 1,  # man
        2: 2,  # woman
        3: 3   # child
    }

    dataset_dicts = []
    for img_info in coco['images']:
        record = {}
        image_path = os.path.join(SELECTED_IMAGES_DIR, img_info['file_name'])

        if not os.path.exists(image_path):
            print(f"⚠️ الصورة غير موجودة: {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ خطأ في قراءة الصورة: {image_path}")
            continue

        height, width = img.shape[:2]
        record["file_name"] = image_path
        record["image_id"] = img_info['id']
        record["height"] = height
        record["width"] = width
        record["annotations"] = []

        for ann in coco['annotations']:
            if ann['image_id'] != img_info['id']:
                continue

            mask_path = ann.get('mask_path')
            if not mask_path or not os.path.exists(mask_path):
                print(f"⚠️ القناع غير موجود: {mask_path}")
                continue

            category_id = ann['category_id']
            if category_id not in CATEGORY_MAPPING:
                print(f"⚠️ فئة غير معروفة: {category_id}")
                continue
            mapped_cat_id = CATEGORY_MAPPING[category_id]

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️ فشل قراءة القناع: {mask_path}")
                continue

            binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)

            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            record["annotations"].append({
                "bbox": ann['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": mapped_cat_id,
                "segmentation": rle,
                "iscrowd": 0
            })

        if len(record["annotations"]) > 0:
            dataset_dicts.append(record)

    print(f"✅ تم تجهيز {len(dataset_dicts)} تسجيلة صالحة من {json_file}")
    return dataset_dicts
if __name__ == '__main__':
    # 🔹 إعداد المسارات
    MODEL_DIR = r"D:\python\output"  # المجلد الذي يحتوي على النموذج المدرب
    MODEL_WEIGHTS = os.path.join(MODEL_DIR, "‏‏model_final -10000 .pth")  # النموذج الأول بدون Masking
    VAL_JSON = r"D:\\python\\val_fixed.json"

    # 🔹 تحميل الإعدادات للنموذج الأول (Classification بدون Masking)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ("val_dataset",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # عدد الفئات
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # ✅ تسجيل مجموعة التقييم
    DatasetCatalog.register("val_dataset", lambda: get_dicts(VAL_JSON))
    MetadataCatalog.get("val_dataset").set(thing_classes=["person", "man", "woman", "child"])

    # 🔹 تحميل النموذج
    predictor = DefaultPredictor(cfg)

    # ✅ إنشاء مقيّم COCO
    evaluator = COCOEvaluator("val_dataset", cfg, False, output_dir=MODEL_DIR)
    val_loader = build_detection_test_loader(cfg, "val_dataset")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # 📊 طباعة نتائج التقييم
    print("\n📊 **نتائج التقييم للنموذج Classification بدون Masking:**")
    print(results)
