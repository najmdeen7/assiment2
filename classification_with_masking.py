from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import os
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from tqdm import tqdm

# ========== 🛠️ إعدادات المسارات ==========
TRAIN_JSON = r"D:\\python\\train_fixed.json"
VAL_JSON = r"D:\\python\\val_fixed.json"
SELECTED_IMAGES_DIR = r"D:\\python\\LV-MHP-v2\\selected_images"

# ========== 🗂️ فئات البيانات ==========
CATEGORY_MAPPING = {
    4: 0,  # person
    1: 1,  # man
    2: 2,  # woman
    3: 3   # child
}
CLASS_NAMES = ["person", "man", "woman", "child"]

# ========== 🔍 فحص الصور التالفة ==========
def check_corrupted_images(json_file):
    corrupted = []
    print(f"\n🔎 فحص الصور في {json_file}...")

    with open(json_file, "r") as f:
        data = json.load(f)

    for img_info in tqdm(data['images']):
        img_path = os.path.join(SELECTED_IMAGES_DIR, img_info['file_name'])
        if not os.path.exists(img_path):
            corrupted.append({
                'image_id': img_info['id'],
                'path': img_path,
                'error': "File not found"
            })
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            corrupted.append({
                'image_id': img_info['id'],
                'path': img_path,
                'error': str(e)
            })

    return corrupted

# ========== 📦 تحميل البيانات ==========
def get_dicts(json_file):
    with open(json_file, "r") as file:
        coco = json.load(file)
    print(f"\n📂 تحميل {len(coco['images'])} صورة من {json_file}...")

    # تجميع التعليقات التوضيحية حسب معرف الصورة
    anns_dict = {ann['image_id']: [] for ann in coco['annotations']}
    for ann in coco['annotations']:
        anns_dict[ann['image_id']].append(ann)

    dataset_dicts = []
    for img_info in coco['images']:
        record = {}
        image_path = os.path.join(SELECTED_IMAGES_DIR, img_info['file_name'])

        if not os.path.exists(image_path):
            print(f"⚠️ الصورة غير موجودة: {image_path}")
            continue

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("فشل قراءة الصورة")
        except Exception as e:
            print(f"❌ خطأ في الصورة: {image_path} - {str(e)}")
            continue

        height, width = img.shape[:2]
        record = {
            "file_name": image_path,
            "image_id": img_info['id'],
            "height": height,
            "width": width,
            "annotations": []
        }

        for ann in anns_dict.get(img_info['id'], []):
            try:
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

                # ✅ التأكد من أن القناع بنفس أبعاد الصورة
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

                # 🔹 إنشاء القناع الثنائي
                binary_mask = np.zeros_like(mask, dtype=np.uint8)
                if category_id == 2:  # المرأة
                    binary_mask[(mask != 3) & (mask != 7) & (mask != 8)] = 1
                else:
                    binary_mask[mask > 0] = 1

                if np.sum(binary_mask) < 50:
                    print(f"⚠️ قناع صغير جدًا: {mask_path} ({np.sum(binary_mask)} بيكسل)")
                    continue

                # ✅ تحويل القناع إلى RLE
                rle = mask_util.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                record["annotations"].append({
                    "bbox": ann['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": mapped_cat_id,
                    "segmentation": rle,
                    "iscrowd": 0
                })

            except Exception as e:
                print(f"⚠️ خطأ في التعليق {ann['id']}: {str(e)}")
                continue

        if len(record["annotations"]) > 0:
            dataset_dicts.append(record)

    print(f"✅ تم تجهيز {len(dataset_dicts)} تسجيلة صالحة من {json_file}")
    return dataset_dicts

# ========== 🚀 بدء التدريب ==========
if __name__ == "__main__":
    # فحص الصور التالفة
    for json_file in [TRAIN_JSON, VAL_JSON]:
        corrupted = check_corrupted_images(json_file)
        if corrupted:
            print(f"\n❌ يوجد {len(corrupted)} صور تالفة في {json_file}:")
            for c in corrupted:
                print(f"- {c['path']} (الخطأ: {c['error']})")
            exit(1)

    # تسجيل المجموعات في Detectron2
    DatasetCatalog.clear()
    DatasetCatalog.register("train_dataset", lambda: get_dicts(TRAIN_JSON))
    MetadataCatalog.get("train_dataset").set(thing_classes=CLASS_NAMES)

    DatasetCatalog.register("val_dataset", lambda: get_dicts(VAL_JSON))
    MetadataCatalog.get("val_dataset").set(thing_classes=CLASS_NAMES)

    # ضبط إعدادات النموذج
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 12500
    cfg.SOLVER.STEPS = (500, 750)
    cfg.SOLVER.GAMMA = 0.1
    cfg.TEST.EVAL_PERIOD = 200

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    try:
        print("\n🚀 بدء عملية التدريب...")
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        print("\n🎉 تم الانتهاء من التدريب بنجاح!")
    except Exception as e:
        print(f"\n❌ فشل التدريب: {str(e)}")
