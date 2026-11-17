import argparse
import json
import os
from pathlib import Path
import shutil
import cv2
import pandas as pd
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huan luyen YOLO11-micro Ban V10 voi Pseudo-Labeling cho ESP32-S3"
    )
    parser.add_argument(
        "--submission",
        type=str,
        default="outputs/submissions/submission_yolo11_micro.csv",
        help="File submission chua du doan tap test (mac dinh: outputs/submissions/submission_yolo11_micro.csv)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/test/images",
        help="Thu muc chua anh test (mac dinh: data/test/images)",
    )
    parser.add_argument(
        "--test_labels_dir",
        type=str,
        default="data/test/labels",
        help="Thu muc luu nhan pseudo cua tap test (mac dinh: data/test/labels)",
    )
    parser.add_argument(
        "--conf_thr",
        type=float,
        default=0.85,
        help="Nguong confidence de loc bounding box sinh pseudo-label (mac dinh: 0.85)",
    )
    parser.add_argument(
        "--train_txt",
        type=str,
        default="data/train_v10_pseudo.txt",
        help="File danh sach anh train ket hop (mac dinh: data/train_v10_pseudo.txt)",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="dataset_v10_pseudo.yaml",
        help="File yaml cau hinh dataset (mac dinh: dataset_v10_pseudo.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Duong dan trong so khoi tao (de trong se tu dong tim v8_best -> micro_best -> yolo11n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="So epoch fine-tune (mac dinh: 150)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Batch size (mac dinh: 64)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Kich thuoc anh huan luyen (mac dinh: 320)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Learning rate khoi tao (mac dinh: 0.001)",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=10.0,
        help="He so loss box (mac dinh: 10.0)",
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=30,
        help="So epoch cuoi tat mosaic (mac dinh: 30)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Patience early stopping (mac dinh: 40)",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Chi sinh pseudo-labels va file train_v10_pseudo.txt, khong chay train",
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Thiet bi huan luyen (mac dinh: 0)",
    )
    return parser.parse_args()


def generate_pseudo_labels_from_csv(sub_path, test_img_dir, test_labels_dir, conf_thr=0.85):
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] DANG DOC DU LIEU PREDICTIONS TU: {sub_path}")
    df = pd.read_csv(sub_path)
    df_det = df[df["task_type"] == "DETECTION"]

    # Anh xa sample_id sang file anh
    img_files = {}
    for p in test_img_dir.iterdir():
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            img_files[p.stem] = p

    pseudo_images = []
    total_boxes = 0

    for idx, row in df_det.iterrows():
        sample_id = str(row["sample_id"])
        pred_val = row["prediction"]
        if pd.isna(pred_val):
            continue

        try:
            objs = json.loads(str(pred_val))
        except Exception:
            continue

        high_conf_objs = [o for o in objs if float(o.get("score", 0.0)) >= conf_thr]
        if not high_conf_objs:
            continue

        img_path = img_files.get(sample_id)
        if img_path is None or not img_path.exists():
            continue

        # Doc kich thuoc anh
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = []
        for obj in high_conf_objs:
            cls_id = int(obj["class_id"])
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            x1 = max(0.0, min(float(w), float(x1)))
            x2 = max(0.0, min(float(w), float(x2)))
            y1 = max(0.0, min(float(h), float(y1)))
            y2 = max(0.0, min(float(h), float(y2)))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1.0 or bh <= 1.0:
                continue

            # Chuan hoa toa do ve [0, 1] theo dinh dang YOLO (x_center, y_center, width, height)
            x_center = (x1 + x2) / (2.0 * w)
            y_center = (y1 + y2) / (2.0 * h)
            norm_w = bw / w
            norm_h = bh / h

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        if lines:
            label_file = test_labels_dir / f"{sample_id}.txt"
            with open(label_file, "w", encoding="utf-8") as f:
                f.writelines(lines)

            pseudo_images.append(str(img_path.resolve()))
            total_boxes += len(lines)

    return pseudo_images, total_boxes


def build_combined_train_list(base_dir, pseudo_image_paths, output_txt_path):
    train_orig_images = []

    # 1. Thu doc tu train_images.txt va val_images.txt de du 6.000 anh
    train_txt_file = base_dir / "data" / "train_images.txt"
    val_txt_file = base_dir / "data" / "val_images.txt"

    if train_txt_file.exists() and val_txt_file.exists():
        with open(train_txt_file, "r", encoding="utf-8") as f:
            train_orig_images.extend([line.strip() for line in f if line.strip()])
        with open(val_txt_file, "r", encoding="utf-8") as f:
            train_orig_images.extend([line.strip() for line in f if line.strip()])
    else:
        # Neu khong co file txt, quet truc tiep thu muc train/images
        train_img_dir = base_dir / "data" / "train" / "images"
        for p in sorted(train_img_dir.iterdir()):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                train_orig_images.append(str(p.resolve()))

    # Loai bo trung lap va giu nguyen danh sach 6.000 anh goc
    train_orig_images = list(dict.fromkeys(train_orig_images))

    # 2. Gop 6.000 anh train goc voi cac anh test co pseudo label
    combined_list = list(dict.fromkeys(train_orig_images + pseudo_image_paths))

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for p in combined_list:
            f.write(f"{p}\n")

    return len(train_orig_images), len(combined_list)


def create_dataset_yaml(base_dir, train_txt_path, output_yaml_path):
    data_dir = base_dir / "data"
    val_txt = data_dir / "val_images.txt"
    if not val_txt.exists():
        val_txt = train_txt_path

    yaml_content = f"""path: {data_dir.as_posix()}
train: {train_txt_path.name}
val: {val_txt.name}

nc: 7
names:
  0: bottle
  1: cup
  2: phone
  3: book
  4: mouse
  5: box
  6: toy_car
"""
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def resolve_model_and_weights(base_dir, custom_weights=""):
    if custom_weights:
        w_path = Path(custom_weights)
        if not w_path.is_absolute():
            w_path = base_dir / w_path
        if w_path.exists():
            print(f"[+] SU DUNG TRONG SO CHI DINH: {w_path}")
            return YOLO(str(w_path))

    # Uu tien 1: yolo11_micro_v8_best.pt
    v8_weights = base_dir / "yolo11_micro_v8_best.pt"
    if v8_weights.exists():
        print(f"[+] KHOI TAO MODEL TU CHECKPOINT V8: {v8_weights}")
        return YOLO(str(v8_weights))

    # Uu tien 2: yolo11_micro_best.pt
    micro_best = base_dir / "yolo11_micro_best.pt"
    if micro_best.exists():
        print(f"[+] KHOI TAO MODEL TU CHECKPOINT TOT NHAT HIEN TAI: {micro_best}")
        return YOLO(str(micro_best))

    # Uu tien 3: yolo11_micro.yaml nap yolo11n.pt
    model_yaml = base_dir / "models" / "yolo11_micro.yaml"
    pretrained = base_dir / "yolo11n.pt"
    print(f"[+] KHOI TAO MODEL TU: {model_yaml}")
    if pretrained.exists():
        print(f"[+] NAP TRONG SO PRETRAINED TU: {pretrained}")
        return YOLO(str(model_yaml)).load(str(pretrained))

    return YOLO(str(model_yaml))


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    print("=" * 60)
    print("[+] KHOI DONG TIEN TRINH HUAN LUYEN YOLO11-MICRO BAN V10 (PSEUDO-LABELING)")
    print("=" * 60)

    # 1. Duong dan tap tin
    test_img_dir = base_dir / args.test_dir
    test_labels_dir = base_dir / args.test_labels_dir
    train_txt_path = base_dir / args.train_txt
    data_yaml_path = base_dir / args.data_yaml

    sub_path = Path(args.submission)
    if not sub_path.is_absolute():
        sub_path = base_dir / sub_path

    if not sub_path.exists():
        fallback_sub = base_dir / "outputs" / "submissions" / "submission.csv"
        if fallback_sub.exists():
            print(f"[!] KHONG TIM THAY {sub_path}, CHUYEN SANG FILE: {fallback_sub}")
            sub_path = fallback_sub
        else:
            raise FileNotFoundError(f"Khong tim thay file submission de tao pseudo-labels tai {sub_path}")

    # 2. Sinh pseudo-labels tu tap Test voi score >= conf_thr (0.85)
    print(f"[+] TIEN HANH SINH PSEUDO-LABELS TU TEST SET VOI NGUONG CONF >= {args.conf_thr}...")
    pseudo_imgs, total_pseudo_boxes = generate_pseudo_labels_from_csv(
        sub_path=sub_path,
        test_img_dir=test_img_dir,
        test_labels_dir=test_labels_dir,
        conf_thr=args.conf_thr,
    )
    print(f"[+] SO LUONG ANH TEST CO PSEUDO-LABEL HOP LE: {len(pseudo_imgs)}")
    print(f"[+] TONG SO BOUNDING BOXES PSEUDO DUOC TAO: {total_pseudo_boxes}")
    print(f"[+] THU MUC LUU NHAN TEST: {test_labels_dir}")

    # 3. Gop 6.000 anh train goc va cac anh test co pseudo label vao train_v10_pseudo.txt
    print("[+] DANG GOP 6.000 ANH TRAIN GOC VA ANH TEST CO PSEUDO LABEL...")
    orig_cnt, total_cnt = build_combined_train_list(
        base_dir=base_dir,
        pseudo_image_paths=pseudo_imgs,
        output_txt_path=train_txt_path,
    )
    print(f"[+] SO LUONG ANH TRAIN GOC: {orig_cnt}")
    print(f"[+] SO LUONG ANH TEST CO PSEUDO LABEL THEM VAO: {len(pseudo_imgs)}")
    print(f"[+] TONG SO ANH TRONG FILE TRAIN V10: {total_cnt}")
    print(f"[+] DA XUAT FILE DANH SACH TRAIN TAI: {train_txt_path}")

    # 4. Tao file dataset yaml
    create_dataset_yaml(
        base_dir=base_dir,
        train_txt_path=train_txt_path,
        output_yaml_path=data_yaml_path,
    )
    print(f"[+] DA TAO FILE CAU HINH DATASET YAML TAI: {data_yaml_path}")

    # 5. Khoi tao model
    print("[+] KHOI TAO MODEL YOLO11-MICRO (< 200.000 THAM SO)...")
    model = resolve_model_and_weights(base_dir, args.weights)

    device = args.device
    if device == 0 and not torch.cuda.is_available():
        print("[!] CUDA KHONG KHA DUNG, CHUYEN SANG CPU!")
        device = "cpu"
    print(f"[+] THIET BI HUAN LUYEN: {device}")

    # 6. Thong tin cau hinh fine-tune Ban V10
    print("[+] CAU HINH CHIEN THUAT BAN V10 PSEUDO-LABELING:")
    print(f"    - DATA: {data_yaml_path}")
    print(f"    - IMGSZ: {args.imgsz}")
    print(f"    - EPOCHS: {args.epochs}, BATCH: {args.batch}, PATIENCE: {args.patience}")
    print(f"    - LR0: {args.lr0}, COS_LR: TRUE")
    print(f"    - BOX LOSS WEIGHT: {args.box}, CLS LOSS: 0.7")
    print(f"    - CLOSE_MOSAIC: {args.close_mosaic}, MOSAIC: 1.0, MIXUP: 0.1")

    if args.prepare_only:
        print("=" * 60)
        print("[!] CHE DO PREPARE_ONLY: DA CHUAN BI DU LIEU THANH CONG!")
        print("[!] KHONG CHAY HUAN LUYEN TREN MAY CA NHAN THEO QUY TAC.")
        print("=" * 60)
        return

    # 7. Tien hanh train fine-tune neu khong bat prepare_only
    run_name = "yolo11_micro_esp32_v10_pseudo"
    print(f"[+] BAT DAU FINE-TUNE RUN: {run_name}...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        cos_lr=True,
        box=args.box,
        cls=0.7,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=args.close_mosaic,
        device=device,
        project="runs",
        name=run_name,
        exist_ok=True,
        amp=True,
    )

    save_dir = base_dir / "runs" / run_name
    best_weights = save_dir / "weights" / "best.pt"
    v10_weights = base_dir / "yolo11_micro_v10_best.pt"
    root_weights = base_dir / "yolo11_micro_best.pt"

    print("=" * 60)
    print("[+] HUAN LUYEN BAN V10 PSEUDO-LABELING HOAN TAT!")
    print(f"[+] THU MUC LUU KET QUA: {save_dir}")
    if best_weights.exists():
        print(f"[+] TRONG SO TOT NHAT: {best_weights}")
        shutil.copy(str(best_weights), str(v10_weights))
        shutil.copy(str(best_weights), str(root_weights))
        print(f"[+] DA DONG BO TRONG SO VAO: {v10_weights} VA {root_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
