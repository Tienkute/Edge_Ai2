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
        description="Huan luyen YOLO11-micro Ban V13 (Multi-Scale + Pseudo-Labeling + High-Res Calibration)"
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
        "--conf_pseudo",
        type=float,
        default=0.88,
        help="Nguong confidence de sinh pseudo-label sieu sach (mac dinh: 0.88)",
    )
    parser.add_argument(
        "--train_txt",
        type=str,
        default="data/train_v13_pseudo.txt",
        help="File danh sach anh train ket hop V13 (mac dinh: data/train_v13_pseudo.txt)",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="dataset_v13_pseudo.yaml",
        help="File yaml cau hinh dataset (mac dinh: dataset_v13_pseudo.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Duong dan checkpoint khoi tao (de trong se tu dong tim v12_best -> v11_best -> v10_best -> micro_best)",
    )
    parser.add_argument(
        "--epochs_stage2",
        type=int,
        default=100,
        help="So epoch huan luyen Multi-Scale giai doan 2 (mac dinh: 100)",
    )
    parser.add_argument(
        "--epochs_stage3",
        type=int,
        default=25,
        help="So epoch tinh chinh High-Res Calibration giai doan 3 (mac dinh: 25)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Batch size (mac dinh: 64)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=35,
        help="Patience early stopping (mac dinh: 35)",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Chi sinh pseudo-labels va kiem tra pipeline, khong chay train tren may ca nhan",
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Thiet bi huan luyen (mac dinh: 0)",
    )
    return parser.parse_args()


def generate_pseudo_labels_from_csv(sub_path, test_img_dir, test_labels_dir, conf_thr=0.88):
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] GIAI DOAN 1: SINH PSEUDO-LABELS SIEU SACH VOI CONF >= {conf_thr}...")
    print(f"[+] DOC DU LIEU DU DOAN TU: {sub_path}")
    df = pd.read_csv(sub_path)
    df_det = df[df["task_type"] == "DETECTION"]

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

        ultra_clean_objs = [o for o in objs if float(o.get("score", 0.0)) >= conf_thr]
        if not ultra_clean_objs:
            continue

        img_path = img_files.get(sample_id)
        if img_path is None or not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = []
        for obj in ultra_clean_objs:
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

    train_txt_file = base_dir / "data" / "train_images.txt"
    val_txt_file = base_dir / "data" / "val_images.txt"

    if train_txt_file.exists() and val_txt_file.exists():
        with open(train_txt_file, "r", encoding="utf-8") as f:
            train_orig_images.extend([line.strip() for line in f if line.strip()])
        with open(val_txt_file, "r", encoding="utf-8") as f:
            train_orig_images.extend([line.strip() for line in f if line.strip()])
    else:
        train_img_dir = base_dir / "data" / "train" / "images"
        for p in sorted(train_img_dir.iterdir()):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                train_orig_images.append(str(p.resolve()))

    train_orig_images = list(dict.fromkeys(train_orig_images))
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


def resolve_checkpoint(base_dir, custom_weights=""):
    if custom_weights:
        p = Path(custom_weights)
        if not p.is_absolute():
            p = base_dir / p
        if p.exists():
            print(f"[+] SU DUNG CHECKPOINT CHI DINH: {p}")
            return str(p)

    for cand_name in [
        "yolo11_micro_v12_best.pt",
        "yolo11_micro_v11_best.pt",
        "yolo11_micro_v10_best.pt",
        "yolo11_micro_best.pt",
    ]:
        cand_p = base_dir / cand_name
        if cand_p.exists():
            print(f"[+] KHOI TAO TU CHECKPOINT: {cand_p}")
            return str(cand_p)

    candidates = list((base_dir / "runs").rglob("best.pt"))
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"[+] KHOI TAO TU CHECKPOINT TRONG RUNS: {candidates[0]}")
        return str(candidates[0])

    yaml_v13 = base_dir / "models" / "yolo11_micro_v13.yaml"
    print(f"[+] KHOI TAO TU ARCHITECTURE YAML: {yaml_v13}")
    return str(yaml_v13)


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    print("=" * 60)
    print("[+] KHOI DONG TIEN TRINH HUAN LUYEN YOLO11-MICRO BAN V13")
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
            print(f"[!] CHUYEN SANG FILE SUBMISSION: {fallback_sub}")
            sub_path = fallback_sub
        else:
            raise FileNotFoundError(f"Khong tim thay file submission tai {sub_path}")

    # GIAI DOAN 1: Sinh nhan gia sieu sach (conf >= 0.88)
    pseudo_imgs, total_pseudo_boxes = generate_pseudo_labels_from_csv(
        sub_path=sub_path,
        test_img_dir=test_img_dir,
        test_labels_dir=test_labels_dir,
        conf_thr=args.conf_pseudo,
    )
    print(f"[+] SO LUONG ANH TEST CO PSEUDO-LABEL SIEU SACH: {len(pseudo_imgs)}")
    print(f"[+] TONG SO BOUNDING BOXES PSEUDO SIEU SACH: {total_pseudo_boxes}")

    orig_cnt, total_cnt = build_combined_train_list(
        base_dir=base_dir,
        pseudo_image_paths=pseudo_imgs,
        output_txt_path=train_txt_path,
    )
    print(f"[+] SO LUONG ANH TRAIN GOC: {orig_cnt}")
    print(f"[+] TONG SO ANH TRONG TAP TRAIN V13: {total_cnt}")

    create_dataset_yaml(
        base_dir=base_dir,
        train_txt_path=train_txt_path,
        output_yaml_path=data_yaml_path,
    )
    print(f"[+] DA KHOI TAO DATASET YAML TAI: {data_yaml_path}")

    # Xac dinh checkpoint
    weights_path = resolve_checkpoint(base_dir, args.weights)

    device = args.device
    if device == 0 and not torch.cuda.is_available():
        print("[!] CUDA KHONG KHA DUNG, CHUYEN SANG CPU!")
        device = "cpu"
    print(f"[+] THIET BI HUAN LUYEN: {device}")

    # GIAI DOAN 2: Multi-Scale Training (320px)
    print("\n" + "=" * 60)
    print("[+] CAU HINH GIAI DOAN 2: MULTI-SCALE TRAINING (320PX)")
    print("=" * 60)
    print("    - IMGSZ: 320, MULTI_SCALE: TRUE, COS_LR: TRUE")
    print(f"    - EPOCHS: {args.epochs_stage2}, BATCH: {args.batch}, PATIENCE: {args.patience}")
    print("    - LOSS WEIGHTS: BOX = 12.5, DFL = 1.8, CLS = 0.6")
    print("    - AUGMENTATIONS: MOSAIC = 1.0, CLOSE_MOSAIC = 35, MIXUP = 0.1, AMP = TRUE")

    # GIAI DOAN 3: High-Res Anchor Calibration (384px)
    print("\n" + "=" * 60)
    print("[+] CAU HINH GIAI DOAN 3: HIGH-RES ANCHOR CALIBRATION (384PX)")
    print("=" * 60)
    print(f"    - IMGSZ: 384, EPOCHS: {args.epochs_stage3}, LR0: 0.0002, MOSAIC: 0.0")
    print("    - LOSS WEIGHTS: BOX = 12.5, DFL = 1.8, CLS = 0.6, COS_LR: TRUE")
    print("    - MUC TIEU: TINH CHINH CHINH XAC VIEN KHUNG BAO (IOU) O DO PHAN GIAI CAO")

    if args.prepare_only:
        print("\n" + "=" * 60)
        print("[!] CHE DO PREPARE_ONLY: DA CHUAN BI DU LIEU VA KIEM TRA PIPELINE THANH CONG!")
        print("[!] KHONG CHAY TRAIN TREN MAY CA NHAN THEO QUY TAC.")
        print("=" * 60)
        return

    # Thuc thi Giai doan 2 neu khong bat prepare_only
    print(f"\n[+] BAT DAU THUC THI GIAI DOAN 2 TU: {weights_path}...")
    if weights_path.endswith(".yaml"):
        pretrained = base_dir / "yolo11n.pt"
        if pretrained.exists():
            model = YOLO(weights_path).load(str(pretrained))
        else:
            model = YOLO(weights_path)
    else:
        model = YOLO(weights_path)

    run_stage2 = "yolo11_micro_v13_stage2_multiscale"
    results_stage2 = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs_stage2,
        patience=args.patience,
        imgsz=320,
        batch=args.batch,
        multi_scale=True,
        cos_lr=True,
        box=12.5,
        dfl=1.8,
        cls=0.6,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=35,
        device=device,
        project="runs",
        name=run_stage2,
        exist_ok=True,
        amp=True,
    )

    stage2_best = base_dir / "runs" / run_stage2 / "weights" / "best.pt"
    if not stage2_best.exists():
        stage2_best = Path(weights_path)

    # Thuc thi Giai doan 3: High-Res Anchor Calibration
    print(f"\n[+] BAT DAU THUC THI GIAI DOAN 3 CALIBRATION TU: {stage2_best}...")
    model_calib = YOLO(str(stage2_best))

    run_stage3 = "yolo11_micro_v13_stage3_calibration"
    results_stage3 = model_calib.train(
        data=str(data_yaml_path),
        epochs=args.epochs_stage3,
        patience=args.epochs_stage3,
        imgsz=384,
        batch=args.batch,
        lr0=0.0002,
        mosaic=0.0,
        cos_lr=True,
        box=12.5,
        dfl=1.8,
        cls=0.6,
        device=device,
        project="runs",
        name=run_stage3,
        exist_ok=True,
        amp=True,
    )

    stage3_best = base_dir / "runs" / run_stage3 / "weights" / "best.pt"
    v13_final = base_dir / "yolo11_micro_v13_best.pt"
    root_final = base_dir / "yolo11_micro_best.pt"

    print("\n" + "=" * 60)
    print("[+] HUAN LUYEN PIPELINE BAN V13 HOAN TAT!")
    print(f"[+] THU MUC LUU KET QUA: {base_dir / 'runs' / run_stage3}")
    if stage3_best.exists():
        print(f"[+] TRONG SO TOT NHAT SAU CALIBRATION: {stage3_best}")
        shutil.copy(str(stage3_best), str(v13_final))
        shutil.copy(str(stage3_best), str(root_final))
        print(f"[+] DA DONG BO TRONG SO VAO: {v13_final} VA {root_final}")
    print("=" * 60)


if __name__ == "__main__":
    main()
