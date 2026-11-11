import argparse
import json
import os
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Create submission file for Edge AI Challenge")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO weights checkpoint (.pt). If omitted, prioritizes yolo11s_qualifiers then newest best.pt.",
    )
    return parser.parse_args()


def format_detection_result(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "[]"

    objects = []
    clss = boxes.cls.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    xyxys = boxes.xyxy.cpu().tolist()

    for cls_id, conf, xyxy in zip(clss, confs, xyxys):
        objects.append({
            "class_id": int(cls_id),
            "score": round(float(conf), 4),
            "bbox": [round(float(c), 2) for c in xyxy],
        })

    return json.dumps(objects, separators=(",", ":"))


def count_boxes(df_det):
    total = 0
    empty_cnt = 0
    for pred in df_det["prediction"]:
        try:
            objs = json.loads(pred)
            total += len(objs)
            if len(objs) == 0:
                empty_cnt += 1
        except Exception:
            pass
    return total, empty_cnt


def main():
    args = parse_args()

    print("=" * 60)
    print("TOI UU HOA SUY LUAN DETECTION (CONF=0.10, IOU=0.55, TTA=TRUE, IMGSZ=640)")
    print("=" * 60)

    base_dir = Path("D:/visual code/python/Edge_Ai2")
    sample_sub_path = base_dir / "data" / "sample_submission.csv"
    intent_pred_path = base_dir / "outputs" / "predictions" / "intent_predictions.csv"
    test_img_dir = base_dir / "data" / "test" / "images"
    sub_dir = base_dir / "outputs" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    opt_sub_path = sub_dir / "submission_opt_conf10_tta.csv"
    main_sub_path = sub_dir / "submission.csv"

    # Dem so bbox cu neu co file submission.csv cu
    old_box_count = 0
    if main_sub_path.exists():
        try:
            df_old = pd.read_csv(main_sub_path)
            old_det = df_old[df_old["task_type"] == "DETECTION"]
            old_box_count, _ = count_boxes(old_det)
            print(f"[+] So luong bounding boxes lan truoc: {old_box_count:,}")
        except Exception as e:
            print(f"[!] Khong doc duoc file cu: {e}")

    # 1. Kiem tra file sample_submission.csv
    print(f"[+] Kiem tra sample_submission: {sample_sub_path}")
    df_sample = pd.read_csv(sample_sub_path)
    print(f"[+] So luong dong sample_submission: {len(df_sample)}")
    print(f"[+] Cac cot: {df_sample.columns.tolist()}")

    # 2. Xac dinh checkpoint YOLO
    chosen_ckpt = None
    if args.weights:
        custom_ckpt = Path(args.weights)
        if custom_ckpt.exists():
            chosen_ckpt = custom_ckpt
            print(f"[+] Su dung checkpoint chi dinh tu tham so --weights: {chosen_ckpt}")
        else:
            print(f"[!] Khong tim thay checkpoint chi dinh: {custom_ckpt}, chuyen sang tim tu dong.")

    if chosen_ckpt is None:
        yolo11s_ckpt = base_dir / "runs" / "yolo11s_qualifiers" / "weights" / "best.pt"
        if yolo11s_ckpt.exists():
            chosen_ckpt = yolo11s_ckpt
            print(f"[+] Uu tien chon checkpoint YOLO11s: {chosen_ckpt}")
        else:
            runs_dir = base_dir / "runs"
            candidates = list(runs_dir.rglob("best.pt"))
            if not candidates:
                raise FileNotFoundError(f"Khong tim thay bat ky checkpoint best.pt nao trong {runs_dir}")
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            chosen_ckpt = candidates[0]
            print(f"[+] Da tu dong chon checkpoint moi nhat: {chosen_ckpt}")

    print(f"[+] Checkpoint duoc su dung: {chosen_ckpt}")

    # 3. Chay du doan voi bo tham so toi uu
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] Load model YOLO tu {chosen_ckpt} (device={device})...")
    model = YOLO(str(chosen_ckpt))

    print("[+] CAU HINH TOI UU HOA SUY LUAN:")
    print("    - conf = 0.10 (ha nguong tu tin de tang Recall)")
    print("    - iou = 0.55 (tang nguong de giu vat the sat nhau)")
    print("    - augment = True (bat Test-Time Augmentation - TTA)")
    print("    - imgsz = 640 (nang cao do phan giai)")
    print(f"[+] Dang du doan Detection tren toan bo anh tai {test_img_dir} (stream=True)...")

    detection_preds = {}
    results = model.predict(
        source=str(test_img_dir),
        stream=True,
        imgsz=640,
        conf=0.10,
        iou=0.55,
        augment=True,
        device=device,
        verbose=False,
    )

    count_det = 0
    for r in results:
        sample_id = Path(r.path).stem
        json_pred = format_detection_result(r)
        detection_preds[sample_id] = json_pred
        count_det += 1
        if count_det % 1000 == 0:
            print(f"[+] Da du doan xong {count_det}/4000 anh Detection...")

    print(f"[+] Hoan thanh du doan {count_det} anh Detection.")

    # 4. Doc ket qua Intent Predictions
    print(f"[+] Dang doc du lieu Intent tu: {intent_pred_path}")
    df_intent = pd.read_csv(intent_pred_path, encoding="utf-8")
    intent_preds = dict(zip(df_intent["sample_id"], df_intent["prediction"]))
    print(f"[+] So luong mau Intent doc duoc: {len(intent_preds)}")

    # 5. Ghep ket qua theo thu tu sample_submission.csv
    print("[+] Dang ghep ket qua theo thu tu chuan...")
    final_rows = []
    missing_count = 0

    for idx, row in df_sample.iterrows():
        t_type = row["task_type"]
        s_id = row["sample_id"]

        if t_type == "DETECTION":
            pred = detection_preds.get(s_id, "[]")
        elif t_type == "INTENT":
            pred = intent_preds.get(s_id, "")
            if not pred:
                missing_count += 1
                pred = "OUT_OF_SCOPE"
        else:
            pred = ""

        final_rows.append({
            "task_type": t_type,
            "sample_id": s_id,
            "prediction": pred,
        })

    df_submission = pd.DataFrame(final_rows)

    # Xuat ra file submission_opt_conf10_tta.csv
    df_submission.to_csv(opt_sub_path, index=False, encoding="utf-8")
    print(f"[+] Da xuat file submission toi uu tai: {opt_sub_path}")

    # Ghi de vao file submission.csv
    df_submission.to_csv(main_sub_path, index=False, encoding="utf-8")
    print(f"[+] Da ghi de vao file submission chinh tai: {main_sub_path}")

    # 6. Kiem tra xac thuc va thong ke so bounding boxes
    print("\n" + "=" * 60)
    print("KIEM TRA XAC THUC VA THONG KE SO SANH")
    print("=" * 60)

    with open(main_sub_path, "r", encoding="utf-8") as f:
        total_file_lines = sum(1 for _ in f)
    print(f"[+] Tong so dong trong file (tinh ca header): {total_file_lines} (Yeu cau: 8001)")

    null_count = df_submission["prediction"].isnull().sum()
    empty_count = (df_submission["prediction"] == "").sum()
    print(f"[+] So gia tri null: {null_count}")
    print(f"[+] So gia tri rong: {empty_count}")

    # Thong ke bounding boxes moi
    det_new = df_submission[df_submission["task_type"] == "DETECTION"]
    new_box_count, new_empty_imgs = count_boxes(det_new)

    print(f"\n[+] THONG KE BOUNDING BOXES:")
    print(f"    - So boxes truoc toi uu: {old_box_count:,}")
    print(f"    - So boxes sau toi uu:   {new_box_count:,}")
    diff_boxes = new_box_count - old_box_count
    pct_increase = (diff_boxes / old_box_count * 100) if old_box_count > 0 else 0
    print(f"    - Tang them: +{diff_boxes:,} boxes (+{pct_increase:.2f}%)")
    print(f"    - So anh co 0 vat the: {new_empty_imgs}")

    print("\n[+] 5 dong dau tien:")
    print(df_submission.head())
    print("\n[+] 5 dong cuoi cung:")
    print(df_submission.tail())
    print("=" * 60)


if __name__ == "__main__":
    main()
