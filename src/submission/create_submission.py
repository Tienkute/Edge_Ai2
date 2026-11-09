import json
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO


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


def main():
    print("=" * 60)
    print("NHIEM VU 7: GHEP KET QUA VA XUAT SUBMISSION.CSV CHUAN KAGGLE")
    print("=" * 60)

    base_dir = Path("D:/visual code/python/Edge_Ai2")
    sample_sub_path = base_dir / "data" / "sample_submission.csv"
    intent_pred_path = base_dir / "outputs" / "predictions" / "intent_predictions.csv"
    test_img_dir = base_dir / "data" / "test" / "images"
    sub_dir = base_dir / "outputs" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    final_sub_path = sub_dir / "submission.csv"

    # 1. Kiem tra file sample_submission.csv
    print(f"[+] Kiem tra sample_submission: {sample_sub_path}")
    df_sample = pd.read_csv(sample_sub_path)
    print(f"[+] So luong dong sample_submission: {len(df_sample)}")
    print(f"[+] Cac cot: {df_sample.columns.tolist()}")

    # 2. Xac dinh checkpoint YOLO tot nhat
    ckpt_fast_full = base_dir / "runs" / "yolo11n_qualifiers_full" / "weights" / "best.pt"
    ckpt_smoke = base_dir / "runs" / "yolo11n_qualifiers" / "weights" / "best.pt"

    if ckpt_fast_full.exists():
        chosen_ckpt = ckpt_fast_full
        print(f"[+] Su dung checkpoint YOLO full: {chosen_ckpt}")
    elif ckpt_smoke.exists():
        chosen_ckpt = ckpt_smoke
        print(f"[+] Su dung checkpoint YOLO smoke: {chosen_ckpt}")
    else:
        raise FileNotFoundError(f"Khong tim thay checkpoint YOLO tai {ckpt_fast_full} hoac {ckpt_smoke}")

    # 3. Chay du doan Detection tren tap test images
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] Load model YOLO tu {chosen_ckpt} (device={device})...")
    model = YOLO(str(chosen_ckpt))

    print(f"[+] Dang du doan Detection tren toan bo anh tai {test_img_dir} (stream=True)...")
    detection_preds = {}

    results = model.predict(
        source=str(test_img_dir),
        stream=True,
        imgsz=320,
        conf=0.25,
        iou=0.45,
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
    df_submission.to_csv(final_sub_path, index=False, encoding="utf-8")
    print(f"[+] Da xuat file submission tai: {final_sub_path}")
    print(f"[+] Tong so dong (khong tinh header): {len(df_submission)}")

    # 6. Kiem tra xac thuc file submission
    print("\n" + "=" * 60)
    print("KIEM TRA XAC THUC SUBMISSION.CSV")
    print("=" * 60)
    with open(final_sub_path, "r", encoding="utf-8") as f:
        total_file_lines = sum(1 for _ in f)
    print(f"[+] Tong so dong trong file (tinh ca header): {total_file_lines} (Yeu cau: 8001)")

    null_count = df_submission["prediction"].isnull().sum()
    empty_count = (df_submission["prediction"] == "").sum()
    print(f"[+] So gia tri null: {null_count}")
    print(f"[+] So gia tri rong: {empty_count}")

    print("\n[+] 5 dong dau tien:")
    print(df_submission.head())
    print("\n[+] 5 dong cuoi cung:")
    print(df_submission.tail())
    print("=" * 60)


if __name__ == "__main__":
    main()

