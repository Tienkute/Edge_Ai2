import argparse
import json
import os
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Tao file submission cho Edge AI Challenge")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11_micro_best.pt",
        help="Duong dan den file trong so YOLO (mac dinh: yolo11_micro_best.pt tai thu muc goc)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=160,
        help="Kich thuoc anh suy luan (mac dinh: 160)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Nguong confidence (mac dinh: 0.001 de vet toi da Recall cho mAP50)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.60,
        help="Nguong NMS IoU (mac dinh: 0.60)",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=100,
        help="So luong bounding box toi da moi anh (mac dinh: 100)",
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Thiet bi suy luan (mac dinh: 0)",
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

    # Chuyen thu muc lam viec ve goc repo
    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    print("=" * 60)
    print("SUY LUAN DETECTION YOLO11-MICRO VA GHEP FILE SUBMISSION")
    print("=" * 60)

    sample_sub_path = base_dir / "data" / "sample_submission.csv"
    intent_pred_path = base_dir / "outputs" / "predictions" / "intent_predictions.csv"
    test_img_dir = base_dir / "data" / "test" / "images"
    sub_dir = base_dir / "outputs" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    micro_sub_path = sub_dir / "submission_yolo11_micro.csv"
    main_sub_path = sub_dir / "submission.csv"

    # 1. Kiem tra file sample_submission.csv
    print(f"[+] KIEM TRA SAMPLE_SUBMISSION: {sample_sub_path}")
    df_sample = pd.read_csv(sample_sub_path)
    print(f"[+] SO LUONG DONG SAMPLE_SUBMISSION: {len(df_sample)}")

    # 2. Xac dinh checkpoint YOLO (tro truc tiep yolo11_micro_best.pt o goc repo)
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = base_dir / weights_path

    if not weights_path.exists():
        print(f"[!] KHONG TIM THAY: {weights_path}, DANG TIM KIEM TRONG RUNS...")
        candidates = list((base_dir / "runs").rglob("best.pt"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            weights_path = candidates[0]
            print(f"[+] DA CHON CHECKPOINT: {weights_path}")
        else:
            raise FileNotFoundError(f"Khong tim thay checkpoint YOLO tai {weights_path}")

    print(f"[+] CHECKPOINT DUOC SU DUNG: {weights_path}")

    # 3. Chay suy luan voi cau hinh toi uu Recall cho mAP50
    # imgsz=160, conf=0.001, iou=0.60, max_det=100, augment=True
    device = args.device
    if device == 0 and not torch.cuda.is_available():
        print("[!] CUDA KHONG KHA DUNG, CHUYEN SANG CPU!")
        device = "cpu"

    print(f"[+] LOAD MODEL YOLO TU: {weights_path} (DEVICE={device})...")
    model = YOLO(str(weights_path))

    print(f"[+] CAU HINH SUY LUAN: IMGSZ={args.imgsz}, CONF={args.conf}, IOU={args.iou}, MAX_DET={args.max_det}, AUGMENT=TRUE, DEVICE={device}")
    print(f"[+] DANG DU DOAN DETECTION TREN TOAN BO ANH TAI: {test_img_dir}...")

    detection_preds = {}
    results = model.predict(
        source=str(test_img_dir),
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
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
            print(f"[+] DA DU DOAN XONG {count_det}/4000 ANH DETECTION...")

    print(f"[+] HOAN THANH DU DOAN {count_det} ANH DETECTION.")

    # 4. Doc ket qua Intent Predictions (TextCNN)
    print(f"[+] DANG DOC DU LIEU INTENT TU: {intent_pred_path}")
    if intent_pred_path.exists():
        df_intent = pd.read_csv(intent_pred_path, encoding="utf-8")
        intent_preds = dict(zip(df_intent["sample_id"].astype(str), df_intent["prediction"]))
        print(f"[+] SO LUONG MAU INTENT DOC DUOC: {len(intent_preds)}")
    else:
        print(f"[!] KHONG TIM THAY FILE INTENT: {intent_pred_path}, SE GAN MAC DINH OUT_OF_SCOPE")
        intent_preds = {}

    # 5. Ghep ket qua theo thu tu chuan cua sample_submission.csv
    print("[+] DANG GHEP KET QUA THEO THU TU CHUAN CUA SAMPLE_SUBMISSION...")
    final_rows = []
    missing_count = 0

    for idx, row in df_sample.iterrows():
        t_type = row["task_type"]
        s_id = str(row["sample_id"])

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

    # Xuat file submission_yolo11_micro.csv
    df_submission.to_csv(micro_sub_path, index=False, encoding="utf-8")
    print(f"[+] DA XUAT FILE SUBMISSION YOLO11-MICRO TAI: {micro_sub_path}")

    # Dong thoi cap nhat vao file submission.csv chinh
    df_submission.to_csv(main_sub_path, index=False, encoding="utf-8")
    print(f"[+] DA CAP NHAT FILE SUBMISSION CHINH TAI: {main_sub_path}")

    # 6. Kiem tra xac thuc va thong ke
    print("\n" + "=" * 60)
    print("KIEM TRA XAC THUC VA THONG KE SUBMISSION")
    print("=" * 60)

    with open(micro_sub_path, "r", encoding="utf-8") as f:
        total_file_lines = sum(1 for _ in f)
    print(f"[+] TONG SO DONG TRONG FILE (TINH CA HEADER): {total_file_lines} (YEU CAU: 8001)")

    null_count = df_submission["prediction"].isnull().sum()
    empty_count = (df_submission["prediction"] == "").sum()
    print(f"[+] SO GIA TRI NULL: {null_count}")
    print(f"[+] SO GIA TRI RONG: {empty_count}")

    det_part = df_submission[df_submission["task_type"] == "DETECTION"]
    total_boxes, zero_box_imgs = count_boxes(det_part)
    print(f"[+] TONG SO BOUNDING BOXES DETECTION: {total_boxes:,}")
    print(f"[+] SO ANH DETECTION CO 0 VAT THE: {zero_box_imgs}")

    print("\n[+] 5 DONG DAU TIEN:")
    print(df_submission.head())
    print("\n[+] 5 DONG CUOI CUNG:")
    print(df_submission.tail())
    print("=" * 60)
    print("[+] HOAN TAT QUA TRINH TAO FILE SUBMISSION!")
    print("=" * 60)


if __name__ == "__main__":
    main()
