import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tao file submission V13 voi 10-View Multi-Scale & Flip WBF va Agnostic NMS"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Duong dan den file trong so YOLO (de trong se tu tim v13_best -> v12_best -> v11_best -> micro_best)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/test/images",
        help="Thu muc chua anh test (mac dinh: data/test/images)",
    )
    parser.add_argument(
        "--intent_csv",
        type=str,
        default="outputs/predictions/intent_predictions.csv",
        help="File du doan intent (mac dinh: outputs/predictions/intent_predictions.csv)",
    )
    parser.add_argument(
        "--sample_sub",
        type=str,
        default="data/sample_submission.csv",
        help="File sample submission (mac dinh: data/sample_submission.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/submissions/submission_yolo11_micro_v13.csv",
        help="Duong dan xuat file submission V13 (mac dinh: outputs/submissions/submission_yolo11_micro_v13.csv)",
    )
    parser.add_argument(
        "--iou_thr",
        type=float,
        default=0.58,
        help="Nguong IoU cho Weighted Boxes Fusion (mac dinh: 0.58)",
    )
    parser.add_argument(
        "--skip_box_thr",
        type=float,
        default=0.05,
        help="Nguong loc box co confidence thap truoc khi WBF (mac dinh: 0.05)",
    )
    parser.add_argument(
        "--conf_raw",
        type=float,
        default=0.01,
        help="Nguong conf khi goi model.predict de lay ung vien (mac dinh: 0.01)",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=30,
        help="So luong bounding box toi da moi anh sau fusion (mac dinh: 30)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Gioi han so luong anh detection suy luan de kiem tra nhanh (mac dinh: None)",
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Thiet bi suy luan (mac dinh: 0)",
    )
    return parser.parse_args()


def extract_norm_boxes(result, w, h):
    boxes_n = []
    scores = []
    labels = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes_n, scores, labels

    clss = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    xyxys = result.boxes.xyxy.cpu().numpy()

    for cls_id, conf, xyxy in zip(clss, confs, xyxys):
        x1, y1, x2, y2 = xyxy
        x1 = max(0.0, min(float(w), float(x1)))
        y1 = max(0.0, min(float(h), float(y1)))
        x2 = max(0.0, min(float(w), float(x2)))
        y2 = max(0.0, min(float(h), float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes_n.append([x1 / w, y1 / h, x2 / w, y2 / h])
        scores.append(float(conf))
        labels.append(int(cls_id))
    return boxes_n, scores, labels


def extract_norm_boxes_flip(result, w, h):
    boxes_n = []
    scores = []
    labels = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes_n, scores, labels

    clss = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    xyxys = result.boxes.xyxy.cpu().numpy()

    for cls_id, conf, xyxy in zip(clss, confs, xyxys):
        xf1, y1, xf2, y2 = xyxy
        # Giai lat toa do ve anh goc: x1 = w - x2, x2 = w - x1
        x1 = float(w) - float(xf2)
        x2 = float(w) - float(xf1)
        x1 = max(0.0, min(float(w), x1))
        y1 = max(0.0, min(float(h), float(y1)))
        x2 = max(0.0, min(float(w), x2))
        y2 = max(0.0, min(float(h), float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes_n.append([x1 / w, y1 / h, x2 / w, y2 / h])
        scores.append(float(conf))
        labels.append(int(cls_id))
    return boxes_n, scores, labels


def predict_single_image_10view_wbf(
    model,
    img,
    weights,
    conf_raw=0.01,
    iou_thr=0.58,
    skip_box_thr=0.05,
    max_det=30,
    device=0,
):
    h, w = img.shape[:2]

    # --- 5 GOC NHIN ANH GOC (ORIGINAL) ---
    # View 1: Anh goc, imgsz=224, agnostic_nms=True, augment=False (trong so 1.0)
    r1 = model.predict(img, imgsz=224, conf=conf_raw, agnostic_nms=True, augment=False, device=device, verbose=False)[0]
    b1, s1, l1 = extract_norm_boxes(r1, w, h)

    # View 2: Anh goc, imgsz=256, agnostic_nms=True, augment=False (trong so 1.2)
    r2 = model.predict(img, imgsz=256, conf=conf_raw, agnostic_nms=True, augment=False, device=device, verbose=False)[0]
    b2, s2, l2 = extract_norm_boxes(r2, w, h)

    # View 3: Anh goc, imgsz=320, agnostic_nms=True, augment=True (trong so 2.5 - do phan giai goc)
    r3 = model.predict(img, imgsz=320, conf=conf_raw, agnostic_nms=True, augment=True, device=device, verbose=False)[0]
    b3, s3, l3 = extract_norm_boxes(r3, w, h)

    # View 4: Anh goc, imgsz=352, agnostic_nms=True, augment=True (trong so 2.0)
    r4 = model.predict(img, imgsz=352, conf=conf_raw, agnostic_nms=True, augment=True, device=device, verbose=False)[0]
    b4, s4, l4 = extract_norm_boxes(r4, w, h)

    # View 5: Anh goc, imgsz=416, agnostic_nms=True, augment=True (trong so 1.5 - bat vat nho)
    r5 = model.predict(img, imgsz=416, conf=conf_raw, agnostic_nms=True, augment=True, device=device, verbose=False)[0]
    b5, s5, l5 = extract_norm_boxes(r5, w, h)

    # --- 5 GOC NHIN ANH LAT NGANG (HORIZONTAL FLIP) ---
    img_flip = cv2.flip(img, 1)

    # View 6: Anh lat ngang, imgsz=224, agnostic_nms=True, augment=False (trong so 1.0)
    r6 = model.predict(img_flip, imgsz=224, conf=conf_raw, agnostic_nms=True, augment=False, device=device, verbose=False)[0]
    b6, s6, l6 = extract_norm_boxes_flip(r6, w, h)

    # View 7: Anh lat ngang, imgsz=256, agnostic_nms=True, augment=False (trong so 1.2)
    r7 = model.predict(img_flip, imgsz=256, conf=conf_raw, agnostic_nms=True, augment=False, device=device, verbose=False)[0]
    b7, s7, l7 = extract_norm_boxes_flip(r7, w, h)

    # View 8: Anh lat ngang, imgsz=320, agnostic_nms=True, augment=False (trong so 2.0)
    r8 = model.predict(img_flip, imgsz=320, conf=conf_raw, agnostic_nms=True, augment=False, device=device, verbose=False)[0]
    b8, s8, l8 = extract_norm_boxes_flip(r8, w, h)

    # View 9: Anh lat ngang, imgsz=352, agnostic_nms=True, augment=False (trong so 1.5)
    r9 = model.predict(img_flip, imgsz=352, conf=conf_raw, agnostic_nms=True, augment=False, device=device, verbose=False)[0]
    b9, s9, l9 = extract_norm_boxes_flip(r9, w, h)

    # View 10: Anh lat ngang, imgsz=416, agnostic_nms=True, augment=False (trong so 1.2)
    r10 = model.predict(img_flip, imgsz=416, conf=conf_raw, agnostic_nms=True, augment=False, device=device, verbose=False)[0]
    b10, s10, l10 = extract_norm_boxes_flip(r10, w, h)

    boxes_list = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]
    scores_list = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
    labels_list = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]

    total_candidates = sum(len(b) for b in boxes_list)
    if total_candidates == 0:
        return "[]"

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    if len(fused_boxes) == 0:
        return "[]"

    objects = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1 = round(float(box[0] * w), 2)
        y1 = round(float(box[1] * h), 2)
        x2 = round(float(box[2] * w), 2)
        y2 = round(float(box[3] * h), 2)
        objects.append({
            "class_id": int(label),
            "score": round(float(score), 4),
            "bbox": [x1, y1, x2, y2],
        })

    if max_det is not None and max_det > 0:
        objects = objects[:max_det]

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


def resolve_weights(base_dir, custom_weights=""):
    if custom_weights:
        w_path = Path(custom_weights)
        if not w_path.is_absolute():
            w_path = base_dir / w_path
        if w_path.exists():
            return w_path

    # Thu tu tim kiem checkpoint uu tien
    for candidate_name in [
        "yolo11_micro_v13_best.pt",
        "yolo11_micro_v12_best.pt",
        "yolo11_micro_v11_best.pt",
        "yolo11_micro_v10_best.pt",
        "yolo11_micro_best.pt",
    ]:
        p = base_dir / candidate_name
        if p.exists():
            return p

    # Tim trong runs
    candidates = list((base_dir / "runs").rglob("best.pt"))
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]

    raise FileNotFoundError("Khong tim thay checkpoint YOLO hop le de suy luan!")


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    print("=" * 60)
    print("SUY LUAN DETECTION 10-VIEW WBF VA AGNOSTIC NMS (BAN V13)")
    print("=" * 60)

    sample_sub_path = Path(args.sample_sub)
    if not sample_sub_path.is_absolute():
        sample_sub_path = base_dir / sample_sub_path

    intent_pred_path = Path(args.intent_csv)
    if not intent_pred_path.is_absolute():
        intent_pred_path = base_dir / intent_pred_path

    test_img_dir = Path(args.test_dir)
    if not test_img_dir.is_absolute():
        test_img_dir = base_dir / test_img_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_sub_path = base_dir / "outputs" / "submissions" / "submission.csv"

    # 1. Kiem tra sample_submission.csv
    print(f"[+] KIEM TRA SAMPLE_SUBMISSION: {sample_sub_path}")
    if not sample_sub_path.exists():
        raise FileNotFoundError(f"Khong tim thay file sample_submission tai {sample_sub_path}")
    df_sample = pd.read_csv(sample_sub_path)
    print(f"[+] SO LUONG DONG SAMPLE_SUBMISSION: {len(df_sample)}")

    # 2. Xac dinh checkpoint YOLO
    weights_path = resolve_weights(base_dir, args.weights)
    print(f"[+] CHECKPOINT DUOC SU DUNG: {weights_path}")

    device = args.device
    if device == 0 and not torch.cuda.is_available():
        print("[!] CUDA KHONG KHA DUNG, CHUYEN SANG CPU!")
        device = "cpu"

    print(f"[+] LOAD MODEL YOLO TU: {weights_path} (DEVICE={device})...")
    model = YOLO(str(weights_path))

    # 10 Goc nhin (5 scales: 224, 256, 320, 352, 416 x Original & Flip)
    weights_10view = [1.0, 1.2, 2.5, 2.0, 1.5, 1.0, 1.2, 2.0, 1.5, 1.2]
    print("[+] CAU HINH SUY LUAN BAN V13 10-VIEW MULTI-SCALE & FLIP WBF + AGNOSTIC NMS:")
    print("    - VIEW 1: ANH GOC, IMGSZ=224, AGNOSTIC_NMS=TRUE, WEIGHT=1.0")
    print("    - VIEW 2: ANH GOC, IMGSZ=256, AGNOSTIC_NMS=TRUE, WEIGHT=1.2")
    print("    - VIEW 3: ANH GOC, IMGSZ=320, AGNOSTIC_NMS=TRUE, WEIGHT=2.5")
    print("    - VIEW 4: ANH GOC, IMGSZ=352, AGNOSTIC_NMS=TRUE, WEIGHT=2.0")
    print("    - VIEW 5: ANH GOC, IMGSZ=416, AGNOSTIC_NMS=TRUE, WEIGHT=1.5")
    print("    - VIEW 6: ANH LAT, IMGSZ=224, AGNOSTIC_NMS=TRUE, WEIGHT=1.0")
    print("    - VIEW 7: ANH LAT, IMGSZ=256, AGNOSTIC_NMS=TRUE, WEIGHT=1.2")
    print("    - VIEW 8: ANH LAT, IMGSZ=320, AGNOSTIC_NMS=TRUE, WEIGHT=2.0")
    print("    - VIEW 9: ANH LAT, IMGSZ=352, AGNOSTIC_NMS=TRUE, WEIGHT=1.5")
    print("    - VIEW 10: ANH LAT, IMGSZ=416, AGNOSTIC_NMS=TRUE, WEIGHT=1.2")
    print(f"    - WBF PARAMS: IOU_THR={args.iou_thr}, SKIP_BOX_THR={args.skip_box_thr}, CONF_RAW={args.conf_raw}")

    # Doc du lieu Intent Predictions (TextCNN)
    print(f"[+] DANG DOC DU LIEU INTENT TU: {intent_pred_path}")
    if intent_pred_path.exists():
        df_intent = pd.read_csv(intent_pred_path, encoding="utf-8")
        intent_preds = dict(zip(df_intent["sample_id"].astype(str), df_intent["prediction"]))
        print(f"[+] SO LUONG MAU INTENT DOC DUOC: {len(intent_preds)}")
    else:
        print(f"[!] KHONG TIM THAY FILE INTENT: {intent_pred_path}, SE GAN MAC DINH OUT_OF_SCOPE")
        intent_preds = {}

    # Tim tat ca cac anh test
    image_file_map = {}
    if test_img_dir.exists():
        for p in test_img_dir.iterdir():
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                image_file_map[p.stem] = p
    print(f"[+] TIM THAY {len(image_file_map)} ANH TRONG {test_img_dir}")

    # Danh sach detection samples tu sample_submission
    det_samples = df_sample[df_sample["task_type"] == "DETECTION"]["sample_id"].astype(str).tolist()
    total_det = len(det_samples)
    print(f"[+] TONG SO MAU DETECTION CAN DU DOAN: {total_det}")

    if args.limit is not None and args.limit > 0:
        print(f"[!] CHE DO KIEM TRA NHANH: CHI CHAY CHO {args.limit} ANH DAU TIEN")

    detection_preds = {}
    print("[+] DANG TIEN HANH SUY LUAN DETECTION 10-VIEW WBF + AGNOSTIC NMS...")
    count_det = 0
    for idx, s_id in enumerate(det_samples):
        if args.limit is not None and idx >= args.limit:
            detection_preds[s_id] = "[]"
            continue

        img_path = image_file_map.get(s_id)
        if img_path is None or not img_path.exists():
            detection_preds[s_id] = "[]"
            count_det += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            detection_preds[s_id] = "[]"
            count_det += 1
            continue

        pred_json = predict_single_image_10view_wbf(
            model=model,
            img=img,
            weights=weights_10view,
            conf_raw=args.conf_raw,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
            max_det=args.max_det,
            device=device,
        )
        detection_preds[s_id] = pred_json
        count_det += 1
        if count_det % 100 == 0:
            print(f"[+] DA SUY LUAN XONG {count_det}/{total_det} ANH...")

    print(f"[+] HOAN THANH SUY LUAN DETECTION TREN {count_det} ANH.")

    # Ghep ket qua theo thu tu chuan cua sample_submission
    print("[+] DANG GHEP KET QUA THEO THU TU CHUAN CUA SAMPLE_SUBMISSION...")
    final_rows = []
    missing_intent = 0

    for idx, row in df_sample.iterrows():
        t_type = row["task_type"]
        s_id = str(row["sample_id"])

        if t_type == "DETECTION":
            pred = detection_preds.get(s_id, "[]")
        elif t_type == "INTENT":
            pred = intent_preds.get(s_id, "")
            if not pred:
                missing_intent += 1
                pred = "OUT_OF_SCOPE"
        else:
            pred = ""

        final_rows.append({
            "task_type": t_type,
            "sample_id": s_id,
            "prediction": pred,
        })

    df_submission = pd.DataFrame(final_rows)

    # Xuat file submission V13
    df_submission.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[+] DA XUAT FILE SUBMISSION YOLO11-MICRO V13 TAI: {output_path}")

    # Dong thoi cap nhat vao file submission.csv chinh
    df_submission.to_csv(main_sub_path, index=False, encoding="utf-8")
    print(f"[+] DA CAP NHAT FILE SUBMISSION CHINH TAI: {main_sub_path}")

    # Kiem tra xac thuc va thong ke
    print("\n" + "=" * 60)
    print("KIEM TRA XAC THUC VA THONG KE SUBMISSION")
    print("=" * 60)

    with open(output_path, "r", encoding="utf-8") as f:
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
    print("[+] HOAN TAT QUA TRINH TAO FILE SUBMISSION V13 WBF!")
    print("=" * 60)


if __name__ == "__main__":
    main()
