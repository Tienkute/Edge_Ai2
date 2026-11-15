import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Xuat mo hinh YOLO11-micro cho chip ESP32-S3")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/yolo11_micro_esp32_v6/weights/best.pt",
        help="Duong dan den file trong so best.pt cua yolo11_micro",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Kich thuoc anh dau vao (mac dinh: 320 cho ESP32-S3)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset.yaml",
        help="File cau hinh dataset dung de calibration INT8",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Chuyen thu muc lam viec ve goc repo
    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = base_dir / weights_path

    if not weights_path.exists():
        fallback_candidates = [
            base_dir / "yolo11_micro_best.pt",
            base_dir / "runs" / "yolo11_micro_esp32" / "weights" / "best.pt",
        ]
        chosen = next((p for p in fallback_candidates if p.exists()), None)
        if chosen:
            print(f"[!] KHONG TIM THAY: {weights_path}, CHUYEN SANG DUNG: {chosen}")
            weights_path = chosen
        else:
            print(f"[!] KHONG TIM THAY TRONG SO TAI: {weights_path}")
            print("[!] VUI LONG HUAN LUYEN TRUOC HOAC TRUYEN DUONG DAN QUA THAM SO --weights")
            return

    print("=" * 60)
    print("[+] BAT DAU XUAT MO HINH YOLO11-MICRO CHO ESP32-S3 (BAN V6 - 320PX)")
    print(f"[+] TRONG SO DAU VAO: {weights_path}")
    print(f"[+] KICH THUOC ANH (IMGSZ): {args.imgsz}")
    print("=" * 60)

    # Load model da huan luyen
    model = YOLO(str(weights_path))

    # 1. Xuat dinh dang ONNX (imgsz=320, simplify=True, opset=12)
    print(f"[+] DANG XUAT DINH DANG ONNX (IMGSZ={args.imgsz}, SIMPLIFY=TRUE, OPSET=12)...")
    try:
        onnx_file = model.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
            opset=12,
        )
        print(f"[+] XUAT ONNX THANH CONG: {onnx_file}")
    except Exception as e:
        print(f"[!] LOI KHI XUAT ONNX: {e}")

    # 2. Xuat dinh dang TFLite INT8 cho ESP32-S3
    print(f"[+] DANG XUAT DINH DANG TFLITE INT8 (IMGSZ={args.imgsz}, INT8=TRUE)...")
    try:
        tflite_file = model.export(
            format="tflite",
            imgsz=args.imgsz,
            int8=True,
            data=args.data,
        )
        print(f"[+] XUAT TFLITE INT8 THANH CONG: {tflite_file}")
    except Exception as e:
        print(f"[!] LOI KHI XUAT TFLITE: {e}")

    print("=" * 60)
    print("[+] HOAN TAT QUA TRINH XUAT MO HINH CHO ESP32-S3!")
    print("=" * 60)


if __name__ == "__main__":
    main()
