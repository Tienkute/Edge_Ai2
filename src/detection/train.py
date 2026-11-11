import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO for Edge AI Challenge")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke", "fast_full", "full", "yolo11s"],
        default="fast_full",
        help="Training mode: 'smoke' (2 epochs), 'fast_full' (20 epochs), 'full' (50 epochs), or 'yolo11s' (70 epochs, imgsz 640)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    print("=" * 60)
    print(f"BAT DAU HUAN LUYEN YOLO - CHE DO: {mode.upper()}")
    print("=" * 60)

    base_path = Path("D:/visual code/python/Edge_Ai2")
    yaml_path = base_path / "dataset.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found at: {yaml_path}")

    # Kiem tra thiet bi
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[+] Su dung GPU: {gpu_name}")
        device = 0
    else:
        print("[!] Khong tim thay CUDA GPU, su dung CPU!")
        device = "cpu"

    workers = 0
    amp = True

    # Cau hinh tham so theo tung mode
    if mode == "smoke":
        model_name = "yolo11n.pt"
        epochs = 2
        patience = 100
        imgsz = 320
        batch = 8
        name = "yolo11n_qualifiers"
        extra_train_kwargs = {}
        print("[+] Che do SMOKE TEST: epochs=2, imgsz=320, batch=8, workers=0, amp=True")
    elif mode == "fast_full":
        model_name = "yolo11n.pt"
        epochs = 20
        patience = 7
        imgsz = 320
        batch = 8
        name = "yolo11n_qualifiers_full"
        extra_train_kwargs = {}
        print("[+] Che do FAST FULL TRAINING: epochs=20, patience=7, imgsz=320, batch=8, workers=0, amp=True")
    elif mode == "full":
        model_name = "yolo11n.pt"
        epochs = 50
        patience = 15
        imgsz = 320
        batch = 8
        name = "yolo11n_qualifiers_full"
        extra_train_kwargs = {}
        print("[+] Che do FULL TRAINING: epochs=50, patience=15, imgsz=320, batch=8, workers=0, amp=True")
    elif mode == "yolo11s":
        model_name = "yolo11s.pt"
        epochs = 70
        patience = 15
        imgsz = 640
        batch = 16
        name = "yolo11s_qualifiers"
        extra_train_kwargs = {
            "box": 8.5,
            "cls": 0.7,
            "close_mosaic": 10,
            "mosaic": 1.0,
            "mixup": 0.15,
            "degrees": 10.0,
            "scale": 0.5,
        }
        print("[+] Che do YOLO11s (SMALL): model=yolo11s.pt, epochs=70, patience=15, imgsz=640, batch=16, amp=True")
        print("    Augmentations: box=8.5, cls=0.7, close_mosaic=10, mosaic=1.0, mixup=0.15, degrees=10.0, scale=0.5")

    # Khoi tao model
    print(f"[+] Dang khoi tao model {model_name}...")
    model = YOLO(model_name)

    # Tien hanh train
    print(f"[+] Bat dau qua trinh train tren device: {device} (name={name})...")
    train_args = {
        "data": str(yaml_path),
        "epochs": epochs,
        "patience": patience,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "project": str(base_path / "runs"),
        "name": name,
        "exist_ok": True,
        "amp": amp,
    }
    train_args.update(extra_train_kwargs)

    results = model.train(**train_args)

    save_dir = Path(base_path / "runs" / name)
    weights_path = save_dir / "weights" / "best.pt"
    last_weights_path = save_dir / "weights" / "last.pt"

    print("\n" + "=" * 60)
    print("[+] HUAN LUYEN HOAN TAT!")
    print(f"[+] Thu muc luu ket qua: {save_dir}")
    if weights_path.exists():
        print(f"[+] Trong so tot nhat (best.pt): {weights_path}")
    if last_weights_path.exists():
        print(f"[+] Trong so cuoi cung (last.pt): {last_weights_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
