import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11n for Edge AI Challenge")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke", "fast_full", "full"],
        default="fast_full",
        help="Training mode: 'smoke' (2 epochs), 'fast_full' (20 epochs), or 'full' (50 epochs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    print("=" * 60)
    print(f"BAT DAU HUAN LUYEN YOLO11n - CHE DO: {mode.upper()}")
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

    # Cau hinh tham so theo mode
    patience = 100
    if mode == "smoke":
        epochs = 2
        name = "yolo11n_qualifiers"
        print("[+] Che do SMOKE TEST: epochs=2, imgsz=320, batch=8, workers=0, amp=True")
    elif mode == "fast_full":
        epochs = 20
        patience = 7
        name = "yolo11n_qualifiers_full"
        print("[+] Che do FAST FULL TRAINING: epochs=20, patience=7, imgsz=320, batch=8, workers=0, amp=True")
    else:
        epochs = 50
        name = "yolo11n_qualifiers_full"
        print("[+] Che do FULL TRAINING: epochs=50, imgsz=320, batch=8, workers=0, amp=True")

    imgsz = 320
    batch = 8
    workers = 0
    amp = True

    # Khoi tao model YOLO11n
    print("[+] Dang khoi tao model yolo11n.pt...")
    model = YOLO("yolo11n.pt")

    # Tien hanh train
    print(f"[+] Bat dau qua trinh train tren device: {device} (name={name})...")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(base_path / "runs"),
        name=name,
        exist_ok=True,
        amp=amp,
    )

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
