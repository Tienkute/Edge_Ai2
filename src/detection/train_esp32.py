import os
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


def main():
    # Chuyen thu muc lam viec ve goc repository
    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    print("=" * 60)
    print("[+] BAT DAU HUAN LUYEN YOLO11-MICRO BAN V6 (320PX PRETRAINED TRANSFER)")
    print("=" * 60)

    # 1. Kiem tra thiet bi
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] THIET BI HUAN LUYEN: {device}")

    # 2. Khoi tao model tu yaml va load trong so khoi tao tu yolo11n.pt
    model_yaml = "models/yolo11_micro.yaml"
    pretrained_weights = "yolo11n.pt"
    print(f"[+] KHOI TAO MODEL TU: {model_yaml}")
    print(f"[+] LOAD TRONG SO KHOI TAO TU: {pretrained_weights}")
    model = YOLO(model_yaml).load(pretrained_weights)

    # 3. Thong bao cau hinh huan luyen Ban V6
    print("[+] CAU HINH CHIEN THUAT BAN V6:")
    print("    - imgsz = 320 (toi da hoa do phan giai cho chip ESP32-S3)")
    print("    - epochs = 300, patience = 50, batch = 64")
    print("    - optimizer = auto (mac dinh), cos_lr = True")
    print("    - loss weights: box = 8.5, cls = 0.7")
    print("    - augmentations: mosaic = 1.0, mixup = 0.1, degrees = 10.0, scale = 0.5, close_mosaic = 40, amp = True")

    # 4. Tien hanh train
    run_name = "yolo11_micro_esp32_v6"
    results = model.train(
        data="dataset.yaml",
        epochs=300,
        patience=50,
        imgsz=320,
        batch=64,
        cos_lr=True,
        box=8.5,
        cls=0.7,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        scale=0.5,
        close_mosaic=40,
        device=device,
        project="runs",
        name=run_name,
        exist_ok=True,
        amp=True,
    )

    # 5. Kiem tra va dong bo trong so tot nhat
    save_dir = base_dir / "runs" / run_name
    best_weights = save_dir / "weights" / "best.pt"
    root_weights = base_dir / "yolo11_micro_best.pt"

    print("=" * 60)
    print("[+] HUAN LUYEN BAN V6 HOAN TAT!")
    print(f"[+] THU MUC LUU KET QUA: {save_dir}")
    if best_weights.exists():
        print(f"[+] TRONG SO TOT NHAT (best.pt): {best_weights}")
        shutil.copy(str(best_weights), str(root_weights))
        print(f"[+] DA DONG BO TRONG SO VAO THU MUC GOC: {root_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
