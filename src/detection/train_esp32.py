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
    print("[+] BAT DAU HUAN LUYEN YOLO11-MICRO BAN V3 (HOI TU SAU 160X160)")
    print("=" * 60)

    # 1. Kiem tra thiet bi
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] THIET BI HUAN LUYEN: {device}")

    # 2. Khoi tao model tu file yaml micro (khong load pretrained .pt de giu nguyen kien truc)
    model_yaml = "models/yolo11_micro.yaml"
    print(f"[+] KHOI TAO MODEL TU: {model_yaml}")
    model = YOLO(model_yaml)

    # 3. Thong bao cau hinh huan luyen Ban V3
    print("[+] CAU HINH CHIEN THUAT BAN V3:")
    print("    - imgsz = 160 (xuyen suot 300 epochs, khong train 320px gay soc mo hinh)")
    print("    - epochs = 300, patience = 40, batch = 64")
    print("    - optimizer = AdamW, lr0 = 0.003, lrf = 0.01, cos_lr = True")
    print("    - loss weights: box = 12.0, dfl = 2.0, cls = 0.8 (ep khung bao that khit)")
    print("    - augmentations: mosaic = 1.0, mixup = 0.1, degrees = 10.0, scale = 0.5, close_mosaic = 30")

    # 4. Tien hanh train
    run_name = "yolo11_micro_esp32_v3"
    results = model.train(
        data="dataset.yaml",
        epochs=300,
        patience=40,
        imgsz=160,
        batch=64,
        optimizer="AdamW",
        lr0=0.003,
        lrf=0.01,
        cos_lr=True,
        box=12.0,
        dfl=2.0,
        cls=0.8,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        scale=0.5,
        close_mosaic=30,
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
    print("[+] HUAN LUYEN BAN V3 HOAN TAT!")
    print(f"[+] THU MUC LUU KET QUA: {save_dir}")
    if best_weights.exists():
        print(f"[+] TRONG SO TOT NHAT (best.pt): {best_weights}")
        shutil.copy(str(best_weights), str(root_weights))
        print(f"[+] DA DONG BO TRONG SO VAO THU MUC GOC: {root_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
