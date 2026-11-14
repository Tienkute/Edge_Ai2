import os
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    # Chuyen thu muc lam viec ve goc repository
    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    # Kiem tra thiet bi
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] THIET BI HUAN LUYEN: {device}")

    # =========================================================================
    # GIAI DOAN 1: HOC DAC TRUNG RO NET (IMGSZ=320, 120 EPOCHS)
    # =========================================================================
    print("=" * 60)
    print("[+] BAT DAU CHIEN THUAT PROGRESSIVE SHRINKING CHO YOLO11-MICRO")
    print("[+] GIAI DOAN 1: HOC DAC TRUNG RO NET (IMGSZ=320, 120 EPOCHS)")
    print("=" * 60)

    model_yaml = "models/yolo11_micro.yaml"
    print(f"[+] KHOI TAO MODEL MICRO TU: {model_yaml}")
    model_stage1 = YOLO(model_yaml)

    print("[+] CAU HINH GIAI DOAN 1: IMGSZ=320, EPOCHS=120, BATCH=64, COS_LR=TRUE")
    print("[+] AUGMENTATION: BOX=10.0, CLS=0.7, MOSAIC=1.0, MIXUP=0.0, DEGREES=0.0, CLOSE_MOSAIC=20")

    model_stage1.train(
        data="dataset.yaml",
        epochs=120,
        imgsz=320,
        batch=64,
        device=device,
        project="runs",
        name="yolo11_micro_stage1_320",
        exist_ok=True,
        amp=True,
        cos_lr=True,
        box=10.0,
        cls=0.7,
        mosaic=1.0,
        mixup=0.0,
        degrees=0.0,
        close_mosaic=20,
    )

    stage1_best = base_dir / "runs" / "yolo11_micro_stage1_320" / "weights" / "best.pt"
    print(f"[+] HOAN TAT GIAI DOAN 1! TRONG SO TOT NHAT: {stage1_best}")

    # =========================================================================
    # GIAI DOAN 2: EP CHUAN ESP32 160X160 (IMGSZ=160, 150 EPOCHS, LR0=0.002)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[+] GIAI DOAN 2: EP CHUAN ESP32 160X160 (FINE-TUNE IMGSZ=160, 150 EPOCHS)")
    print("=" * 60)

    if stage1_best.exists():
        print(f"[+] LOAD TRONG SO GIAI DOAN 1 TU: {stage1_best}")
        model_stage2 = YOLO(str(stage1_best))
    else:
        print("[!] CANH BAO: KHONG TIM THAY BEST.PT CUA GIAI DOAN 1, KHOI TAO TU YAML")
        model_stage2 = YOLO(model_yaml)

    print("[+] CAU HINH GIAI DOAN 2: IMGSZ=160, EPOCHS=150, BATCH=64, LR0=0.002, COS_LR=TRUE")
    print("[+] AUGMENTATION: BOX=10.0, CLS=0.7, MOSAIC=0.5, MIXUP=0.0, CLOSE_MOSAIC=30")

    model_stage2.train(
        data="dataset.yaml",
        epochs=150,
        imgsz=160,
        batch=64,
        lr0=0.002,
        device=device,
        project="runs",
        name="yolo11_micro_stage2_160",
        exist_ok=True,
        amp=True,
        cos_lr=True,
        box=10.0,
        cls=0.7,
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=30,
    )

    stage2_best = base_dir / "runs" / "yolo11_micro_stage2_160" / "weights" / "best.pt"
    print(f"[+] HOAN TAT GIAI DOAN 2! TRONG SO TOT NHAT: {stage2_best}")

    # Sao chep hoac cap nhat trong so vao thu muc goc yolo11_micro_best.pt
    root_best = base_dir / "yolo11_micro_best.pt"
    if stage2_best.exists():
        import shutil
        shutil.copy(str(stage2_best), str(root_best))
        print(f"[+] DA DONG BO TRONG SO VAO THU MUC GOC: {root_best}")

    print("=" * 60)
    print("[+] TOAN BO CHIEN THUAT PROGRESSIVE SHRINKING HOAN TAT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
