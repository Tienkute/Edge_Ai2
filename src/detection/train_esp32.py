import os
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    # Chuyen thu muc lam viec ve goc repository
    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    # Khoi tao model tu file yaml micro, khong load pretrained pt de tranh bi ghi de kien truc
    model_config = "models/yolo11_micro.yaml"
    model = YOLO(model_config)

    # Thong bao khoi tao model bang tieng Viet khong dau
    print("[+] DA KHOI TAO MODEL MICRO CHO ESP32-S3 VOI IMGSZ=160")

    # Kiem tra thiet bi huan luyen
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] THIET BI HUAN LUYEN: {device}")

    # Thiet lap cau hinh huan luyen hoi tu sau cho mang sieu nho (< 200k params)
    print("[+] BAT DAU HUAN LUYEN HOI TU SAU CHO YOLO11-MICRO ESP32-S3...")
    print("[+] CAU HINH: IMGSZ=160, EPOCHS=200, BATCH=64, PATIENCE=30, COS_LR=TRUE")
    print("[+] TANG CUONG DU LIEU: BOX=8.5, CLS=0.7, MOSAIC=1.0, MIXUP=0.1, CLOSE_MOSAIC=20, AMP=TRUE")

    results = model.train(
        data="dataset.yaml",
        epochs=200,
        patience=30,
        imgsz=160,
        batch=64,
        device=device,
        project="runs",
        name="yolo11_micro_esp32",
        exist_ok=True,
        amp=True,
        cos_lr=True,
        box=8.5,
        cls=0.7,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=20,
    )

    save_dir = Path(base_dir) / "runs" / "yolo11_micro_esp32"
    print("[+] HUAN LUYEN HOAN TAT!")
    print(f"[+] KET QUA DUOC LUU TAI: {save_dir}")


if __name__ == "__main__":
    main()
