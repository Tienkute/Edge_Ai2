import os
from pathlib import Path
from ultralytics import YOLO


def main():
    # Chuyen thu muc lam viec ve goc repo
    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    # Khoi tao model tu file yaml, khong dung pretrained weights de tranh bi ghi de kien truc
    model = YOLO("models/yolo11_micro.yaml")

    # In thong bao khoi tao thanh cong theo quy tac khong dau
    print("[+] DA KHOI TAO MODEL MICRO CHO ESP32-S3 VOI IMGSZ=160")

    # Huan luyen model voi cac tham so toi uu cho ESP32-S3
    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=160,
        batch=32,
        device=0,
        project="runs",
        name="yolo11_micro_esp32",
        amp=True,
    )

    print("[+] HUAN LUYEN HOAN TAT!")


if __name__ == "__main__":
    main()
