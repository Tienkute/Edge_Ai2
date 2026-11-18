import argparse
import os
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huan luyen YOLO11-micro Ban V12 (Copy-Paste Augmentation & Box Loss 15.0)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Duong dan checkpoint khoi tao (mac dinh: tu tim v11_best -> v10_best -> micro_best -> yolo11n.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset_v10_pseudo.yaml",
        help="File yaml dataset (mac dinh: dataset_v10_pseudo.yaml hoac dataset.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="So epoch huan luyen (mac dinh: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Batch size (mac dinh: 64)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Kich thuoc anh huan luyen (mac dinh: 320)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.0005,
        help="Learning rate khoi tao (mac dinh: 0.0005)",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=15.0,
        help="He so loss box (mac dinh: 15.0)",
    )
    parser.add_argument(
        "--dfl",
        type=float,
        default=2.0,
        help="He so loss dfl (mac dinh: 2.0)",
    )
    parser.add_argument(
        "--copy_paste",
        type=float,
        default=0.2,
        help="Xac suat copy-paste augmentation (mac dinh: 0.2)",
    )
    parser.add_argument(
        "--erasing",
        type=float,
        default=0.4,
        help="Xac suat random erasing augmentation (mac dinh: 0.4)",
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=30,
        help="So epoch cuoi tat mosaic (mac dinh: 30)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Patience early stopping (mac dinh: 40)",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Chi kiem tra cau hinh va checkpoint, khong chay train tren may ca nhan",
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Thiet bi huan luyen (mac dinh: 0)",
    )
    return parser.parse_args()


def resolve_checkpoint(base_dir, custom_weights=""):
    if custom_weights:
        p = Path(custom_weights)
        if not p.is_absolute():
            p = base_dir / p
        if p.exists():
            print(f"[+] SU DUNG CHECKPOINT CHI DINH: {p}")
            return str(p)

    # Uu tien 1: yolo11_micro_v11_best.pt
    v11_pt = base_dir / "yolo11_micro_v11_best.pt"
    if v11_pt.exists():
        print(f"[+] KHOI TAO TU CHECKPOINT V11: {v11_pt}")
        return str(v11_pt)

    # Uu tien 2: yolo11_micro_v10_best.pt
    v10_pt = base_dir / "yolo11_micro_v10_best.pt"
    if v10_pt.exists():
        print(f"[+] KHOI TAO TU CHECKPOINT V10: {v10_pt}")
        return str(v10_pt)

    # Uu tien 3: yolo11_micro_best.pt o goc repo
    micro_best = base_dir / "yolo11_micro_best.pt"
    if micro_best.exists():
        print(f"[+] KHOI TAO TU CHECKPOINT TOT NHAT HIEN TAI: {micro_best}")
        return str(micro_best)

    # Uu tien 4: Tim trong runs
    candidates = list((base_dir / "runs").rglob("best.pt"))
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"[+] KHOI TAO TU CHECKPOINT TRONG RUNS: {candidates[0]}")
        return str(candidates[0])

    # Uu tien 5: yolo11_micro_v12.yaml
    yaml_v12 = base_dir / "models" / "yolo11_micro_v12.yaml"
    print(f"[+] KHOI TAO MODEL TU FILE YAML: {yaml_v12}")
    return str(yaml_v12)


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent
    os.chdir(base_dir)

    print("=" * 60)
    print("[+] KHOI DONG TIEN TRINH HUAN LUYEN YOLO11-MICRO BAN V12")
    print("=" * 60)

    # 1. Kiem tra dataset yaml
    data_path = base_dir / args.data
    if not data_path.exists():
        fallback_data = base_dir / "dataset.yaml"
        if fallback_data.exists():
            print(f"[!] KHONG TIM THAY {data_path}, CHUYEN SANG: {fallback_data}")
            data_path = fallback_data
        else:
            raise FileNotFoundError(f"Khong tim thay file dataset yaml tai {data_path}")
    print(f"[+] DATASET YAML DUOC SU DUNG: {data_path}")

    # 2. Xac dinh checkpoint de fine-tune
    weights_path = resolve_checkpoint(base_dir, args.weights)

    # 3. Kiem tra thiet bi
    device = args.device
    if device == 0 and not torch.cuda.is_available():
        print("[!] CUDA KHONG KHA DUNG, CHUYEN SANG CPU!")
        device = "cpu"
    print(f"[+] THIET BI HUAN LUYEN: {device}")

    # 4. Khoi tao model
    print(f"[+] LOAD MODEL TU: {weights_path}...")
    if weights_path.endswith(".yaml"):
        pretrained = base_dir / "yolo11n.pt"
        if pretrained.exists():
            print(f"[+] NAP TRONG SO PRETRAINED TU: {pretrained}")
            model = YOLO(weights_path).load(str(pretrained))
        else:
            model = YOLO(weights_path)
    else:
        model = YOLO(weights_path)

    # 5. Thong bao cau hinh huan luyen Ban V12
    print("[+] CAU HINH CHIEN THUAT BAN V12:")
    print(f"    - IMGSZ = {args.imgsz}, EPOCHS = {args.epochs}, BATCH = {args.batch}")
    print(f"    - LR0 = {args.lr0}, COS_LR = TRUE")
    print(f"    - LOSS WEIGHTS: BOX = {args.box}, DFL = {args.dfl}, CLS = 0.7")
    print(f"    - AUGMENTATION: COPY_PASTE = {args.copy_paste}, ERASING = {args.erasing}")
    print(f"    - MOSAIC = 1.0, CLOSE_MOSAIC = {args.close_mosaic}, MIXUP = 0.1, AMP = TRUE")

    if args.prepare_only:
        print("=" * 60)
        print("[!] CHE DO PREPARE_ONLY: DA KIEM TRA CAU HINH THANH CONG!")
        print("[!] KHONG CHAY TRAIN TREN MAY CA NHAN THEO QUY TAC.")
        print("=" * 60)
        return

    # 6. Tien hanh train
    run_name = "yolo11_micro_esp32_v12"
    print(f"[+] BAT DAU FINE-TUNE RUN: {run_name}...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        cos_lr=True,
        box=args.box,
        dfl=args.dfl,
        cls=0.7,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=args.close_mosaic,
        device=device,
        project="runs",
        name=run_name,
        exist_ok=True,
        amp=True,
    )

    # 7. Kiem tra va dong bo trong so
    save_dir = base_dir / "runs" / run_name
    best_weights = save_dir / "weights" / "best.pt"
    v12_weights = base_dir / "yolo11_micro_v12_best.pt"
    root_weights = base_dir / "yolo11_micro_best.pt"

    print("=" * 60)
    print("[+] HUAN LUYEN BAN V12 HOAN TAT!")
    print(f"[+] THU MUC LUU KET QUA: {save_dir}")
    if best_weights.exists():
        print(f"[+] TRONG SO TOT NHAT (best.pt): {best_weights}")
        shutil.copy(str(best_weights), str(v12_weights))
        shutil.copy(str(best_weights), str(root_weights))
        print(f"[+] DA DONG BO TRONG SO VAO: {v12_weights} VA {root_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
