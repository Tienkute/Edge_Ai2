import os
import random
from pathlib import Path

# Đường dẫn dữ liệu
base_path = Path("D:/visual code/python/Edge_Ai2")
train_img_dir = base_path / "data/train/images"
train_txt = base_path / "data/train_images.txt"
val_txt = base_path / "data/val_images.txt"

# Lấy danh sách toàn bộ ảnh
all_images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
all_images = [str(img.resolve()) for img in all_images]

# Chia 80% train, 20% validation voi random seed co dinh
random.seed(42)
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

# Ghi ra file text để YOLO sử dụng
with open(train_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_imgs))

with open(val_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_imgs))

print(f"Done! Train: {len(train_imgs)} images, Val: {len(val_imgs)} images.")