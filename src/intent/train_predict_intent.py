import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Thiet lap seed de dam bao tinh tai lap
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Danh sach cac nhan intent theo thu tu co dinh
LABEL_LIST = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}

# Ham tach tu cho tieng Viet co ban
def tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return tokens

# Xay dung tu dien voi <PAD>=0, <UNK>=1
def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        tokens = tokenize(text)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

# Chuyen van ban thanh danh sach token id voi max_len=40
def text_to_indices(text, vocab, max_len=40):
    tokens = tokenize(text)
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(indices) < max_len:
        indices = indices + [vocab["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

# Dataset PyTorch cho Intent
class IntentDataset(Dataset):
    def __init__(self, texts, labels=None, vocab=None, max_len=40):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = text_to_indices(self.texts[idx], self.vocab, self.max_len)
        x = torch.tensor(indices, dtype=torch.long)
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x

# Kien truc mang TextCNN
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=5, kernel_sizes=[2, 3, 4, 5], num_filters=128, dropout_rate=0.3):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Cac nhanh Conv1d song song voi kernel sizes [2, 3, 4, 5]
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k),
                nn.BatchNorm1d(num_filters),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch_size, max_len)
        emb = self.embedding(x)  # (batch_size, max_len, embed_dim)
        emb = emb.permute(0, 2, 1)  # (batch_size, embed_dim, max_len)
        
        pooled_outputs = []
        for branch in self.conv_branches:
            conv_out = branch(emb)  # (batch_size, num_filters, out_len)
            pooled = torch.max(conv_out, dim=-1)[0]  # Global Max Pooling: (batch_size, num_filters)
            pooled_outputs.append(pooled)
            
        concat = torch.cat(pooled_outputs, dim=1)  # (batch_size, len(kernel_sizes) * num_filters)
        dropped = self.dropout(concat)
        logits = self.fc(dropped)  # (batch_size, num_classes)
        return logits

def main():
    set_seed(42)
    
    print("=" * 60)
    print("NHIEM VU 5: HUAN LUYEN TEXTCNN VA DU DOAN INTENT TEST")
    print("=" * 60)
    
    base_dir = Path("D:/visual code/python/Edge_Ai2")
    train_path = base_dir / "data" / "intent_train.csv"
    test_path = base_dir / "data" / "intent_test.csv"
    
    model_save_dir = base_dir / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_save_dir / "intent_textcnn_best.pt"
    vocab_save_path = model_save_dir / "vocab.json"
    
    pred_save_dir = base_dir / "outputs" / "predictions"
    pred_save_dir.mkdir(parents=True, exist_ok=True)
    pred_output_path = pred_save_dir / "intent_predictions.csv"
    
    # Kiem tra thiet bi
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Thiet bi tinh toan: {device}")
    if torch.cuda.is_available():
        print(f"[+] Ten GPU: {torch.cuda.get_device_name(0)}")

    # 1. Doc du lieu train va test
    print(f"[+] Dang doc du lieu tu: {train_path} va {test_path}...")
    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_test = pd.read_csv(test_path, encoding="utf-8")
    
    print(f"[+] So luong mau tap train: {len(df_train)}")
    print(f"[+] So luong mau tap test: {len(df_test)}")
    
    # 2. Xay dung tu dien tu tap train
    texts_train = df_train["question"].astype(str).tolist()
    labels_train = [LABEL2ID[lbl] for lbl in df_train["intent"]]
    
    vocab = build_vocab(texts_train)
    vocab_size = len(vocab)
    print(f"[+] Kich thuoc bo tu dien (Vocab size): {vocab_size} tu")
    
    # Luu tu dien
    with open(vocab_save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[+] Da luu bo tu dien vao: {vocab_save_path}")
    
    # 3. Chia tap train / val ty le 80/20 co phan tang
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts_train,
        labels_train,
        test_size=0.2,
        stratify=labels_train,
        random_state=42
    )
    print(f"[+] Da chia train/val (80/20): Train = {len(train_texts)} mau, Val = {len(val_texts)} mau")
    
    # 4. Tao DataLoader
    max_len = 40
    batch_size = 64
    
    train_dataset = IntentDataset(train_texts, train_labels, vocab, max_len=max_len)
    val_dataset = IntentDataset(val_texts, val_labels, vocab, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Khoi tao model TextCNN
    model = TextCNN(vocab_size=vocab_size, embed_dim=128, num_classes=5, kernel_sizes=[2, 3, 4, 5], num_filters=128, dropout_rate=0.3)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Tong so tham so TextCNN: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 6. Huan luyen 25 epochs
    epochs = 25
    best_val_acc = 0.0
    print("\n" + "=" * 60)
    print("BAT DAU HUAN LUYEN TEXTCNN (25 EPOCHS)")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y_b).sum().item()
            total_train += len(y_b)
            
        train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                total_val_loss += loss.item() * len(y_b)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y_b).sum().item()
                total_val += len(y_b)
                
        val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val
        
        is_best = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            is_best = " [* BEST - SAVED]"
            
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%{is_best}")
        
    print("\n" + "=" * 60)
    print(f"[+] HUAN LUYEN HOAN TAT! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"[+] Model tot nhat da duoc luu tai: {best_model_path}")
    print("=" * 60)
    
    # 7. Du doan 4.000 cau Intent Test voi model tot nhat
    print("\n[+] Dang load model tot nhat de du doan tap test (4.000 cau)...")
    best_model = TextCNN(vocab_size=vocab_size, embed_dim=128, num_classes=5, kernel_sizes=[2, 3, 4, 5], num_filters=128, dropout_rate=0.3)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    best_model.to(device)
    best_model.eval()
    
    test_texts = df_test["question"].astype(str).tolist()
    test_sample_ids = df_test["question_id"].tolist()
    
    test_dataset = IntentDataset(test_texts, vocab=vocab, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for x_b in test_loader:
            x_b = x_b.to(device)
            logits = best_model(x_b)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            
    predicted_labels = [ID2LABEL[idx] for idx in all_preds]
    
    df_pred = pd.DataFrame({
        "sample_id": test_sample_ids,
        "prediction": predicted_labels
    })
    
    df_pred.to_csv(pred_output_path, index=False, encoding="utf-8")
    print(f"[+] Da xuat ket qua du doan Intent vao: {pred_output_path}")
    print(f"[+] Tong so dong du doan: {len(df_pred)}")
    
    print("\n[+] Phan bo nhan du doan tren tap test:")
    print(df_pred["prediction"].value_counts())
    print("\n[+] 5 dong dau tien:")
    print(df_pred.head())

if __name__ == "__main__":
    main()
