# CNN : 개/고양이 이미지 고해상도 분류 - PyTorch 버전
# 목적: cats_vs_dogs 데이터셋을 이용하여 개/고양이 이미지를 이진 분류하는 CNN 모델 학습
#   1. tensorflow_datasets는 데이터 다운로드/저장용으로만 사용
#   2. torchvision.datasets.ImageFolder로 폴더 이미지 로딩
#   3. torchvision.transforms로 이미지 증강
#   4. DataLoader로 batch 단위 학습
#   5. nn.Module로 CNN 모델 정의
#   6. BCEWithLogitsLoss로 이진 분류 학습

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# TensorFlow Datasets 목록 확인
# 사용 가능한 tfds dataset 목록 확인. 너무 많이 출력되므로 필요할 때만 주석 해제
# print(tfds.list_builders())

# cats_vs_dogs 데이터 다운로드
# as_supervised=True: 데이터를 (image, label) 형태로 반환
# label: 0 -> cat, 1 -> dog
(dataset, info) = tfds.load(
    "cats_vs_dogs", with_info=True, as_supervised=True
)
print(dataset)
print(info)

label_names = info.features["label"].names
print("label_names:", label_names)  # ['cat', 'dog']
print("dataset keys:", dataset.keys())  # 보통 train만 지원

# 6. 원본 데이터 1장 확인
for image, label in dataset["train"].skip(0).take(1):
    print("image shape:", image.shape)
    print("image dtype:", image.dtype)
    print("label:", label.numpy())

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(label_names[label.numpy()])
    plt.axis("off")
    plt.show()

# 7. 폴더 구조 생성
# PyTorch ImageFolder는 아래와 같은 폴더 구조를 사용한다.
# cats_and_dogs_filtered/
# ├── train/
# │   ├── cats/
# │   └── dogs/
# └── validation/
#     ├── cats/
#     └── dogs/
# ImageFolder는 하위 폴더명을 class 이름으로 인식한다.

base_dir = "./cats_and_dogs_filtered"

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")

val_cats_dir = os.path.join(validation_dir, "cats")
val_dogs_dir = os.path.join(validation_dir, "dogs")

for d in [train_cats_dir, train_dogs_dir, val_cats_dir, val_dogs_dir]:
    os.makedirs(d, exist_ok=True)

# 8. 이미 저장된 데이터가 있는지 확인
# 매번 tfds 데이터를 이미지 파일로 저장하면 시간이 오래 걸릴 수 있다.
# 이미 폴더에 이미지가 있으면 저장 과정을 건너뛰도록 처리한다.
def count_images_in_folder(folder):
    if not os.path.exists(folder):
        return 0

    valid_exts = (".jpg", ".jpeg", ".png")
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(valid_exts)
    ])


existing_count = (
    count_images_in_folder(train_cats_dir)
    + count_images_in_folder(train_dogs_dir)
    + count_images_in_folder(val_cats_dir)
    + count_images_in_folder(val_dogs_dir)
)
print("현재 저장된 이미지 수:", existing_count)

# 9. TensorFlow Dataset 이미지를 JPG 파일로 저장하는 함수
# PyTorch ImageFolder로 로딩하기 위해 이미지를 폴더에 저장한다.
# label: 0 -> cats 폴더, 1 -> dogs 폴더
IMG_SIZE = (150, 150)

def save_image_func(img, label, idx, split):
    # TensorFlow Tensor 이미지를 150x150으로 resize한 뒤 cats 또는 dogs 폴더에 jpg 파일로 저장하는 함수
    img = tf.image.resize(img, IMG_SIZE)  # 이미지 크기 조정
    img = tf.cast(img, tf.uint8).numpy()  # Tensor를 uint8 numpy 이미지로 변환

    # split과 label에 따라 저장 폴더 결정
    if split == "train":
        folder = train_cats_dir if int(label) == 0 else train_dogs_dir
    else:
        folder = val_cats_dir if int(label) == 0 else val_dogs_dir

    path = os.path.join(folder, f"{idx}.jpg") 
    tf.keras.preprocessing.image.save_img(path, img)  # 이미지 저장


# 10. train / validation 분리 후 이미지 저장
# cats_vs_dogs는 train split만 제공되므로, 전체 데이터의 80%는 train, 20%는 validation으로 사용한다.
total = info.splits["train"].num_examples
train_size = int(0.8 * total)
print("전체 이미지 수:", total)
print("train 이미지 수:", train_size)
print("validation 이미지 수:", total - train_size)

# 이미 저장된 이미지가 없을 때만 저장
if existing_count == 0:
    print("이미지를 폴더에 저장합니다...")
    for i, (img, label) in enumerate(dataset["train"]):
        if i < train_size:
            save_image_func(img, label, i, "train")
        else:
            save_image_func(img, label, i, "val")
    print("데이터 준비 완료")
else:
    print("이미 저장된 이미지가 있어 저장 과정을 건너뜁니다.")

# 11. 폴더 존재 여부 및 이미지 개수 확인
for p in [
    train_dir, train_cats_dir, train_dogs_dir, validation_dir,
    val_cats_dir, val_dogs_dir ]:
    print(p, "->", os.path.exists(p))

print(
    "cats(train):", len(os.listdir(train_cats_dir)),
    "| dogs(train):", len(os.listdir(train_dogs_dir))
)
print(
    "cats(val):", len(os.listdir(val_cats_dir)),
    "| dogs(val):", len(os.listdir(val_dogs_dir))
)

# 12. Colab에서 폴더 전체를 zip으로 압축하여 다운로드하기
# 필요할 때만 주석 해제해서 사용
"""
from google.colab import files
shutil.make_archive(
    "cats_and_dogs_filtered", "zip", "cats_and_dogs_filtered"
)
files.download("cats_and_dogs_filtered.zip")
"""

# 13. PyTorch transforms 정의
# Keras ImageDataGenerator 역할을 PyTorch에서는 transforms가 담당한다.
# train_transform: 학습 이미지에 적용할 증강
# val_transform:   검증 이미지에 적용할 기본 변환
# 주의: 검증 데이터에는 랜덤 증강을 적용하지 않는다. 검증 데이터는 Resize, ToTensor 정도만 적용한다.

IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 128
EPOCHS = 20

train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(degrees=15), # -15도 ~ +15도 랜덤 회전
    transforms.RandomAffine(   # 가로/세로 방향 최대 10% 평행 이동
        degrees=0, translate=(0.1, 0.1)
    ),
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전

    # PIL Image -> Tensor    픽셀값 0~255를 0~1로 변환   shape: (H, W, C) -> (C, H, W)
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

# 14. ImageFolder로 데이터셋 생성
# ImageFolder는 폴더명을 기준으로 class label을 자동 생성한다.
# 예:
# train/
# ├── cats
# └── dogs
# class_to_idx: {'cats': 0, 'dogs': 1}
train_dataset = datasets.ImageFolder(
    root=train_dir, transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=validation_dir, transform=val_transform
)
print("class_to_idx:", train_dataset.class_to_idx)

idx_to_name = {
    v: k for k, v in train_dataset.class_to_idx.items()
}
print("idx_to_name:", idx_to_name)

# 15. DataLoader 생성
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# 16. 학습 데이터 미리보기
# DataLoader에서 한 batch를 꺼내 증강된 이미지를 확인한다.
imgs, labels = next(iter(train_loader))

n_show = min(12, imgs.shape[0])
cols = 6
rows = int(np.ceil(n_show / cols))

plt.figure(figsize=(10, 2 * rows))

for i in range(n_show):
    ax = plt.subplot(rows, cols, i + 1)
    img_np = imgs[i].permute(1, 2, 0).numpy()
    ax.imshow(img_np)
    ax.set_title(idx_to_name[int(labels[i])])
    ax.axis("off")

plt.suptitle("Sample preview train", fontsize=10)
plt.tight_layout()
plt.show()

# 17. CNN 모델 정의
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()

        self.features = nn.Sequential(
            # 입력: (batch, 3, 150, 150)
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 출력: (batch, 16, 150, 150)

            # 출력: (batch, 16, 75, 75)
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 출력: (batch, 32, 75, 75)

            # 출력: (batch, 32, 37, 37)
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 출력: (batch, 64, 37, 37)
            # 출력: (batch, 64, 18, 18)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 18 * 18, 512),   # 64 * 18 * 18 = 20736
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)   # 이진 분류이므로 출력 1개
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.squeeze(1)   # shape: (batch, 1) -> (batch,)
        return x


model = CatDogCNN().to(device)
print(model)

# 18. 손실 함수와 Optimizer 설정. 이진 분류에서는 BCEWithLogitsLoss 사용
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 19. 학습 함수 정의
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)

        labels = labels.float().to(device)  # BCEWithLogitsLoss는 float label을 사용

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)   # logit -> probability

        # 개/고양이 예측 : ImageFolder 기준: cats -> 0,  dogs -> 1
        preds = (probs >= 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


# 20. 평가 함수 정의
def evaluate(model, data_loader, criterion, device):
    """
    검증 데이터에 대해 모델을 평가하는 함수
    평가 시: - model.eval()  - torch.no_grad()
    를 사용해서 Dropout을 비활성화하고 gradient 계산을 하지 않는다.
    """

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


# 21. Checkpoint / EarlyStopping 설정
os.makedirs("chkpoints", exist_ok=True)
best_model_path = "chkpoints/catdog_best.pth"

best_val_acc = 0.0
patience = 5
patience_counter = 0

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

# 22. 모델 학습 실행
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(
        model, val_loader, criterion, device
    )

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    # val_accuracy 기준 최고 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save( model.state_dict(), best_model_path )
        print(f"  -> best model 저장: {best_model_path}")
    else:
        patience_counter += 1
        print(f"  -> val_acc 개선 없음: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("EarlyStopping 실행")
            break

# 23. best model 불러오기
model.load_state_dict(
    torch.load(best_model_path, map_location=device)
)
model.to(device)

# 24. 검증 데이터 평가
val_loss, val_acc = evaluate(
    model, val_loader, criterion, device
)
print(f"검증 평가 결과 : acc : {val_acc:.4f}, loss : {val_loss:.4f}")

# 25. 학습 과정 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc_history, label="train acc")
plt.plot(val_acc_history, label="val acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label="train loss")
plt.plot(val_loss_history, label="val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)

plt.show()

# 26. 검증 이미지 예측 미리보기용 DataLoader 생성
# 평가용 val_loader는 shuffle=False로 유지했다.
# 예측 미리보기는 cats/dogs를 섞어서 확인하기 위해 shuffle=True인 별도 DataLoader를 만든다.
preview_loader = DataLoader(
    val_dataset, batch_size=24, shuffle=True, num_workers=2
)

# 27. cats / dogs 각각 6개씩 수집
n_each = 6

cats_images = []
dogs_images = []
cats_labels = []
dogs_labels = []

for images, labels in preview_loader:
    for im, lb in zip(images, labels):
        if int(lb) == 0 and len(cats_images) < n_each:
            cats_images.append(im)
            cats_labels.append(lb)
        elif int(lb) == 1 and len(dogs_images) < n_each:
            dogs_images.append(im)
            dogs_labels.append(lb)

        if len(cats_images) >= n_each and len(dogs_images) >= n_each:
            break

    if len(cats_images) >= n_each and len(dogs_images) >= n_each:
        break

# 28. 수집한 이미지 예측
model.eval()

def predict_images(model, image_list, device):
    """
    이미지 tensor 리스트를 받아 dog 확률을 반환하는 함수
    반환값: probs - dog일 확률
    """
    batch = torch.stack(image_list).to(device)

    with torch.no_grad():
        outputs = model(batch)
        probs = torch.sigmoid(outputs).cpu().numpy()

    return probs

cats_probs = predict_images(model, cats_images, device)
dogs_probs = predict_images(model, dogs_images, device)

# 29. 예측 결과 시각화
# 0행: 실제 cats 이미지,  1행: 실제 dogs 이미지
# p_dog: 모델이 dog라고 판단한 확률
rows, cols = 2, n_each
plt.figure(figsize=(2.5 * cols, 5))

for i in range(n_each):
    # cats row
    ax = plt.subplot(rows, cols, i + 1)
    img_np = cats_images[i].permute(1, 2, 0).numpy()
    ax.imshow(img_np)
    ax.axis("off")

    p = cats_probs[i]
    pred_name = "dogs" if p >= 0.5 else "cats"
    ax.set_title(
        f"True:cats | Pred:{pred_name}\n(p_dog={p:.2f})", fontsize=9
    )

    # dogs row
    ax = plt.subplot(rows, cols, cols + i + 1)
    img_np = dogs_images[i].permute(1, 2, 0).numpy()

    ax.imshow(img_np)
    ax.axis("off")

    p = dogs_probs[i]
    pred_name = "dogs" if p >= 0.5 else "cats"

    ax.set_title(
        f"True:dogs | Pred:{pred_name}\n(p_dog={p:.2f})",
        fontsize=9
    )

plt.suptitle("validation preview mixed cats & dogs", fontsize=12)
plt.tight_layout()
plt.show()