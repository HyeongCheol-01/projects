# tf_flowers dataset 5종 꽃 이미지 분류 : 전이학습 + 미세조정 예제
# 데이터셋: tf_flowers
# 클래스: ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
# 백본: MobileNetV2
# 목표:
#   1. tfds로 데이터셋 로드
#   2. PyTorch Dataset/DataLoader로 변환
#   3. MobileNetV2 전이학습
#   4. 일부 layer 미세조정
#   5. 검증 이미지 예측 및 시각화

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 2. tf_flowers 데이터셋 로드
# tfds는 데이터 다운로드와 split 용도로만 사용.
# 실제 학습은 PyTorch Dataset/DataLoader로 진행.
(train_tfds, val_tfds), ds_info = tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:]"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# 샘플 타입 확인
for image, label in train_tfds.take(1):
    print("원본 image 타입:", type(image))
    print("원본 label 타입:", type(label))
    print("원본 image shape:", image.shape)
    print("원본 label:", label.numpy())

# 3. 클래스 정보 확인
class_names = ds_info.features["label"].names
num_classes = ds_info.features["label"].num_classes
print("클래스 이름:", class_names)
print("클래스 개수:", num_classes)

print("train 개수:", ds_info.splits["train"].num_examples)
print("실제 train split 80%:", int(ds_info.splits["train"].num_examples * 0.8))
print("실제 val split 20%:", int(ds_info.splits["train"].num_examples * 0.2))

# 4. PyTorch Dataset 클래스 정의
class FlowersTFDSDataset(Dataset):
    def __init__(self, tf_dataset, transform=None):
        """
        tf_dataset: tfds.load()로 가져온 TensorFlow Dataset
        transform: torchvision.transforms로 정의한 PyTorch 전처리
        """

        # tf.data.Dataset을 numpy 형태로 변환해서 리스트로 저장
        # 데이터 수가 크지 않은 tf_flowers에서는 이렇게 처리해도 무난하다.
        self.samples = list(tfds.as_numpy(tf_dataset))

        # 이미지 전처리 함수
        self.transform = transform

    def __len__(self):
        # 전체 샘플 수 반환
        return len(self.samples)

    def __getitem__(self, idx):
        # idx번째 이미지와 라벨 가져오기
        image, label = self.samples[idx]

        # image는 numpy 배열 형태이며 shape은 (H, W, C)
        # torchvision transform은 PIL Image를 주로 사용하므로 PIL로 변환
        image = Image.fromarray(image)

        # 이미지 전처리 적용
        if self.transform is not None:
            image = self.transform(image)

        # CrossEntropyLoss는 label이 정수형 LongTensor여야 합니다.
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# 5. 이미지 전처리 정의
# PyTorch pretrained MobileNetV2에서는 ImageNet 정규화를 사용하는 것이 일반적.
# transforms.ToTensor(): [0, 255] 범위의 PIL 이미지를 [0, 1] 범위 Tensor로 변환
#   shape도 (H, W, C)에서 (C, H, W)로 바뀜
# Normalize(): ImageNet 평균/표준편차 기준으로 정규화
IMG_SIZE = 160
BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 6. PyTorch Dataset 생성
train_dataset = FlowersTFDSDataset(train_tfds, transform=train_transform)
val_dataset = FlowersTFDSDataset(val_tfds, transform=val_transform)
print("PyTorch train dataset 개수:", len(train_dataset))
print("PyTorch val dataset 개수:", len(val_dataset))


# 7. 전처리 결과 확인
sample_img, sample_label = train_dataset[0]
print("전처리 후 이미지 dtype:", sample_img.dtype)
print("전처리 후 이미지 shape:", sample_img.shape)
print("전처리 후 이미지 min/max:", float(sample_img.min()), float(sample_img.max()))
print("전처리 후 label:", sample_label.item(), class_names[sample_label.item()])


# 8. DataLoader 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# batch 확인
images_batch, labels_batch = next(iter(train_loader))
print("입력 배치 shape:", images_batch.shape)
print("라벨 배치 shape:", labels_batch.shape)

# 출력 예:
# 입력 배치 shape: torch.Size([32, 3, 160, 160])
# 라벨 배치 shape: torch.Size([32])

# 9. MobileNetV2 전이학습 모델 생성
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
print("기존 classifier:")
print(model.classifier)

# MobileNetV2의 마지막 classifier 교체
# model.last_channel은 보통 1280입니다.
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
print("수정된 classifier:")
print(model.classifier)

model = model.to(device)

# 10. 특징맵 shape 확인
# PyTorch MobileNetV2에서 특징 추출기는 model.features다.
# 입력: (batch, 3, 160, 160)
# 특징맵: (batch, 1280, 5, 5)
model.eval()

with torch.no_grad():
    sample_batch = images_batch.to(device)
    feature_batch = model.features(sample_batch)

print("입력 배치 shape:", sample_batch.shape)
print("특징맵 배치 shape:", feature_batch.shape)

# 11. 전이학습 1단계: base model 동결
# 특징 추출기 model.features는 동결하고, 새로 교체한 classifier만 학습한다.
for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

# 학습 가능한 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("전체 파라미터 수:", total_params)
print("학습 가능한 파라미터 수:", trainable_params)


# 12. 손실함수와 optimizer 정의
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# 13. 학습 함수 정의
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    # 학습 모드
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in dataloader:
        # 데이터를 GPU 또는 CPU로 이동
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # logits shape: (batch, num_classes)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 통계 계산
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        # 가장 큰 logit을 가진 클래스가 예측 클래스
        preds = torch.argmax(logits, dim=1)

        running_correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    # 평가 모드
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    # 평가에서는 gradient 계산 불필요
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            preds = torch.argmax(logits, dim=1)

            running_correct += (preds == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc


# 14. fit 함수 정의
def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    initial_epoch=0,
    scheduler=None
):
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(initial_epoch, initial_epoch + epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # scheduler가 있으면 val_loss 기준으로 learning rate 조정
        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}] "
            f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

    return history


# 15. 전이학습 실행
# 특징 추출기는 동결되어 있고, classifier 부분만 학습된다.
print("전이학습 시작...")

EPOCHS_TRANSFER = 5

history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=EPOCHS_TRANSFER
)


# 16. 전이학습 후 검증 평가
loss, acc = evaluate(model, val_loader, criterion, device)
print(f"전이학습 후 validation loss: {loss:.4f}, acc: {acc:.4f}")


# 17. Fine Tuning 준비
print("MobileNetV2 feature block 수:", len(model.features))

# 먼저 features 전체를 학습 가능하게 열어둠
for param in model.features.parameters():
    param.requires_grad = True

# 앞쪽 block은 다시 동결 : 0~13번 block은 동결, 14번 이후 block만 fine tuning
fine_tune_at = 14

for block in model.features[:fine_tune_at]:
    for param in block.parameters():
        param.requires_grad = False

# classifier는 계속 학습 가능
for param in model.classifier.parameters():
    param.requires_grad = True


# 학습 가능한 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Fine Tuning 전체 파라미터 수:", total_params)
print("Fine Tuning 학습 가능한 파라미터 수:", trainable_params)


# 18. Fine Tuning용 optimizer 설정
optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-6
)

scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ft,
    mode="min",
    factor=0.5,
    patience=2
)

# 19. Fine Tuning 실행
print("미세 조정 시작...")
EPOCHS_FINETUNE = 5

history_ft = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer_ft,
    device=device,
    epochs=EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_TRANSFER,
    scheduler=scheduler_ft
)

# 20. Fine Tuning 후 검증 평가
loss, acc = evaluate(model, val_loader, criterion, device)
print(f"Fine Tuning 후 validation loss: {loss:.4f}, acc: {acc:.4f}")

# 21. 검증 데이터에서 이미지 batch 1개 추출 후 예측
# PyTorch 예측 과정:
#   1. model.eval()
#   2. torch.no_grad()
#   3. logits = model(images)
#   4. probs = softmax(logits)
#   5. pred_classes = argmax(probs)

model.eval()

sample_images, sample_labels = next(iter(val_loader))
sample_images_device = sample_images.to(device)

with torch.no_grad():
    logits = model(sample_images_device)
    pred_probs = torch.softmax(logits, dim=1)
    pred_classes = torch.argmax(pred_probs, dim=1)

print("예측 확률:", pred_probs[:5])
print("예측 클래스:", pred_classes[:5])
print("클래스 이름:", class_names)

# 22. 예측값과 실제값 출력
pred_classes_cpu = pred_classes.cpu()
sample_labels_cpu = sample_labels.cpu()

for i in range(len(sample_images)):
    predict_index = int(pred_classes_cpu[i])
    actual_index = int(sample_labels_cpu[i])

    predict_name = class_names[predict_index]
    actual_name = class_names[actual_index]

    print(
        f"[{i:02}] "
        f"pred: {predict_index} ({predict_name}) | "
        f"actual: {actual_index} ({actual_name})"
    )


# 23. 시각화를 위한 unnormalize 함수
# 학습용 이미지는 ImageNet 기준으로 Normalize 되어 있다.
# matplotlib으로 보기 위해서는 다시 원래 색상 범위에 가깝게 복원해야 한다.
def unnormalize(img_tensor):
    """
    Normalize된 PyTorch Tensor 이미지를 시각화 가능한 형태로 복원합니다.
    입력 shape: (3, H, W),   출력 shape: (H, W, 3)
    """

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)

    # PyTorch: (C, H, W)
    # matplotlib: (H, W, C)
    img = img.permute(1, 2, 0).numpy()

    return img

# 24. 예측 결과 시각화
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(1, 10, i + 1)

    img = unnormalize(sample_images[i])
    plt.imshow(img)

    predicted_label = class_names[int(pred_classes_cpu[i])]
    actual_label = class_names[int(sample_labels_cpu[i])]

    color = "black" if predicted_label == actual_label else "red"

    plt.title(
        f"pred:{predicted_label}\nactual:{actual_label}",
        color=color,
        fontsize=10
    )

    plt.axis("off")

plt.tight_layout()
plt.show()

# 25. 학습 결과 시각화
def concat_hist(h1, h2):
    out = {}

    for key in h1.keys():
        out[key] = h1[key] + h2[key]

    return out


hist_all = concat_hist(history, history_ft)

acc = hist_all["accuracy"]
val_acc = hist_all["val_accuracy"]
loss = hist_all["loss"]
val_loss = hist_all["val_loss"]

epochs = range(1, len(acc) + 1)
split_epoch = EPOCHS_TRANSFER

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, marker="o", label="train acc")
plt.plot(epochs, val_acc, marker="s", label="val acc")
plt.axvline(split_epoch, linestyle="--", alpha=0.7, label="fine tuning start")
plt.title("Accuracy transfer learning -> fine tuning")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(loc="lower right")

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, marker="o", label="train loss")
plt.plot(epochs, val_loss, marker="s", label="val loss")
plt.axvline(split_epoch, linestyle="--", alpha=0.7, label="fine tuning start")
plt.title("Loss transfer learning -> fine tuning")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()

# 26. 모델 저장
torch.save(model.state_dict(), "tf_flowers_mobilenetv2_finetuned.pth")

print("모델 저장 완료: tf_flowers_mobilenetv2_finetuned.pth")