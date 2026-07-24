# tfds cats_vs_dogs 이미지 분류 + PyTorch 전이학습 + 미세조정
# 목표: 결과보다 "전이학습과 미세조정 과정" 이해
# 백본: MobileNetV2

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 2. tfds에서 cats_vs_dogs 데이터 로드
# PyTorch에는 cats_vs_dogs가 기본 내장 데이터셋으로 제공되지 않기 때문에
# 여기서는 tfds를 "데이터 다운로드/분할 용도"로만 사용.
# 실제 모델 학습, 전처리, DataLoader, 학습 루프는 모두 PyTorch 방식입니다.
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    with_info=True,
    as_supervised=True
)

print(raw_train)
print(raw_validation)
print(raw_test)
print(metadata)

total = metadata.splits["train"].num_examples
print("train 전체 수:", total)
print("raw_train 수:", int(total * 0.8))
print("raw_val 수:", int(total * 0.1))
print("raw_test 수:", int(total * 0.1))

# 3. label 이름 확인
get_label_name = metadata.features["label"].int2str
print("label 0:", get_label_name(0))
print("label 1:", get_label_name(1))

# 4. 원본 샘플 확인
# tfds 원본 이미지는 shape이 제각각임. 예: (262, 350, 3), (500, 375, 3) 등
# 따라서 CNN에 넣기 전에 Resize가 필요.
for image, label in raw_train.take(1):
    print("원본 이미지 shape:", image.shape)
    print("라벨 번호:", label.numpy())
    print("라벨 이름:", get_label_name(label.numpy()))

    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label.numpy()))
    plt.axis("off")
    plt.show()

# 5. PyTorch Dataset 클래스 정의
# __len__: 데이터 개수 반환
# __getitem__: index에 해당하는 이미지와 라벨 반환
class CatsDogsTFDSDataset(Dataset):
    def __init__(self, tf_dataset, transform=None):
        """
        tf_dataset: tfds에서 읽은 데이터셋
        transform: PyTorch 이미지 전처리 함수. Resize, ToTensor, Normalize 등을 수행
        """

        # tfds Dataset을 numpy 형태로 변환 후 리스트로 저장
        # 각 샘플은 (image, label) 형태
        self.samples = list(tfds.as_numpy(tf_dataset))

        # 이미지 전처리 함수
        self.transform = transform

    def __len__(self):
        return len(self.samples)   # 전체 샘플 개수 반환

    def __getitem__(self, idx):
        # idx번째 이미지와 라벨 가져오기
        image, label = self.samples[idx]

        # image는 numpy 배열이며 shape은 (H, W, C)
        # PyTorch transform은 PIL Image를 주로 사용하므로 PIL로 변환
        image = Image.fromarray(image)

        # 전처리 적용
        if self.transform is not None:
            image = self.transform(image)

        # binary classification이므로 label을 float32로 변환
        # BCEWithLogitsLoss는 label이 float 형태여야 함
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


# 6. 이미지 전처리
# PyTorch torchvision의 pretrained MobileNetV2는 
# ImageNet 평균/표준편차 정규화를 사용하는 것이 일반적.
# ToTensor(): PIL Image / numpy image를 Tensor로 변환
#     [0, 255] -> [0, 1]
#     shape: (H, W, C) -> (C, H, W)
# Normalize(): ImageNet 기준 정규화

IMG_SIZE = 160

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 7. PyTorch Dataset 생성
train_dataset = CatsDogsTFDSDataset(raw_train, transform=train_transform)
validation_dataset = CatsDogsTFDSDataset(raw_validation, transform=eval_transform)
test_dataset = CatsDogsTFDSDataset(raw_test, transform=eval_transform)
print("PyTorch train dataset 수:", len(train_dataset))
print("PyTorch validation dataset 수:", len(validation_dataset))
print("PyTorch test dataset 수:", len(test_dataset))

# 8. 전처리 검증
# PyTorch 이미지 텐서 shape: TensorFlow: (H, W, C), PyTorch:    (C, H, W)
# 따라서 출력은 보통: torch.Size([3, 160, 160])
img, label = train_dataset[0]
print("전처리 샘플 dtype:", img.dtype)
print("전처리 샘플 shape:", img.shape)
print("전처리 샘플 min/max:", float(img.min()), float(img.max()))
print("라벨:", label)

# 9. DataLoader 생성
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# 10. batch 단위 데이터 확인
images_batch, labels_batch = next(iter(train_loader))
print("입력 배치 shape:", images_batch.shape)
print("라벨 배치 shape:", labels_batch.shape)

# 11. MobileNetV2 전이학습 모델 생성
# TensorFlow 코드: base_model = MobileNetV2(include_top=False, weights='imagenet')
# PyTorch 코드: model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
#
# torchvision MobileNetV2 구조: model.features -> 특징 추출기, model.classifier -> 분류기
# 기존 ImageNet 분류기는 1000개 클래스를 출력.
# cats_vs_dogs는 이진 분류이므로 마지막 Linear를 1개 출력으로 교체.
weights = MobileNet_V2_Weights.DEFAULT

model = mobilenet_v2(weights=weights)
# 기존 classifier 확인
print("기존 classifier:")
print(model.classifier)

# 마지막 분류기 교체
# MobileNetV2의 마지막 feature 차원은 1280
model.classifier[1] = nn.Linear(model.last_channel, 1)
print("수정된 classifier:")
print(model.classifier)

model = model.to(device)

# 12. 특징맵 shape 확인
# TensorFlow: feature_batch = base_model(images_batch)   shape: (32, 5, 5, 1280)
# PyTorch: feature_batch = model.features(images_batch)  shape: (32, 1280, 5, 5)
# PyTorch는 채널이 앞에 온다.
model.eval()

with torch.no_grad():
    sample_images = images_batch.to(device)
    feature_batch = model.features(sample_images)

print("입력 배치 shape:", sample_images.shape)
print("특징맵 배치 shape:", feature_batch.shape)

# GlobalAveragePooling2D와 같은 역할
gap = nn.AdaptiveAvgPool2d((1, 1)).to(device)

with torch.no_grad():
    gap_out = gap(feature_batch)
    gap_out_flatten = torch.flatten(gap_out, 1)

print("GAP 후 shape:", gap_out_flatten.shape)
# 의미: feature map: (32, 1280, 5, 5),  GAP 후: (32, 1280)
# 즉, 각 이미지가 1280차원의 특징 벡터로 요약된다.

# 13. 전이학습 1단계: base model 동결
# TensorFlow: base_model.trainable = False
# PyTorch:    requires_grad = False
# MobileNetV2의 features 부분을 동결하고, classifier만 학습.
for param in model.features.parameters():
    param.requires_grad = False

# classifier는 학습 가능해야 하므로 True
for param in model.classifier.parameters():
    param.requires_grad = True

# 학습 가능한 파라미터 확인
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print("전체 파라미터 수:", total_params)
print("학습 가능한 파라미터 수:", trainable_params)

# 14. 손실함수와 optimizer 정의
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# 15. 학습/평가 함수 정의
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # labels shape: (batch,)
        # logits shape: (batch, 1)
        # BCEWithLogitsLoss 계산을 위해 labels도 (batch, 1)로 변경
        labels = labels.view(-1, 1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # batch loss 누적
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        # sigmoid를 적용해서 0.5 이상이면 dog, 아니면 cat으로 판단
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        running_correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            labels = labels.view(-1, 1)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            running_correct += (preds == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc

# 16. fit 함수 정의
# Keras history.history와 비슷한 구조를 만들기 위해
# loss, accuracy, val_loss, val_accuracy를 dict에 저장.
def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    initial_epoch=0,
    scheduler=None,
    checkpoint_path=None,
    early_stopping_patience=None
):
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_acc = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(initial_epoch, epochs):
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

        # ReduceLROnPlateau는 validation loss를 기준으로 learning rate 조정
        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        # checkpoint 저장
        if checkpoint_path is not None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0

                torch.save(model.state_dict(), checkpoint_path)

                print(f"  -> best model 저장: {checkpoint_path}")
            else:
                patience_counter += 1

        # EarlyStopping
        if early_stopping_patience is not None:
            if patience_counter >= early_stopping_patience:
                print(f"EarlyStopping 발생: {early_stopping_patience} epoch 동안 val_accuracy 개선 없음")
                print(f"Best epoch: {best_epoch}, Best val_acc: {best_val_acc:.4f}")
                break

    # best checkpoint가 있으면 다시 로드
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("best model 가중치 복원 완료")

    return history


# 17. 전이학습 학습
EPOCHS_TRANSFER = 10

history = fit(
    model=model,
    train_loader=train_loader,
    val_loader=validation_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=EPOCHS_TRANSFER,
    initial_epoch=0
)

# 18. test 데이터 평가
test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion,
    device
)
print(f"전이학습 후 test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")

# 19. 전이학습 결과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["loss"], "b-", label="loss")
plt.plot(history["val_loss"], "r--", label="val_loss")
plt.xlabel("epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["accuracy"], "b-", label="accuracy")
plt.plot(history["val_accuracy"], "r--", label="val_accuracy")
plt.xlabel("epoch")
plt.legend()

plt.show()


# 20. Fine Tuning 준비
# TensorFlow 코드:
#     base_model.trainable = True
#     fine_tune_at = 100
#     for layer in base_model.layers[:fine_tune_at]:
#         layer.trainable = False
#
# PyTorch MobileNetV2는 TensorFlow처럼 layer가 150개 단위로 보이지 않고,
# model.features 안에 큰 블록 단위로 구성되어 있다.
#
# torchvision MobileNetV2: len(model.features)는 보통 19개.
# 여기서는 앞쪽 feature block은 동결하고,뒤쪽 feature block만 학습에 참여시킨다.

print("MobileNetV2 feature block 수:", len(model.features))

# 전체 features를 먼저 학습 가능하게 열어둠
for param in model.features.parameters():
    param.requires_grad = True

# 앞쪽 feature block은 다시 동결.  0~13번 block은 동결, 14번 이후 block만 미세조정
fine_tune_at = 14

for block in model.features[:fine_tune_at]:
    for param in block.parameters():
        param.requires_grad = False

# classifier는 계속 학습
for param in model.classifier.parameters():
    param.requires_grad = True


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print("Fine Tuning 전체 파라미터 수:", total_params)
print("Fine Tuning 학습 가능한 파라미터 수:", trainable_params)


# 21. Fine Tuning용 optimizer / scheduler / checkpoint 설정
# 미세조정에서는 이미 학습된 pretrained weight를 조금만 조정해야 하므로
# learning rate를 매우 작게 둔다.
optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-6
)

scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ft,
    mode="min",
    factor=0.5,
    patience=3
)

os.makedirs("checkpoints", exist_ok=True)
ckpt_path_ft = "checkpoints/finetune_best.pth"


# 22. Fine Tuning 학습
EPOCHS_FINETUNE = 10

history_ft = fit(
    model=model,
    train_loader=train_loader,
    val_loader=validation_loader,
    criterion=criterion,
    optimizer=optimizer_ft,
    device=device,
    epochs=EPOCHS_TRANSFER + EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_TRANSFER,
    scheduler=scheduler_ft,
    checkpoint_path=ckpt_path_ft,
    early_stopping_patience=5
)

# 23. Fine Tuning 후 test 평가
test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion,
    device
)
print(f"Fine Tuning 후 test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")


# 24. history 연결 함수 : Keras history.history와 비슷하게 만든 dict를 이어 붙임.
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

# 25. 전이학습 -> 미세조정 성능 시각화
plt.figure(figsize=(12, 5))

# accuracy 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, marker="o", label="train acc")
plt.plot(epochs, val_acc, marker="s", label="val acc")

for i, v in enumerate(acc):
    plt.text(
        epochs[i],
        v,
        f"{v * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

for i, v in enumerate(val_acc):
    plt.text(
        epochs[i],
        v,
        f"{v * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.axvline(
    split_epoch,
    linestyle="--",
    alpha=0.7,
    label="fine tuning start"
)

plt.title("Accuracy transfer learning -> fine tuning")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(loc="lower right")

# loss 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, marker="o", label="train loss")
plt.plot(epochs, val_loss, marker="s", label="val loss")

for i, v in enumerate(loss):
    plt.text(
        epochs[i],
        v,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

for i, v in enumerate(val_loss):
    plt.text(
        epochs[i],
        v,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.axvline(
    split_epoch,
    linestyle="--",
    alpha=0.7,
    label="fine tuning start"
)

plt.title("Loss transfer learning -> fine tuning")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()

# 26. 최종 모델 저장
torch.save(model.state_dict(), "cats_vs_dogs_mobilenetv2_finetuned.pth")
print("최종 모델 저장 완료: cats_vs_dogs_mobilenetv2_finetuned.pth")