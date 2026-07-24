# 쓰레기 재활용 분류기 모델 : MobileNetV2 전이학습 + 미세조정
# 데이터셋 폴더 구조 예:
# garbage_classification/
# ├── cardboard/
# ├── glass/
# ├── metal/
# ├── paper/
# ├── plastic/
# └── trash/
#
# 목표:
#   1. Google Drive에 저장된 이미지 폴더 읽기
#   2. train / validation / test = 8 : 1 : 1 분할
#   3. MobileNetV2 전이학습
#   4. 일부 backbone layer 미세조정
#   5. confusion matrix, classification report 출력
#   6. 새로운 이미지 1장 예측

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import confusion_matrix, classification_report

# 1. Google Drive 사용 시 
# Colab에서 실행한다면 아래 주석을 해제.
# from google.colab import drive
# drive.mount("/content/drive")

# 2. 기본 설정
DATASET_PATH = "/content/drive/MyDrive/garbage_classification"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# 4. 이미지 전처리 정의
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # 데이터 증강
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(
        size=IMG_SIZE, scale=(0.9, 1.0)
    ),

    transforms.ColorJitter(contrast=0.1),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ImageFolder로 전체 데이터셋 읽기 : ImageFolder는 폴더명을 class label로 자동 인식한다.
# 예:
#   garbage_classification/cardboard/*.jpg -> label 0
#   garbage_classification/glass/*.jpg     -> label 1
full_dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=eval_transform
)

class_names = full_dataset.classes
num_classes = len(class_names)
print("class_names:", class_names)
print("전체 이미지 수:", len(full_dataset))
print("클래스 개수:", num_classes)

# 6. train / validation / test 분할
total_size = len(full_dataset)
train_size = int(total_size * 0.8)
remaining_size = total_size - train_size

val_size = remaining_size // 2
test_size = remaining_size - val_size

generator = torch.Generator().manual_seed(SEED)

train_subset, val_subset, test_subset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=generator
)
print("train 개수:", len(train_subset))
print("val 개수:", len(val_subset))
print("test 개수:", len(test_subset))

# 7. train dataset에만 데이터 증강 적용
# random_split으로 나눈 subset은 내부적으로 같은 full_dataset을 참조한다.
# 따라서 train에만 augmentation을 적용하려면 Dataset을 하나 더 만들어
# 같은 index를 사용하도록 구성한다.
train_dataset_aug = datasets.ImageFolder(
    root=DATASET_PATH, transform=train_transform
)
val_dataset_eval = datasets.ImageFolder(
    root=DATASET_PATH, transform=eval_transform
)
test_dataset_eval = datasets.ImageFolder(
    root=DATASET_PATH, transform=eval_transform
)

# random_split으로 얻은 index를 재사용
train_subset.dataset = train_dataset_aug
val_subset.dataset = val_dataset_eval
test_subset.dataset = test_dataset_eval


# 8. DataLoader 생성
train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# 9. batch shape 확인
images_batch, labels_batch = next(iter(train_loader))
print("images batch shape:", images_batch.shape)
print("labels batch shape:", labels_batch.shape)
# 예:
# images batch shape: torch.Size([32, 3, 224, 224])
# labels batch shape: torch.Size([32])

# 10. MobileNetV2 모델 구성
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
print("기존 classifier:")
print(model.classifier)

# MobileNetV2의 classifier 교체 : model.last_channel은 보통 1280이다.
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, num_classes)
)
print("수정된 classifier:", model.classifier)

model = model.to(device)

# 11. 전이학습 1단계: backbone 동결
for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("전체 파라미터 수:", total_params)
print("학습 가능한 파라미터 수:", trainable_params)

# 12. 손실함수와 optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.3, patience=2
)

# 13. 학습 함수
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 이전 batch의 gradient 초기화
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
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


# 14. EarlyStopping 포함 fit 함수
def fit( model,train_loader, val_loader,criterion,
    optimizer, scheduler, device, epochs=100, patience=5, checkpoint_path="best_model.pth"
):
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
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

        # ReduceLROnPlateau는 validation loss를 기준으로 learning rate 조절
        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:03}/{epochs}] "
            f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
            f"lr: {current_lr:.8f}"
        )

        # validation loss가 좋아지면 best weight 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
            print(f"  -> best model 저장: {checkpoint_path}")
        else:
            patience_counter += 1

        # EarlyStopping
        if patience_counter >= patience:
            print(f"EarlyStopping 발생: {patience} epoch 동안 val_loss 개선 없음")
            break

    # 가장 좋았던 weight 복원
    model.load_state_dict(best_model_wts)
    return history


print("baseline 전이학습 시작...")

history_baseline = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=100,
    patience=5,
    checkpoint_path="baseline_best.pth"
)

baseline_loss, baseline_acc = evaluate(
    model,
    test_loader,
    criterion,
    device
)
print(f"baseline : loss:{baseline_loss:.4f}, acc:{baseline_acc:.4f}")


# 16. Fine Tuning 준비
# PyTorch torchvision MobileNetV2는 features가 block 단위로 구성되어 있다.
# len(model.features)는 보통 19개다.
# 여기서는 마지막 5개 feature block만 학습에 참여시킨다.
# 즉, 앞쪽 feature block은 일반적인 특징 추출기로 유지하고,
# 뒤쪽 고수준 특징 layer만 데이터셋에 맞게 미세조정한다.
print("Fine Tuning 준비...")

# 먼저 전체 features를 학습 가능하게 설정
for param in model.features.parameters():
    param.requires_grad = True

# 앞쪽 block은 다시 동결 : 마지막 5개 block만 미세조정
fine_tune_blocks = 5
freeze_until = len(model.features) - fine_tune_blocks

for block in model.features[:freeze_until]:
    for param in block.parameters():
        param.requires_grad = False

# classifier는 계속 학습 가능
for param in model.classifier.parameters():
    param.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Fine Tuning 전체 파라미터 수:", total_params)
print("Fine Tuning 학습 가능한 파라미터 수:", trainable_params)
print("MobileNetV2 feature block 수:", len(model.features))
print("동결된 block 범위: 0 ~", freeze_until - 1)
print("학습되는 block 범위:", freeze_until, "~", len(model.features) - 1)


# 17. Fine Tuning compile에 해당하는 부분
# TensorFlow: optimizer=Adam(learning_rate=1e-5)
# PyTorch: 새 optimizer를 만들고, requires_grad=True인 파라미터만 전달.
optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
)

scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ft, mode="min", factor=0.3, patience=2
)

# 18. Fine Tuning 학습
print("fine tuning 시작...")
history_finetune = fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer_ft,
    scheduler=scheduler_ft,
    device=device,
    epochs=100,
    patience=5,
    checkpoint_path="finetune_best.pth"
)

finetune_loss, finetune_acc = evaluate(
    model,
    test_loader,
    criterion,
    device
)
print(f"finetune : loss:{finetune_loss:.4f}, acc:{finetune_acc:.4f}")


# 19. PyTorch 모델 저장
save_path = "garbage_classify.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes
    },
    save_path
)
print("모델 저장 완료:", save_path)

# 20. 학습 과정 시각화
acc = history_baseline["accuracy"] + history_finetune["accuracy"]
val_acc = history_baseline["val_accuracy"] + history_finetune["val_accuracy"]
loss = history_baseline["loss"] + history_finetune["loss"]
val_loss = history_baseline["val_loss"] + history_finetune["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

# accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="train acc")
plt.plot(epochs_range, val_acc, label="val acc")
plt.legend(loc="lower right")
plt.title("Train & Validation Accuracy")

# loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="train loss")
plt.plot(epochs_range, val_loss, label="val loss")
plt.legend(loc="upper right")
plt.title("Train & Validation Loss")

plt.show()


# 21. test 데이터 예측
# TensorFlow:
#   predictions = model.predict(images)
#   np.argmax(predictions, axis=1)
# PyTorch:
#   logits = model(images)
#   preds = torch.argmax(logits, dim=1)
y_true = []
y_pred = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())


# 22. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)

plt.xlabel("Predict")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# 23. Classification Report
print(classification_report(y_true, y_pred, target_names=class_names))

# 24. 저장된 PyTorch 모델 불러오기 함수
# PyTorch는 저장된 state_dict를 불러오려면 먼저 동일한 모델 구조를 다시 만들어야 한다.
def create_garbage_model(num_classes):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes)
    )

    return model


def load_garbage_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = create_garbage_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names

# 25. 새로운 이미지 1장 예측 함수
#   PIL.Image.open()
#   transform 적용
#   unsqueeze(0)로 batch 차원 추가
#   model(image)
def predict_garbage_func(img_path, model, class_names, device):
    img = Image.open(img_path).convert("RGB")

    # 평가용 전처리 적용
    img_tensor = eval_transform(img)

    # batch 차원 추가
    # shape: (3, 224, 224) -> (1, 3, 224, 224)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(img_tensor)

        # 다중분류 확률값으로 변환
        probs = torch.softmax(logits, dim=1)

        # 가장 확률이 높은 클래스
        pred_index = torch.argmax(probs, dim=1).item()

        pred_class = class_names[pred_index]
        confidence = probs[0, pred_index].item()

    print("예측 결과:", pred_class)
    print("신뢰도:", round(confidence * 100, 2), "%")

    return pred_class, confidence


# 26. 저장된 모델을 다시 불러와서 새로운 이미지 예측
loaded_model, loaded_class_names = load_garbage_model(
    "garbage_classify.pth", device
)

predict_garbage_func(
    "myimage.jpeg", loaded_model, loaded_class_names, device
)