# MobileNetV2 전이학습 + Fine Tuning 코드
# 구성은 다음 순서이다.
# 1. CIFAR-10 데이터 로드
# 2. MobileNetV2 사전 학습 모델 불러오기
# 3. 백본 전체 동결
# 4. 새 분류기만 학습
# 5. 테스트 평가
# 6. 마지막 일부 층만 동결 해제
# 7. Fine Tuning
# 8. 다시 테스트 평가

# PyTorch에서는 torchvision.models.mobilenet_v2로 MobileNetV2를 불러오고, MobileNet_V2_Weights.DEFAULT로 ImageNet 사전 학습 가중치를 사용할 수 있다.

# MobileNetV2 전이학습으로 CIFAR-10 분류하기 - PyTorch 버전
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt

# GPU가 있으면 GPU 사용, 없으면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 재현성을 위한 시드 고정
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. CIFAR-10 데이터셋 불러오기
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
    )
])

train_full_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)
print("전체 학습 데이터 수:", len(train_full_dataset))
print("테스트 데이터 수:", len(test_dataset))

# 2. 학습 데이터 / 검증 데이터 분리
# 전체 50,000개 중 80%는 학습, 20%는 검증
train_size = int(len(train_full_dataset) * 0.8)
val_size = len(train_full_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
print("학습 데이터 수:", len(train_dataset))
print("검증 데이터 수:", len(val_dataset))

# 3. DataLoader 생성
batch_size = 64

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# 4. CIFAR-10 클래스 이름
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 5. 데이터 확인 시각화
# Normalize가 적용된 이미지는 바로 imshow 하면 색이 이상하게 보일 수 있음
# 따라서 시각화할 때는 mean/std를 이용해 역정규화한 뒤 출력함
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img * std + mean

images, labels = next(iter(train_loader))

plt.figure(figsize=(6, 6))

for i in range(16):
    plt.subplot(4, 4, i + 1)

    img = denormalize(images[i]).clamp(0, 1)
    img = img.permute(1, 2, 0)

    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()
print("라벨:", labels[:16])

# 6. MobileNetV2 사전 학습 모델 불러오기
weights = MobileNet_V2_Weights.DEFAULT
model_tl = mobilenet_v2(weights=weights)

# 7. 기존 분류기 구조 확인
print(model_tl)
# torchvision MobileNetV2의 classifier는 보통 다음 구조임
# classifier = Sequential(
#     Dropout(...)
#     Linear(in_features=1280, out_features=1000)
# )
# ImageNet 사전 학습 모델의 마지막 출력은 1000개 클래스임
# CIFAR-10은 10개 클래스이므로 마지막 Linear 계층을 새로 교체해야 함

# 8. 백본 동결 Freeze
# PyTorch에서는 각 파라미터의 requires_grad를 False로 설정하면
# 해당 파라미터는 gradient가 계산되지 않고 학습에 참여하지 않음
for param in model_tl.features.parameters():
    param.requires_grad = False

# 9. CIFAR-10용 새 분류기 정의
# 기존 ImageNet 분류기: Linear(1280 -> 1000)
# 새 CIFAR-10 분류기: Linear(1280 -> 10)
# classifier[1]이 마지막 Linear 계층
in_features = model_tl.classifier[1].in_features

model_tl.classifier[1] = nn.Linear(
    in_features=in_features, out_features=10
)

model_tl = model_tl.to(device)

# 10. 파라미터 수 확인 함수
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"전체 파라미터 수: {total:,}")
    print(f"학습 가능 파라미터 수: {trainable:,}")
    print(f"동결된 파라미터 수: {non_trainable:,}")

print("\n[전이학습 단계 파라미터 정보]")
count_params(model_tl)

# 11. 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 12. 학습 함수 정의
def train_model(model, train_loader, val_loader, optimizer, epochs=10):
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        # 학습 모드
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # 이전 mini-batch에서 계산된 gradient 초기화
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs, dim=1)

            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # 검증 모드
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # 검증 단계에서는 gradient 계산 불필요
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * x_batch.size(0)

                _, predicted = torch.max(outputs, dim=1)

                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] "
            f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

    return history

# 13. 테스트 평가 함수 정의
def evaluate_model(model, test_loader):
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            test_loss += loss.item() * x_batch.size(0)

            _, predicted = torch.max(outputs, dim=1)

            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()

    test_loss = test_loss / test_total
    test_acc = test_correct / test_total

    return test_loss, test_acc

# 14. 성능 시각화 함수 정의
def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], "b-", label="loss")
    plt.plot(history["val_loss"], "r--", label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title + " - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], "b-", label="accuracy")
    plt.plot(history["val_accuracy"], "r--", label="val_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(title + " - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# 15. 전이학습 1단계: 백본 동결 후 새 분류기만 학습
# PyTorch에서는 requires_grad=True인 파라미터만 optimizer에 전달
# 현재는 classifier의 마지막 Linear 계층만 학습 가능
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model_tl.parameters())
)

print("1단계 전이학습: 백본 동결 + 새 분류기 학습")
history_tl = train_model(
    model=model_tl,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    epochs=10
)

loss, acc = evaluate_model(model_tl, test_loader)
print(f"[Transfer Learning] test loss: {loss:.4f}, test acc: {acc:.4f}")

plot_history(history_tl, "Transfer Learning")

# 16. Fine Tuning 준비
# PyTorch에서는 MobileNetV2의 feature block 중 마지막 일부만 학습 가능하도록 설정
# model_tl.features는 여러 개의 Conv/Block으로 구성된 Sequential
# 여기서는 마지막 10개 모듈만 학습 가능하게 설정
# 주의: TensorFlow의 "마지막 10개 layer"와 PyTorch의 "마지막 10개 module"은
# 내부 구현 단위가 완전히 같지는 않음
# 하지만 의도는 동일함: 앞쪽 특징 추출 층은 동결하고, 뒤쪽 고수준 특징 층만 조금 재학습

# 먼저 feature 전체를 동결
for param in model_tl.features.parameters():
    param.requires_grad = False

# 마지막 10개 feature 모듈만 동결 해제
for layer in model_tl.features[-10:]:
    for param in layer.parameters():
        param.requires_grad = True

# 분류기는 계속 학습 가능해야 함
for param in model_tl.classifier.parameters():
    param.requires_grad = True

print("\n[Fine Tuning 단계 파라미터 정보]")
count_params(model_tl)

# 17. Fine Tuning 학습
# Fine Tuning에서는 이미 학습된 가중치가 크게 망가지지 않도록 매우 작은 학습률을 사용하는 것이 일반적
optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad, model_tl.parameters()),
    lr=1e-6
)

print("2단계 Fine Tuning: 마지막 일부 층만 재학습")
history_ft = train_model(
    model=model_tl,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer_ft,
    epochs=10
)

loss, acc = evaluate_model(model_tl, test_loader)
print(f"[Fine Tuning] test loss: {loss:.4f}, test acc: {acc:.4f}")

plot_history(history_ft, "Fine Tuning")