# 성인 남녀 얼굴 이미지 분류 이진 분류
# 라벨 규칙: 0 -> male,  1 -> female

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 2. Google Drive 연결
from google.colab import drive
drive.mount('/content/drive')

# 3. 이미지 경로 및 라벨 수집
image_dir = '/content/drive/MyDrive/person_img/'

image_paths = []
labels = []

for file in os.listdir(image_dir):
    try:
        # 파일명 예: 30_0_0_20170119195539771.jpg
        # 두 번째 값이 gender label
        gender = int(file.split('_')[1])

        # 0: male, 1: female만 사용
        if gender not in [0, 1]:
            continue

        img_path = os.path.join(image_dir, file)

        # 이미지가 정상적으로 읽히는지 확인
        img = cv2.imread(img_path)

        if img is None:
            continue

        image_paths.append(img_path)
        labels.append(gender)

    except:
        # 파일명 규칙이 다르거나 오류가 나는 파일은 제외
        continue

print('전체 이미지 수:', len(image_paths))
print('전체 라벨 수:', len(labels))
print('첫 번째 이미지 경로:', image_paths[0])
print('첫 번째 라벨:', labels[0])

# 4. 첫 번째 이미지 시각화
img = cv2.imread(image_paths[0])
img = cv2.resize(img, (64, 64))

# OpenCV는 BGR, matplotlib은 RGB 사용
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(3, 3))
plt.imshow(img_rgb)

label_name = 'male' if labels[0] == 0 else 'female'
plt.title(f'label: {labels[0]} ({label_name})')
plt.axis('off')
plt.show()

# 5. Train / Test 데이터 분리
train_paths, test_paths, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print('train 이미지 수:', len(train_paths))
print('test 이미지 수:', len(test_paths))

# 6. PyTorch Dataset 클래스 정의
class FaceGenderDataset(Dataset):
    """
    PyTorch Dataset 클래스
    역할:
    - 이미지 파일 경로를 받아 실제 이미지를 읽음
    - 64x64 크기로 resize
    - BGR -> RGB 변환
    - 0~1 범위로 정규화
    - PyTorch 입력 형식인 (C, H, W)로 변환
    - 이미지 tensor와 label tensor 반환
    """

    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 이미지 읽기
        img = cv2.imread(img_path)

        # 이미지 크기 조정
        img = cv2.resize(img, (64, 64))

        # BGR -> RGB 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 정규화: 0~255 -> 0~1
        img = img.astype(np.float32) / 255.0

        # PyTorch Conv2D 입력 형식: (채널, 높이, 너비)
        # 기존: (64, 64, 3)
        # 변환: (3, 64, 64)
        img = np.transpose(img, (2, 0, 1))

        # tensor 변환
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # BCEWithLogitsLoss 사용을 위해 label은 float
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor

# 7. Dataset / DataLoader 생성
train_dataset = FaceGenderDataset(train_paths, y_train)
test_dataset = FaceGenderDataset(test_paths, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# batch 데이터 shape 확인
images, labels_batch = next(iter(train_loader))

print('batch image shape:', images.shape)       # [batch, 3, 64, 64]
print('batch label shape:', labels_batch.shape) # [batch]

# 8. CNN 모델 정의
class GenderCNN(nn.Module):
    """
    성별 이진 분류 CNN 모델
    입력: 이미지 tensor shape: (batch_size, 3, 64, 64)
    출력:logit 1개, sigmoid를 통과시키면 female일 확률로 해석 가능
    주의:
    - 마지막 layer에 Sigmoid를 넣지 않음
    - BCEWithLogitsLoss가 내부적으로 Sigmoid + Binary Cross Entropy를 처리함
    """

    def __init__(self):
        super(GenderCNN, self).__init__()

        self.features = nn.Sequential(
            # 입력: (3, 64, 64)
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3
            ),
            nn.ReLU(),

            # 출력: (32, 62, 62)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 출력: (32, 31, 31)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            ),
            nn.ReLU(),

            # 출력: (64, 29, 29)
            nn.MaxPool2d(kernel_size=2, stride=2)

            # 출력: (64, 14, 14)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 64 * 14 * 14 = 12544
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),

            # 이진 분류이므로 출력 1개
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        # shape: (batch_size, 1) -> (batch_size,)
        x = x.squeeze(1)

        return x

# 9. GPU / CPU 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('사용 장치:', device)

model = GenderCNN().to(device)
print(model)

# 10. 손실 함수와 Optimizer 설정
# 이진 분류용 손실 함수
# BCEWithLogitsLoss = Sigmoid + Binary Cross Entropy
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 11. 학습 함수 정의
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    1 epoch 동안 모델 학습
    PyTorch 학습 흐름:
    1. model.train()
    2. optimizer.zero_grad()
    3. outputs = model(images)
    4. loss 계산
    5. loss.backward()
    6. optimizer.step()
    """

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # loss 누적
        total_loss += loss.item() * images.size(0)

        # logit -> probability
        probs = torch.sigmoid(outputs)

        # 0.5 이상이면 female(1), 아니면 male(0)
        preds = (probs >= 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc

# 12. 평가 함수 정의
def evaluate(model, data_loader, criterion, device):
    """
    모델 평가 함수
    평가 시에는:
    - model.eval()
    - torch.no_grad() 를 사용해서 gradient 계산을 하지 않음
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

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

# 13. 모델 학습 실행
epochs = 20

train_acc_history = []
test_acc_history = []
train_loss_history = []
test_loss_history = []

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    print(
        f'Epoch [{epoch + 1}/{epochs}] '
        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
        f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}'
    )

# 14. 최종 Test 정확도 확인
loss, acc = evaluate(model, test_loader, criterion, device)
print(f'test acc : {acc:.4f}')

# 15. Test 데이터 전체 예측값 확인
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)

        # logit -> probability
        probs = torch.sigmoid(outputs)

        # 확률 0.5 기준으로 이진 분류
        preds = (probs >= 0.5).int().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy().astype(int))

print('예측값:', np.array(all_preds))
print('실제값:', np.array(all_labels))

# 16. Accuracy 시각화
plt.figure(figsize=(6, 4))
plt.plot(train_acc_history, label='train acc')
plt.plot(test_acc_history, label='test acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train/Test Accuracy')
plt.show()

# 17. Loss 시각화
plt.figure(figsize=(6, 4))
plt.plot(train_loss_history, label='train loss')
plt.plot(test_loss_history, label='test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train/Test Loss')
plt.show()

# 18. 5개 이미지 예측 결과 시각화
model.eval()

plt.figure(figsize=(16, 4))

with torch.no_grad():
    for i in range(5):
        # test_dataset에서 i번째 이미지와 라벨 가져오기
        img_tensor, true_label = test_dataset[i]

        # 모델 입력을 위해 batch 차원 추가
        # 기존: (3, 64, 64)
        # 변경: (1, 3, 64, 64)
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # 예측
        output = model(input_tensor)

        # logit -> probability
        prob = torch.sigmoid(output).item()

        # 0.5 기준 분류
        pred_class = 1 if prob >= 0.5 else 0
        true_class = int(true_label.item())

        # 시각화를 위해 tensor를 이미지 형태로 변환
        # 기존: (3, 64, 64)
        # 변경: (64, 64, 3)
        img_np = img_tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))

        is_correct = pred_class == true_class

        label = 'female' if true_class == 1 else 'male'
        prediction = 'female' if pred_class == 1 else 'male'

        title_color = 'black' if is_correct else 'red'

        plt.subplot(1, 5, i + 1)
        plt.imshow(img_np)
        plt.title(
            f'pred: {prediction}\nactual: {label}\nprob: {prob:.2f}',
            color=title_color
        )
        plt.axis('off')

plt.tight_layout()
plt.show()

# 19. 모델 저장
torch.save(model.state_dict(), 'gender_cnn_pytorch.pth')
print('모델 저장 완료: gender_cnn_pytorch.pth')

# 20. 저장된 모델 불러오기 예시
loaded_model = GenderCNN().to(device)
loaded_model.load_state_dict(torch.load('gender_cnn_pytorch.pth', map_location=device))
loaded_model.eval()
print('저장된 모델 불러오기 완료')