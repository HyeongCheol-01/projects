# Fashion-MNIST 데이터셋 분류 : MNIST와 구조는 동일
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 2. Fashion-MNIST 클래스 이름
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
    'Sandal','Shirt','Sneaker','Bag','Ankle boot'
]

# 3. 데이터 전처리 설정
# TensorFlow 코드에서는 x_train / 255.0으로 직접 정규화했다.
# PyTorch에서는 transforms.ToTensor()가 다음 작업을 해준다.
# 1. PIL Image 또는 NumPy 이미지를 Tensor로 변환
# 2. 픽셀값을 0~255에서 0~1 범위로 자동 정규화
# 3. shape을 (28, 28)에서 (1, 28, 28)로 변경
# 즉, ToTensor() 결과: torch.Size([1, 28, 28])
transform = transforms.ToTensor()

# 4. Fashion-MNIST 데이터 읽기
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)
print(len(train_dataset), len(test_dataset)) # 60000 10000

# 5. 데이터 shape 확인
x0, y0 = train_dataset[0]
print("첫 번째 이미지 shape:", x0.shape)
print("첫 번째 라벨:", y0)
print("첫 번째 이미지 픽셀값:")
print(x0)

# test label 종류 확인
test_labels = [label for _, label in test_dataset]
print(set(map(int, test_labels))) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# 6. 이미지 25개 미리보기
# PyTorch 이미지 shape: (channel, height, width) = (1, 28, 28)
# matplotlib으로 출력할 때는 (28,28)형태가 편하므로 squeeze()로 channel 차원을 제거한다.
plt.figure(figsize=(10, 10))

for i in range(25):
    img, label = train_dataset[i]

    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[label])
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()

# 7. DataLoader 생성
batch_size = 128

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# 8. PyTorch 모델 정의
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)

model = FashionMNISTModel().to(device)
print(model)

# 9. 손실함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 10. 학습
epochs = 10

for epoch in range(epochs):
    model.train()

    train_total_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item() * batch_x.size(0)
        pred = torch.argmax(outputs, dim=1)
        train_correct += (pred == batch_y).sum().item()
        train_total += batch_y.size(0)

    train_loss = train_total_loss / train_total
    train_acc = train_correct / train_total

    print(
        f"Epoch [{epoch + 1:02d}/{epochs}] "
        f"loss:{train_loss:.4f}, accuracy:{train_acc:.4f}"
    )

# 11. 테스트 데이터 평가
# 평가 시에는 model.eval() 사용
# Dropout, BatchNorm 등이 있을 경우 학습 모드와 평가 모드 동작이 달라진다.
# 현재 모델에는 Dropout은 없지만, PyTorch 습관상 평가 시 model.eval()을 사용한다.
# with torch.no_grad(): 평가나 예측 시 gradient 계산을 하지 않도록 설정한다.
model.eval()

test_total_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        test_total_loss += loss.item() * batch_x.size(0)

        pred = torch.argmax(outputs, dim=1)
        test_correct += (pred == batch_y).sum().item()
        test_total += batch_y.size(0)

test_loss = test_total_loss / test_total
test_acc = test_correct / test_total
print(f"test_loss:{test_loss:.4f}, test_acc:{test_acc:.4f}")

# 12. 전체 test 데이터 예측 확률 구하기
# TensorFlow:  pred = model.predict(x_test)
# PyTorch: test_loader를 돌면서 예측값을 모은다.
# 모델 출력은 logits이므로 확률값이 필요하면 softmax를 적용한다.
pred_list = []
true_list = []
img_list = []

model.eval()

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x_device = batch_x.to(device)

        logits = model(batch_x_device)
        probs = torch.softmax(logits, dim=1)

        pred_list.append(probs.cpu().numpy())
        true_list.append(batch_y.numpy())
        img_list.append(batch_x.numpy())

pred = np.concatenate(pred_list, axis=0)
y_test = np.concatenate(true_list, axis=0)
x_test = np.concatenate(img_list, axis=0)

print(pred[0])
print("예측값 : ", np.argmax(pred[0]))
print("실제값 : ", y_test[0])

# 13. 각 이미지 출력용 함수 : 예측 이미지와 실제 레이블 비교
# pred: 모델의 softmax 예측 확률 배열
# y_true: 실제 정답 라벨
# x_img:  PyTorch 이미지 배열     shape: (개수, 1, 28, 28)

def plot_image(i, pred, y_true, x_img):
    pred_arr = pred[i]
    true_label = y_true[i]

    # x_img[i] shape: (1, 28, 28)
    # 시각화할 때는 squeeze()로 (28, 28)로 변경
    img = x_img[i].squeeze()

    pred_label = np.argmax(pred_arr)
    pred_percent = 100 * np.max(pred_arr)

    color = "blue" if pred_label == true_label else "red"

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap="gray")
    plt.xlabel(
        f"예측:{class_names[pred_label]} {pred_percent:.0f}%\n"
        f"실제:{class_names[true_label]}",
        color=color
    )

# 14. 각 이미지에 라벨 등의 정보 표시 : 막대 그래프
def plot_values_arr(i, pred, y_true):
    pred_arr = pred[i]
    true_label = y_true[i]
    pred_label = np.argmax(pred_arr)

    plt.xticks(range(10), class_names, rotation=45, ha="right")
    plt.yticks([])
    plt.ylim([0, 1])

    bars = plt.bar(range(10), pred_arr)

    bars[pred_label].set_color("red")   # 예측값
    bars[true_label].set_color("blue")  # 실제값

# 15. 이미지 1개 예측 결과 출력
def show_one_prediction(i, pred, y_true, x_img):
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plot_image(i, pred, y_true, x_img)

    plt.subplot(1, 2, 2)
    plot_values_arr(i, pred, y_true)

    plt.tight_layout()
    plt.show()

show_one_prediction(1, pred, y_test, x_test)

# 16. 여러 이미지 예측 결과 출력 : 3 x 3 형태로 여러 이미지의 예측 결과 확인
def show_prediction_grid(start, pred, y_true, x_img, rows=3, cols=3):
    plt.figure(figsize=(9, 9))

    for n in range(rows * cols):
        i = start + n

        pred_label = np.argmax(pred[i])
        true_label = y_true[i]
        pred_percent = 100 * np.max(pred[i])

        img = x_img[i].squeeze()

        color = "blue" if pred_label == true_label else "red"

        plt.subplot(rows, cols, n + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap="gray")
        plt.xlabel(
            f"예측:{class_names[pred_label]} {pred_percent:.0f}%\n"
            f"실제:{class_names[true_label]}",
            color=color
        )

    plt.tight_layout()
    plt.show()

show_prediction_grid(0, pred, y_test, x_test)  # 0번부터 9개 보기
show_prediction_grid(15, pred, y_test, x_test)  # 15번부터 9개 보기