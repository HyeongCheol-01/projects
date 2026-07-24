# MNIST 손글씨 숫자 분류
# MNIST는 60,000개의 훈련 이미지와 10,000개의 테스트 이미지로 구성
# 각 이미지는 28 x 28 픽셀 흑백 이미지

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 2. MNIST 데이터 읽기
# PyTorch 코드에서는 sklearn의 fetch_openml을 사용해서 MNIST를 불러온다.
# 단, 인터넷 환경이 안 되거나 다운로드가 오래 걸릴 수 있다.
# 이미 torchvision을 사용 가능한 환경이라면 torchvision.datasets.MNIST를 써도 된다.
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
x = mnist.data.astype("float32")       # shape: (70000, 784)
y = mnist.target.astype("int64")       # shape: (70000,)
print(x.shape, y.shape)
print(x[0])
print(y[0])

# 3. TensorFlow 코드와 같은 구조로 train/test 분리
# 원래 MNIST 구성: train: 60000개, test : 10000개
# fetch_openml도 일반적으로 앞 60000개가 train, 뒤 10000개가 test로 구성되어 있다.
x_train = x[:60000]
y_train = y[:60000]
x_test = x[60000:]
y_test = y[60000:]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 784) (60000,) (10000, 784) (10000,)

# 4. 이미지 확인용 출력 : 현재 x_train[0]은 이미 784개로 펼쳐진 상태
# 시각화할 때는 다시 28 x 28로 reshape 한다.
print(x_train[0])
print(y_train[0])

# 5. 정규화 : 정규화는 필수는 아니지만 모델 학습 안정성과 성능 향상에 도움이 된다.
x_train = x_train / 255.0
x_test = x_test / 255.0
print(x_train[0], x_train.shape)
print(set(map(int, y_test)))  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# 6. PyTorch에서는 라벨 원핫 인코딩을 하지 않음
# nn.CrossEntropyLoss()를 사용하므로 y는 정수 라벨 그대로 사용
# 예) 숫자 5의 라벨은 그냥 5 - [0,0,0,0,0,1,0,0,0,0]으로 만들지 않는다.
print("y_train 첫 번째 라벨:", y_train[0])

# 7. validation data 직접 구성
# train 데이터 60000개 중 뒤쪽 10000개를 validation으로 사용
# x_val   : 10000개 , x_train : 50000개
x_val = x_train[50000:60000]
y_val = y_train[50000:60000]
x_train = x_train[0:50000]
y_train = y_train[0:50000]
print(x_val.shape, x_train.shape)  # (10000, 784) (50000, 784)

# 8. NumPy 배열을 PyTorch Tensor로 변환
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 9. Dataset, DataLoader 생성
batch_size = 128
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False )
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 10. PyTorch 모델 정의
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)

model = MNISTModel().to(device)
print(model)

# 11. 손실함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 12. 학습
epochs = 20

history = {
    "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []
}

for epoch in range(epochs):
    # 학습 모드 : Dropout이 활성화된다.
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

    train_avg_loss = train_total_loss / train_total
    train_avg_acc = train_correct / train_total

    # 검증 모드 : Dropout이 비활성화된다.
    model.eval()

    val_total_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            val_total_loss += loss.item() * batch_x.size(0)

            pred = torch.argmax(outputs, dim=1)
            val_correct += (pred == batch_y).sum().item()
            val_total += batch_y.size(0)

    val_avg_loss = val_total_loss / val_total
    val_avg_acc = val_correct / val_total

    history["loss"].append(train_avg_loss)
    history["accuracy"].append(train_avg_acc)
    history["val_loss"].append(val_avg_loss)
    history["val_accuracy"].append(val_avg_acc)

    print(
        f"Epoch [{epoch + 1:02d}/{epochs}] "
        f"loss: {train_avg_loss:.4f}, "
        f"accuracy: {train_avg_acc:.4f}, "
        f"val_loss: {val_avg_loss:.4f}, "
        f"val_accuracy: {val_avg_acc:.4f}"
    )

# 13. 테스트 데이터 평가
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
print(f"loss:{test_loss:.4f}, accuracy:{test_acc:.4f}")

# 14. 학습 결과 시각화
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(history["accuracy"], label="train accuracy")
plt.plot(history["val_accuracy"], label="val accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# 15. 모델 저장
torch.save(model.state_dict(), "mnist_model.pth")
print("모델 저장 완료: mnist_model.pth")

# 16. 모델 불러오기
mymodel = MNISTModel().to(device)
mymodel.load_state_dict(
    torch.load("mnist_model.pth", map_location=device)
)
mymodel.eval()
print("모델 불러오기 완료")


# 17. 테스트 데이터 1개 예측
# PyTorch에서는 x_test_tensor[1:2]를 모델에 넣는다.
# 모델 출력은 logits이므로, 확률을 보고 싶으면 softmax를 적용한다.
with torch.no_grad():
    sample_x = x_test_tensor[1:2].to(device)
    logits = mymodel(sample_x)
    pred_prob = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(pred_prob, dim=1).item()

print("pred : ", pred_prob.cpu().numpy())
print("예측값 : ", pred_label)
print("실제값 : ", y_test_tensor[1].item())

# 18. 예측 이미지 확인
plt.imshow(x_test_tensor[1].reshape(28, 28), cmap="gray")
plt.title(f"pred: {pred_label}, true: {y_test_tensor[1].item()}")
plt.show()