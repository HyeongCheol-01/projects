# 내가 그린 숫자 이미지 분류 예측
# num.png 이미지를 읽어서 PyTorch로 학습한 MNIST 모델에 입력한 뒤
# 숫자 클래스를 예측한다.
# 이 코드는 앞에서 학습했던 MNISTModel 구조와 저장된 mnist_model.pth 파일이 필요하다.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 1. PyTorch 모델 클래스 정의
# 학습할 때 사용했던 모델 구조와 반드시 동일해야 한다.
# 저장된 mnist_model.pth 파일은 가중치만 저장한 파일이므로,
# 모델 구조를 먼저 정의한 뒤 가중치를 불러와야 한다.

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

# 2. CPU 또는 GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 3. 이미지 읽기
im = Image.open("num.png")

# 4. 이미지 크기 변경 및 흑백 변환
# MNIST 데이터는 28 x 28 흑백 이미지이므로, 내가 그린 이미지도 같은 형태로 맞춘다.

img = np.array(
    im.resize((28, 28), Image.Resampling.LANCZOS).convert("L")
)
print(img.shape)  # (28, 28)

# 5. 이미지 확인
plt.imshow(img, cmap="Greys")
plt.title("input image")
plt.show()

# 6. 이미지 전처리 : 학습할 때 MNIST 이미지는 다음 형태로 사용했다.
# x_train shape: (데이터개수, 784) 따라서 예측할 이미지 1장도 다음 형태로 맞춘다.
# data shape: (1, 784) 그리고 픽셀값을 0~255에서 0~1로 정규화한다.
data = img.reshape(1, 784).astype("float32")
data = data / 255.0
print(data.shape)  # (1, 784)

# 7. NumPy 배열을 PyTorch Tensor로 변환
# 모델 입력은 torch.Tensor 타입이어야 한다.
# 입력 x는 float32 타입 사용
data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

# 8. 모델 생성 및 저장된 가중치 불러오기
# PyTorch:
#   1. 모델 구조 생성 -> 2. state_dict 불러오기 -> 3. eval 모드 전환
mymodel = MNISTModel().to(device)

mymodel.load_state_dict(
    torch.load("mnist_model.pth", map_location=device)
)
mymodel.eval()


# 9. 예측
# PyTorch 모델의 출력은 softmax 전 값인 logits이다.
# 확률 형태로 보고 싶으면 torch.softmax()를 적용한다.
# 예측 클래스는 확률이 가장 큰 위치의 index이다.
with torch.no_grad():
    logits = mymodel(data_tensor)
    new_pred = torch.softmax(logits, dim=1)
    pred_label = torch.argmax(new_pred, dim=1).item()

print("new_pred : ", new_pred.cpu().numpy())
print("예측값 : ", pred_label)