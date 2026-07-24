# ============================================================
# 이전 실습에서 만들어진 PyTorch 모델로 새로운 개/고양이 이미지 분류 예측
# 새 이미지도 학습 때와 동일하게 처리해야 한다.
# 학습 시 전처리: Resize(150, 150) ToTensor()
# 예측 시 전처리도 동일: Resize(150, 150)  ToTensor()
# 모델 출력: logit 1개
# 예측 방식:
#   torch.sigmoid(logit) -> dog일 확률
#   p_dog >= 0.5 -> dogs,  p_dog < 0.5 -> cats

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 2. 기본 설정
MODEL_PATH = "chkpoints/catdog_best.pth"
IMG_HEIGHT, IMG_WIDTH = 150, 150

# sigmoid 확률 기준값 : 0.5보다 크거나 같으면 dogs, 작으면 cats, 상황에 따라 0.4 ~ 0.6 정도로 조정 가능
THRESH = 0.5
idx_to_name = { 0: "cats", 1: "dogs" }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 3. 학습 때 사용한 CNN 모델 구조 정의
# 주의: 저장된 state_dict를 불러오려면 학습 때 사용한 모델 구조와 완전히 같아야 한다.
# 이 모델은 이전 PyTorch 개/고양이 분류 코드의 CatDogCNN과 동일해야 한다.
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()

        self.features = nn.Sequential(
            # 입력: (batch, 3, 150, 150)
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            ),
            nn.ReLU(),

            # 출력: (batch, 16, 75, 75)
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ReLU(),

            # 출력: (batch, 32, 37, 37)
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)  # 출력: (batch, 64, 18, 18)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 18 * 18, 512),  # 64 * 18 * 18 = 20736
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.squeeze(1)   # shape: (batch, 1) -> (batch,)
        return x


# 4. 모델 생성 및 저장된 가중치 불러오기
model = CatDogCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# 예측 모드로 전환 : Dropout, BatchNorm 등이 평가 모드로 동작
model.eval()
print("모델 불러오기 완료", model)

# 5. 이미지 전처리 정의
# PyTorch 학습 때 사용한 val_transform과 동일하게 맞춘다.
# transforms.Resize: 이미지를 150x150 크기로 변경
# transforms.ToTensor: PIL Image를 torch.Tensor로 변환
#   픽셀값 0~255를 0~1로 자동 스케일링,  shape: (H, W, C) -> (C, H, W)
preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor()
])

# 6. 전처리 함수
def preprocess_image_func(img_path):
    """
    새 이미지 1장을 PyTorch 모델 입력 형태로 변환하는 함수
    입력: img_path: 이미지 파일 경로
    출력: image_tensor: shape = (1, 3, 150, 150)
    처리 과정:
        1. 이미지 로드 -> 2. RGB 변환 -> 3. Resize  -> 4. ToTensor -> 5. batch 차원 추가
    """

    # 이미지 파일 존재 여부 확인
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")

    # PIL로 이미지 열기
    # convert("RGB")를 사용하면 흑백/투명 이미지도 RGB 3채널로 맞출 수 있음
    img = Image.open(img_path).convert("RGB")

    # transform 적용
    img_tensor = preprocess_transform(img)

    # batch 차원 추가 -  기존: (3, 150, 150), 변경: (1, 3, 150, 150)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

# 7. 단일 이미지 예측 함수
def predict_one_func(img_path, show=True):
    """
    새 이미지 1장에 대해 cats/dogs 예측 수행
    반환:
        {
            "path": 이미지 경로, "pred": 예측 클래스명,
            "p_dog": dog 확률, "p_cat": cat 확률
        }
    """

    x = preprocess_image_func(img_path) # 이미지 전처리
    x = x.to(device)   # GPU 또는 CPU로 이동

    # 평가 시 gradient 계산 불필요
    with torch.no_grad():
        logit = model(x)  # 모델 출력은 logit
        prob_dog = torch.sigmoid(logit).item()  # sigmoid를 적용하여 dog 확률로 변환

    # threshold 기준으로 class 결정
    pred_idx = int(prob_dog >= THRESH)
    pred_name = idx_to_name[pred_idx]

    prob_cat = 1.0 - prob_dog  # 이진 분류이므로 cat 확률은 1 - dog 확률

    # 결과 시각화
    if show:
        img_disp = Image.open(img_path).convert("RGB")
        img_disp = img_disp.resize((IMG_WIDTH, IMG_HEIGHT))

        plt.figure(figsize=(4, 4))
        plt.imshow(img_disp)
        plt.axis("off")
        plt.title(
            f"pred: {pred_name} | "
            f"p(cat)={prob_cat:.2f}, p(dog)={prob_dog:.2f}"
        )
        plt.show()

    return {
        "path": img_path, "pred": pred_name,
        "p_dog": prob_dog, "p_cat": prob_cat
    }

# 새 이미지 예측 실행
res = predict_one_func("myimage.jpeg", show=True)
print(res)