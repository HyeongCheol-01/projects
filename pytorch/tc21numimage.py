# 손글씨 이미지 읽기
# num.png 이미지를 읽어서 MNIST 모델에 넣을 수 있는 형태로 전처리

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

# 1. 이미지 읽기
im = Image.open('num.png')

# 2. 이미지 크기 변경 및 흑백 변환
# resize((28, 28)) : MNIST와 같은 크기인 28 x 28로 변경
# convert('L')     : 흑백 이미지로 변환
# L 모드: 픽셀값 범위: 0 ~ 255  - 0: 검정, 255: 흰색
img = im.resize((28, 28), Image.Resampling.LANCZOS).convert('L')

# 3. PIL 이미지를 NumPy 배열로 변환
img_np = np.array(img)
print(img_np.shape)  # (28, 28)
print(img_np)

# 4. 원본 흑백 이미지 시각화
plt.imshow(img_np, cmap='Greys')
plt.title('Original grayscale image')
plt.show()

# 5. NumPy 배열을 PyTorch Tensor로 변환
# PyTorch 모델에 입력하려면 일반적으로 Tensor 형태가 필요하다.
# 현재 shape: (28, 28)
# Tensor 변환 후 shape: torch.Size([28, 28])

img_tensor = torch.tensor(img_np, dtype=torch.float32)
print(img_tensor.shape)
print(img_tensor)

# 6. 정규화
# 픽셀값 범위를 0~255에서 0~1로 변환
# 정규화 전: 0 ~ 255
# 정규화 후: 0.0 ~ 1.0

img_tensor = img_tensor / 255.0
print(img_tensor)

# 7. 모델 입력 형태로 변환 - Flatten 방식
# Linear 모델에 넣는 경우
# 기존 NumPy 코드: data = img.reshape([1, 784])
# PyTorch 코드: data = img_tensor.reshape(1, 784)
# shape: torch.Size([1, 784])
# 의미: 1장 이미지, feature 784개

data = img_tensor.reshape(1, 784)
print(data)
print(data.shape)

# 8. 정규화된 이미지 다시 시각화
plt.imshow(data.reshape(28, 28), cmap='Greys')
plt.title('Normalized image')
plt.show()