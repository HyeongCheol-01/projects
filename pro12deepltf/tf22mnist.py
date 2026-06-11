# MNIST는 60,000개의 훈련 이미지와 10,000개의 손글씨 숫자 테스트 이미지를 포함합니다.
# 데이터 세트는 28x28 픽셀 크기의 흑백 이미지로 구성됩니다.

# 손글씨 이미지 읽기
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('num.png')
img = np.array(im.resize((28,28), Image.Resampling.LANCZOS).convert('L'))
# L 모드 : 흑백 이미지 픽셀값이 0 ~ 255 범위(0:검정, 255:흰색)
print(img.shape)

plt.imshow(img, cmap="Greys")
plt.show()

data = img.reshape([1, 784]).astype('float32')
print(data, data.shape)

data = data / 255.0
print(data)