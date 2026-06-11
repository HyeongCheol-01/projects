# 합성곱의 원리 이해 - 컵의 특징 추출

import matplotlib.pyplot as plt
from scipy.ndimage import correlate
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize

# 컵 이미지를 읽어와 (64, 64)로 리사이즈
im = rgb2gray(data.coffee())
im = resize(im, (64,64))
print(im.shape) # (64, 64)

plt.axis('off')
plt.imshow(im, cmap='gray')
plt.show()

# 합성곱 필터 (3 * 3)
filter = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])