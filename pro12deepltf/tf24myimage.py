# 내가 그린 숫자 이미지 분류 예측

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
data /= 255.0

import tensorflow as tf
mymodel = tf.keras.models.load_model('tf23model.keras')
new_pred = mymodel.predict(data, verbose=0)
print('new_pred : ', new_pred)
print('예측값 : ', np.argmax(new_pred, axis=1)[0])
