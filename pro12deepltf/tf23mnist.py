# MNIST는 60,000개의 훈련 이미지와 10,000개의 손글씨 숫자 테스트 이미지를 포함합니다.
# 데이터 세트는 28x28 픽셀 크기의 흑백 이미지로 구성됩니다.
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
print(x_train[0])
print(y_train[0])

# for i in x_train[0]:
#     for j in i:
#         sys.stdout.write('%s  '%j)
#         sys.stdout.write('\n')

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

# 모델 만들기 준비 - 전처리
x_train = x_train.reshape(60000, 784).astype('float32') # 3차원 -> 2차원
x_test = x_test.reshape(10000, 784).astype('float32')
print(x_train[0], x_train.shape)

x_train /= 255.0  # 정규화. 필수는 아니나 모델성능이 향상
x_test /= 255.0
print(x_train[0], x_train.shape)
print(set(map(int, y_test)))    # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# label 원핫 처리 - softmax를 사용하므로
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
print(y_train[0])

# validation data 직접 구성
x_val = x_train[50000:60000]    # 10000개는 학습 도중 검증 데이터로 사용
y_val = y_train[50000:60000]
x_train = x_train[0:50000]  # 50000개는 train 데이터
y_train = y_train[0:50000]
print(x_val.shape, x_train.shape)

# model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation, Dropout

model = Sequential()
model.add(Input(shape =(784, )))
# model.add(Dense(units=64))
# model.add(Activation='relu')
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=10, activation='softmax'))
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, batch_size=128, \
                    validation_data=(x_val, y_val), verbose=2)

score = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
print(f'loss:{score[0]:.4f}, accuracy:{score[1]:.4f}')

#시각화
plt.plot(history.history['loss'], label = 'train loss')
plt.plot(history.history['val_loss'], label = 'val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label = 'train accuracy')
plt.plot(history.history['val_accuracy'], label = 'val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

model.save('tf23model.keras')

mymodel = tf.keras.models.load_model('tf23model.keras')
pred = mymodel.predict(x_test[:1])
print('pred : ', pred)
print('예측값 : ', np.argmax(pred, axis=1)[0])
print('실제값 : ', np.argmax(y_test[0]))