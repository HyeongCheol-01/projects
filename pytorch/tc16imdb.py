# imdb dataset으로 이진 분류 : 영화 리뷰(긍정, 부정)
# train : 25000, test : 25000
#
# PyTorch 버전
#
# 핵심 변환:
# Keras Embedding                -> nn.Embedding
# Keras GlobalAveragePooling1D   -> torch.mean(dim=1)
# Keras Dense                    -> nn.Linear
# Keras Dropout                  -> nn.Dropout
# Keras fit()                    -> 직접 학습 루프 작성
# Keras evaluate()               -> 직접 평가 함수 작성
# Keras ModelCheckpoint          -> torch.save()
# Keras load_model()             -> load_state_dict()

from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split


# -------------------------------------------------------
# 1. 랜덤 고정
# -------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)


# -------------------------------------------------------
# 2. 데이터 읽기
# -------------------------------------------------------
num_words = 10000  # 자주 등장하는 단어 1만 개만 사용

(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=num_words)

print(type(train_data), train_data.shape)  # <class 'numpy.ndarray'> (25000,)
print(type(test_data), test_data.shape)    # <class 'numpy.ndarray'> (25000,)
print(train_data[0], len(train_data[0]))
print(train_label[0])  # 0-부정, 1-긍정


# -------------------------------------------------------
# 3. 0번째 리뷰를 원래 문장으로 복원해 보기
# -------------------------------------------------------
word_index = imdb.get_word_index()

sorted_word_index = sorted(word_index.items(), key=lambda x: x[1])

print("\n[index 기준 앞 10개 단어]")
for word, index in sorted_word_index[:10]:
    print(word, index)

# imdb.load_data()에서는 0~3번을 특수 토큰으로 사용하므로
# 실제 단어 index는 +3을 해야 함
reverse_word_index = {
    index + 3: word
    for word, index in word_index.items()
}

# IMDB 데이터 특수 토큰
reverse_word_index[0] = "<PAD>"      # 패딩
reverse_word_index[1] = "<START>"    # 문장 시작
reverse_word_index[2] = "<UNK>"      # 모르는 단어
reverse_word_index[3] = "<UNUSED>"   # 사용 안 함

decoded_review = " ".join(
    reverse_word_index.get(i, "?")
    for i in train_data[0]
)

print("\n0번째 문장:")
print(decoded_review)

print("\n0번째 라벨:", train_label[0])
print("라벨 의미:", "긍정" if train_label[0] == 1 else "부정")


# -------------------------------------------------------
# 4. 리뷰 길이 확인
# -------------------------------------------------------
review_len = [len(review) for review in train_data]

print("\n리뷰 길이 정보")
print("최소 길이 : ", np.min(review_len))
print("최대 길이 : ", np.max(review_len))
print("평균 길이 : ", np.mean(review_len))
print("중앙 값 : ", np.median(review_len))

plt.figure(figsize=(8, 5))
plt.hist(review_len, bins=50)
plt.xlabel("리뷰길이")
plt.ylabel("건수")
plt.grid(True)
plt.show()


# -------------------------------------------------------
# 5. Padding 함수 정의
# -------------------------------------------------------
# Keras pad_sequences 역할을 PyTorch/Numpy 방식으로 직접 구현
#
# maxlen=200
# - 길이가 200보다 짧으면 앞쪽에 0을 채움
# - 길이가 200보다 길면 앞쪽을 자르고 마지막 200개만 사용
#
# Keras pad_sequences 기본 동작:
# padding='pre', truncating='pre'
def pad_sequences_np(sequences, maxlen):
    result = np.zeros((len(sequences), maxlen), dtype=np.int64)

    for i, seq in enumerate(sequences):
        seq = np.array(seq, dtype=np.int64)

        if len(seq) >= maxlen:
            result[i] = seq[-maxlen:]
        else:
            result[i, -len(seq):] = seq

    return result


maxlen = 200

x_train = pad_sequences_np(train_data, maxlen=maxlen)
x_test = pad_sequences_np(test_data, maxlen=maxlen)

y_train = np.array(train_label).astype(np.float32)
y_test = np.array(test_label).astype(np.float32)

print("\n패딩 후 shape")
print("x_train : ", x_train.shape)  # (25000, 200)
print("x_test  : ", x_test.shape)   # (25000, 200)
print("y_train : ", y_train.shape)  # (25000,)
print("y_test  : ", y_test.shape)   # (25000,)

print("\n패딩된 1번째 리뷰:")
print(x_train[1])


# -------------------------------------------------------
# 6. train / validation 분리
# -------------------------------------------------------
# Keras validation_split=0.2 역할
# 여기서는 train 데이터에서 20%를 검증 데이터로 분리
x_train_part, x_val, y_train_part, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("\n분리 후 shape")
print("x_train_part:", x_train_part.shape)
print("x_val       :", x_val.shape)
print("x_test      :", x_test.shape)


# -------------------------------------------------------
# 7. Tensor 변환
# -------------------------------------------------------
# Embedding 입력은 정수 index여야 하므로 x는 long 타입
# BCEWithLogitsLoss는 y shape이 (N, 1)이어야 하므로 reshape
x_train_tensor = torch.tensor(x_train_part, dtype=torch.long)
y_train_tensor = torch.tensor(y_train_part.reshape(-1, 1), dtype=torch.float32)

x_val_tensor = torch.tensor(x_val, dtype=torch.long)
y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

x_test_tensor = torch.tensor(x_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)


# -------------------------------------------------------
# 8. DataLoader 생성
# -------------------------------------------------------
batch_size = 512

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# -------------------------------------------------------
# 9. PyTorch 모델 정의
# -------------------------------------------------------
# Keras 모델 구조:
#
# Input(shape=(200,))
# Embedding(input_dim=10000, output_dim=32)
# GlobalAveragePooling1D()
# Dense(32, relu)
# Dropout(0.3)
# Dense(16, relu)
# Dropout(0.3)
# Dense(1, sigmoid)
#
# PyTorch에서는 마지막 sigmoid를 모델에 넣지 않음
# 대신 BCEWithLogitsLoss를 사용
#
# 예측할 때만 torch.sigmoid() 적용
class IMDBClassifier(nn.Module):
    def __init__(self, num_words, embed_dim=32):
        super().__init__()

        # 단어 index를 32차원 밀집 벡터로 변환
        # padding_idx=0:
        # 0번 PAD 토큰의 임베딩은 0 벡터로 유지
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.fc1 = nn.Linear(embed_dim, 32)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.3)

        self.output = nn.Linear(16, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape:
        # (batch_size, 200)

        x = self.embedding(x)
        # embedding 후 shape:
        # (batch_size, 200, 32)

        # GlobalAveragePooling1D와 같은 역할
        # 200개의 단어 벡터를 평균내서 리뷰 하나를 32차원 벡터로 요약
        x = torch.mean(x, dim=1)
        # pooling 후 shape:
        # (batch_size, 32)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # sigmoid는 적용하지 않음
        # BCEWithLogitsLoss가 내부적으로 sigmoid + BCE를 처리
        x = self.output(x)

        return x


model = IMDBClassifier(num_words=num_words, embed_dim=32).to(device)

print("\n모델 구조")
print(model)


# -------------------------------------------------------
# 10. 손실 함수와 optimizer
# -------------------------------------------------------
# Keras:
# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
#
# PyTorch:
# BCEWithLogitsLoss = sigmoid + binary_crossentropy
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# -------------------------------------------------------
# 11. 평가 함수 정의
# -------------------------------------------------------
def evaluate_model(model, data_loader):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).float()

            total_loss += loss.item() * batch_x.size(0)
            total_correct += (pred == batch_y).sum().item()
            total_count += batch_x.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count

    return avg_loss, avg_acc


# -------------------------------------------------------
# 12. 모델 저장 폴더 준비
# -------------------------------------------------------
MODEL_DIR = "./imdb_model/"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

modelpath = "./imdb_model/imdb_best_pytorch.pth"


# -------------------------------------------------------
# 13. 학습
# -------------------------------------------------------
# Keras EarlyStopping, ModelCheckpoint를 PyTorch 방식으로 직접 구현
epochs = 50
patience = 3

best_val_loss = np.inf
patience_count = 0

history = {
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

for epoch in range(epochs):
    model.train()

    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 1) 예측값 계산
        logits = model(batch_x)

        # 2) 손실 계산
        loss = criterion(logits, batch_y)

        # 3) 이전 gradient 초기화
        optimizer.zero_grad()

        # 4) 역전파
        loss.backward()

        # 5) 파라미터 업데이트
        optimizer.step()

        # train loss 누적
        train_loss_sum += loss.item() * batch_x.size(0)

        # train accuracy 계산
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()

        train_correct += (pred == batch_y).sum().item()
        train_total += batch_x.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    val_loss, val_acc = evaluate_model(model, val_loader)

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"- loss: {train_loss:.4f} "
        f"- accuracy: {train_acc:.4f} "
        f"- val_loss: {val_loss:.4f} "
        f"- val_accuracy: {val_acc:.4f}"
    )

    # ModelCheckpoint
    # val_loss가 가장 낮은 모델만 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_count = 0

        torch.save(model.state_dict(), modelpath)
        print(f"  -> best model 저장: {modelpath}")

    else:
        patience_count += 1

    # EarlyStopping
    # val_loss가 patience 횟수만큼 개선되지 않으면 학습 중단
    if patience_count >= patience:
        print(f"\nEarlyStopping 발생: {epoch + 1} epoch에서 학습 중단")
        break


# -------------------------------------------------------
# 14. 테스트 데이터 평가
# -------------------------------------------------------
test_loss, test_acc = evaluate_model(model, test_loader)

print("\n[현재 모델 테스트 평가]")
print("테스트 평가 손실 : ", test_loss)
print("테스트 평가 정확도 : ", test_acc)


# -------------------------------------------------------
# 15. loss, accuracy 시각화
# -------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["accuracy"], label="train accuracy")
plt.plot(history["val_accuracy"], label="val accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# -------------------------------------------------------
# 16. 저장된 best model 읽어 평가
# -------------------------------------------------------
print("\n\n저장된 모델 읽어 분류 예측")

best_model = IMDBClassifier(num_words=num_words, embed_dim=32).to(device)
best_model.load_state_dict(torch.load(modelpath, map_location=device))
best_model.eval()

best_loss, best_acc = evaluate_model(best_model, test_loader)

print("best_model 평가 손실 : ", best_loss)
print("best_model 평가 정확도 : ", best_acc)


# -------------------------------------------------------
# 17. 기존 데이터를 사용해 예측
# -------------------------------------------------------
new_data = x_test_tensor[:5]
new_label = y_test_tensor[:5]

new_data = new_data.to(device)

with torch.no_grad():
    logits = best_model(new_data)

    # logits를 sigmoid에 통과시켜 긍정 확률로 변환
    pred_prob = torch.sigmoid(logits)

    # 0.5 이상이면 긍정, 미만이면 부정
    pred_class = (pred_prob >= 0.5).int()

pred_prob_np = pred_prob.cpu().numpy()
pred_class_np = pred_class.cpu().numpy().ravel()
new_label_np = new_label.numpy().astype(int).ravel()

print("예측 확률 : ", pred_prob_np.ravel())
print("예측 값 : ", pred_class_np)
print("실제 값 : ", new_label_np)

for i in range(5):
    result = "긍정" if pred_class_np[i] == 1 else "부정"
    real = "긍정" if new_label_np[i] == 1 else "부정"

    print(
        f"{i + 1}번 리뷰 예측:{result}, "
        f"실제:{real}, "
        f"긍정확률:{pred_prob_np[i][0]:.3f}"
    )