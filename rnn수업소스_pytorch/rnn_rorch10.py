실습10) 네이버 쇼핑 리뷰 분류

# 주피터 노트북에서 최초 한 번 실행
!pip install -q konlpy JPype1

import re
import copy
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split

# 네이버 쇼핑 리뷰 데이터 다운로드
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt",
    filename="ratings_total.txt"
)

# 4. 데이터 로드
total_data = pd.read_table( 'ratings_total.txt',  names=['ratings', 'reviews'])
print('전체 리뷰 개수:', len(total_data))
print(total_data.head())


# 5. 평점을 기준으로 긍정/부정 label 생성
# 평점이 4, 5점이면 긍정 1
# 평점이 1, 2점이면 부정 0
total_data['label'] = np.select([total_data['ratings'] > 3], [1], default=0)
print("\nlabel 생성 후 데이터")
print(total_data.head())

print("\n고유값 개수 확인")
print('ratings 고유값 수:', total_data['ratings'].nunique())
print('reviews 고유값 수:', total_data['reviews'].nunique())
print('label 고유값 수:', total_data['label'].nunique())


# 6. 중복 리뷰 제거
total_data = total_data.drop_duplicates( subset=['reviews'])
print('\n중복 제거 후 총 샘플 수:', len(total_data))
print('NULL 값 존재 여부:', total_data.isnull().values.any())

# 7. 결측치 제거
total_data = total_data.dropna(how='any')
print('NULL 제거 후 총 샘플 수:', len(total_data))

# 8. 훈련 데이터와 테스트 데이터 분리
train_data, test_data = train_test_split(total_data,test_size=0.25, random_state=42, stratify=total_data['label'])

# SettingWithCopyWarning 방지를 위해 복사
train_data = train_data.copy()
test_data = test_data.copy()
print('\n훈련용 리뷰 개수:', len(train_data))
print('테스트용 리뷰 개수:', len(test_data))


# 9. 레이블 분포 확인
train_data['label'].value_counts().plot( kind='bar')
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

print("\n훈련 데이터 label 분포")
print(train_data.groupby('label').size().reset_index(name='count'))

# 10. 리뷰 정제 함수
def clean_review(text):
    text = str(text)                            # 문자열로 변환
    text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', text)  # 한글과 공백만 남김
    text = re.sub(r'\s+', ' ', text)                # 여러 공백을 하나로 변경
    text = text.strip()                          # 앞뒤 공백 제거
    return text


# 11. 훈련/테스트 리뷰 정제
train_data['reviews'] = train_data['reviews'].apply( clean_review)
test_data['reviews'] = test_data['reviews'].apply(clean_review)

# 빈 문자열을 NaN으로 변경
train_data['reviews'] = train_data['reviews'].replace('', np.nan)
test_data['reviews'] = test_data['reviews'].replace('', np.nan)

# NaN 제거
train_data = train_data.dropna(how='any')
test_data = test_data.dropna(how='any')

# 인덱스 재정렬
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
print('\n전처리 후 훈련용 샘플 수:', len(train_data))
print('전처리 후 테스트용 샘플 수:', len(test_data))
print("\n훈련 데이터 결측치 확인")
print(train_data.isnull().sum())
print("\n테스트 데이터 결측치 확인")
print(test_data.isnull().sum())

# 12. Okt 형태소 분석기 생성
okt = Okt()
# Okt 동작 테스트
print("\nOkt 형태소 분석 테스트")
print(okt.morphs( '와 이런 것도 상품이라고 차라리 내가 만드는 게 나을 뻔', stem=True))
# 13. 불용어 목록
stopwords = [
    '도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인',
    '듯', '과', '와', '네', '들', '지', '임', '게', '것', '수', '좀', '너무'
]

# 14. 토큰화 및 불용어 제거 함수
def tokenize_and_remove_stopwords(text):
    tokens = okt.morphs( text, stem=True )
    tokens = [ word for word in tokens if word not in stopwords ]
    return tokens


# 15. 훈련/테스트 데이터 토큰화
train_data['tokenized'] = train_data['reviews'].apply(tokenize_and_remove_stopwords)
test_data['tokenized'] = test_data['reviews'].apply(tokenize_and_remove_stopwords)
print("\n토큰화 결과 확인")
print(train_data[ ['reviews', 'tokenized', 'label']].head())


# 16. 긍정/부정 리뷰 단어 빈도 확인
negative_words = np.hstack(train_data[train_data['label'] == 0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data['label'] == 1]['tokenized'].values)

negative_word_count = Counter(negative_words)
positive_word_count = Counter(positive_words)
print('\n부정 리뷰 빈도 상위 20개')
print(negative_word_count.most_common(20))
print('\n긍정 리뷰 빈도 상위 20개')
print(positive_word_count.most_common(20))


# 17. 리뷰 길이 분포 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

positive_len = train_data[train_data['label'] == 1]['tokenized'].map(lambda value: len(value))

ax1.hist(positive_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('\n긍정 리뷰의 평균 길이:',np.mean(positive_len))

negative_len = train_data[train_data['label'] == 0]['tokenized'].map(lambda value: len(value))

ax2.hist(negative_len, color='blue')
ax2.set_title('Negative Reviews')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')

print('부정 리뷰의 평균 길이:', np.mean(negative_len))

fig.suptitle('Words in texts')
plt.show()


# 18. 입력 데이터와 정답 데이터 분리
X_train = train_data['tokenized'].values
y_train = train_data['label'].values

X_test = test_data['tokenized'].values
y_test = test_data['label'].values
print('\nX_train 개수:', len(X_train))
print('y_train 개수:', len(y_train))
print('X_test 개수:', len(X_test))
print('y_test 개수:', len(y_test))

# PyTorch에는 Keras Tokenizer가 없으므로
# 동일한 기능을 수행하는 클래스를 정의한다.
class Tokenizer:
    def __init__( self, num_words=None, oov_token=None):
        self.num_words = num_words
        self.oov_token = oov_token

        self.word_counts = Counter()
        self.word_index = {}
        self.oov_index = None

    def fit_on_texts(self, texts):
        first_order = {}
        order = 0

        for tokens in texts:
            for word in tokens:
                self.word_counts[word] += 1

                if word not in first_order:
                    first_order[word] = order
                    order += 1

        sorted_words = sorted(
            self.word_counts.items(),
            key=lambda item: (
                -item[1],
                first_order[item[0]]
            )
        )

        start_index = 1

        if self.oov_token is not None:
            self.word_index[self.oov_token] = 1
            self.oov_index = 1
            start_index = 2

        for word, count in sorted_words:
            if word == self.oov_token:
                continue

            self.word_index[word] = start_index
            start_index += 1

    def texts_to_sequences(self, texts):
        result = []

        for tokens in texts:
            sequence = []

            for word in tokens:
                index = self.word_index.get(word)

                if index is None:
                    if self.oov_index is not None:
                        sequence.append(self.oov_index)

                    continue

                # num_words보다 큰 인덱스는 OOV로 처리한다.
                if (self.num_words is not None and index >= self.num_words):
                    if self.oov_index is not None:
                        sequence.append(self.oov_index)
                else:
                    sequence.append(index)

            result.append(sequence)

        return result


# 19. Tokenizer 학습
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# 20. 희귀 단어 비율 확인
threshold = 2

total_cnt = len(tokenizer.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq += value

    if value < threshold:
        rare_cnt += 1
        rare_freq += value

print('\n단어 집합(vocabulary)의 크기:', total_cnt)

print('등장 빈도가 {}번 이하인 희귀 단어의 수: {}'.format(threshold - 1,  rare_cnt ))
print('단어 집합에서 희귀 단어의 비율:', (rare_cnt / total_cnt) * 100)
print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율:', (rare_freq / total_freq) * 100)


# 21. 희귀 단어를 제외한 단어 집합 크기 설정
vocab_size = total_cnt - rare_cnt + 2
print('\n최종 단어 집합의 크기:', vocab_size)


# 22. 제한된 단어 집합으로 Tokenizer 재생성
tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(X_train)

# 23. 텍스트를 정수 시퀀스로 변환
X_train = tokenizer.texts_to_sequences( X_train)
X_test = tokenizer.texts_to_sequences( X_test)
print('\nX_train 상위 3개:', X_train[:3])
print('X_test 상위 3개:', X_test[:3])


# 24. 리뷰 길이 확인
print('\n리뷰의 최대 길이:',  max(len(review) for review in X_train))
print('리뷰의 평균 길이:', sum(map(len, X_train)) / len(X_train))

plt.hist( [len(review) for review in X_train],  bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


# 25. 특정 길이 이하 샘플 비율 확인 함수
def below_threshold_len(max_len, nested_list):
    count = 0

    for sentence in nested_list:
        if len(sentence) <= max_len:
            count += 1

    print(
        '전체 샘플 중 길이가 {} 이하인 샘플의 비율: {}'.format(
            max_len, (count / len(nested_list)) * 100 )
    )


# 26. 패딩 기준 길이 설정
max_len = 80
below_threshold_len( max_len, X_train)

# PyTorch 입력용 패딩 함수 : 문장이 길면 앞부분을 자르고, 짧으면 앞부분에 0을 채운다.
def pad_sequences( sequences, maxlen):
    result = np.zeros((len(sequences), maxlen), dtype=np.int64 )

    for index, sequence in enumerate(sequences):
        sequence = sequence[-maxlen:]

        if len(sequence) > 0:
            result[index, -len(sequence):] = sequence

    return result


# 27. 패딩
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences( X_test, maxlen=max_len)
print('\nX_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# 학습 데이터의 마지막 20%를 검증 데이터로 사용한다.
validation_size = int(len(X_train) * 0.2)

X_fit = X_train[:-validation_size]
y_fit = y_train[:-validation_size]

X_val = X_train[-validation_size:]
y_val = y_train[-validation_size:]


# NumPy 배열을 PyTorch Tensor로 변환
X_fit_tensor = torch.tensor( X_fit, dtype=torch.long)
y_fit_tensor = torch.tensor( y_fit, dtype=torch.float32)
X_val_tensor = torch.tensor( X_val, dtype=torch.long)
y_val_tensor = torch.tensor( y_val, dtype=torch.float32)
X_test_tensor = torch.tensor( X_test, dtype=torch.long)
y_test_tensor = torch.tensor( y_test, dtype=torch.float32)

train_dataset = TensorDataset( X_fit_tensor, y_fit_tensor)
validation_dataset = TensorDataset( X_val_tensor, y_val_tensor)
test_dataset = TensorDataset( X_test_tensor, y_test_tensor)

train_loader = DataLoader( train_dataset,  batch_size=64, shuffle=True)
validation_loader = DataLoader( validation_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 28. GRU 감성 분류 모델 생성
embedding_dim = 100
hidden_units = 128

class SentimentGRUModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 정수 형태의 단어 번호를 100차원 벡터로 변환한다.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
embedding_dim=embedding_dim, padding_idx=0)

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_units, batch_first=True )

        # 긍정 여부를 나타내는 하나의 값을 출력한다.
        self.output = nn.Linear(hidden_units,  1 )

    def forward(self, x):
        x = self.embedding(x)

        gru_output, hidden = self.gru(x)

        # GRU의 마지막 은닉 상태를 사용한다.
        x = hidden[-1]

        # BCEWithLogitsLoss를 사용하므로 Sigmoid를 적용하지 않은 logits를 반환한다.
        logits = self.output(x)

        return logits.squeeze(1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('사용 장치:', device)

model = SentimentGRUModel().to(device)
print(model)

# 29. EarlyStopping 설정
patience = 4
best_val_loss = float('inf')
early_stop_count = 0


# 30. ModelCheckpoint 설정
checkpoint_path = 'best_model.pt'
best_val_accuracy = 0.0


# 31. 손실 함수와 최적화 알고리즘 설정
# BCEWithLogitsLoss는 Sigmoid와 이진 교차 엔트로피를 함께 처리한다.
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.RMSprop( model.parameters())

# 32. 모델 구조 출력
print(model)

# 모델 평가 함수
def evaluate(model, data_loader):
    model.eval()

    total_loss = 0.0
    correct_count = 0
    total_count = 0

    with torch.inference_mode():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits,  y_batch  )
            total_loss += ( loss.item() * len(x_batch) )
            probability = torch.sigmoid(logits)
            predicted = (probability > 0.5).float()

            correct_count += (predicted == y_batch).sum().item()
            total_count += len(y_batch)

    loss = total_loss / total_count
    accuracy = correct_count / total_count

    return loss, accuracy


# 33. 모델 학습
epochs = 15
history = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': []
}

for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    correct_count = 0
    total_count = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True) # 이전에 계산된 기울기를 초기화한다.
        logits = model(x_batch)   # 긍정 리뷰 여부에 대한 logits를 예측한다.
        loss = criterion(logits, y_batch )   # 손실값을 계산한다.
        loss.backward()       # 역전파를 수행한다.
        optimizer.step()      # 모델의 가중치를 수정한다.
        total_loss += (loss.item() * len(x_batch))

        probability = torch.sigmoid(logits)
        predicted = (probability > 0.5 ).float()

        correct_count += (predicted == y_batch ).sum().item()
        total_count += len(y_batch)

    train_loss = total_loss / total_count
    train_accuracy = correct_count / total_count

    val_loss, val_accuracy = evaluate( model, validation_loader )

    history['loss'].append(train_loss)
    history['accuracy'].append(train_accuracy)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)

    print(
        f'Epoch {epoch + 1}/{epochs} - '
        f'loss: {train_loss:.4f} - '
        f'accuracy: {train_accuracy:.4f} - '
        f'val_loss: {val_loss:.4f} - '
        f'val_accuracy: {val_accuracy:.4f}'
    )

    # 검증 정확도가 가장 높은 모델을 저장한다.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save( model.state_dict(), checkpoint_path)

        print(
            f'검증 정확도 개선: '
            f'{best_val_accuracy:.4f}, 모델 저장'
        )

    # 검증 손실이 개선됐는지 확인한다.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

        # 검증 손실이 4번 연속 개선되지 않으면 학습을 종료한다.
        if early_stop_count >= patience:
            print('Early stopping')
            break


# 34. 최적 모델 불러오기
loaded_model = SentimentGRUModel().to(device)
loaded_model.load_state_dict( torch.load( checkpoint_path, map_location=device, weights_only=True ))
loaded_model.eval()

# 35. 테스트 데이터 평가
loss, accuracy = evaluate(loaded_model, test_loader)
print("\n테스트 정확도: %.4f" % accuracy)

# 36. 새 리뷰 감성 예측 함수
def sentiment_predict(new_sentence):
    new_sentence = re.sub( r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence )

    new_sentence = okt.morphs(new_sentence, stem=True )

    new_sentence = [ word for word in new_sentence if word not in stopwords ]
    encoded = tokenizer.texts_to_sequences([new_sentence] )

    pad_new = pad_sequences( encoded, maxlen=max_len  )
    input_tensor = torch.tensor( pad_new,  dtype=torch.long,  device=device )

    with torch.inference_mode():
        logits = loaded_model(input_tensor)
        score = torch.sigmoid(logits).item()

    if score > 0.5:
        print( "{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100 ) )
    else:
        print( "{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100) )

# 37. 예측 테스트
sentiment_predict('이 상품 진짜 좋아요... 저는 강추합니다. 대박')
sentiment_predict('진짜 배송도 늦고 개짜증나네요. 뭐 이런 걸 상품이라고 만듬?')
