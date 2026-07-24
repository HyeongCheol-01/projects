# DQN : Q-learning을 딥러닝 신경망으로 확장한 강화학습 알고리즘
# Q-table 대신 신경망을 사용
# 구조
"""
환경 Environment
   ->
현재 상태 state
   ->
행동별 Q값 예측
   ->
행동 action 선택
   ->
환경에 action 실행
   ->
reward, next_state, done (경험)
   ->
ReplayBuffer에 저장
   ->
랜덤하게 샘플링
   ->
Target Network로 target을 갱신
   ->
Q-Network 학습 (현재 상태를 입력 받아 각 행동의 Q값을 갱신)

*** DQN 동작 비유 ***
학생(Q-Network)이 환경에서 문제를 경험함 : Q-Network가 action을 선택해 행동하면
                                           state, action, reward, next_state, done 발생
    ->
문제은행(Replay Buffer)에 저장 : (state, action, reward, next_state, done)
    ->
문제은행에서 랜덤으로 문제를 꺼냄 : Replay Buffer에서 batch sample
    ->
정답지(Target Network)를 보고 목표값 계산 : target = reward + gammar * max(TargetNetwork(next_state))
    ->
학생 신경망을 갱신 : Q-Network 학습, loss = 현재 Q값 - target Q값


~~~ 이전 CartPole 유지하기 예제를 DQN으로 작업 ~~~
'기존 Q-table 방식'             'DQN 방식'
상태를 이산화해야 함            연속된 상태를 그대로 사용
Q-table로 Q값 예측              신경망 모델이 Q값 예측
argmax(Q[state])                argmax(model.predict(state))
Q[state][action] = r+감마...    model.fit()으로 현재상태 갱신
"""

# 목표 : 200번의 step(시간) 단계 동안 Pole이 넘어지지 않고 유지하기
# 종료 : Pole이 수직으로부터 약 ±12도 이상 기울거나, 카트가 환경 밖으로 벗어나거나,
#        최대 시간 단계인 200에 도달한 경우

# !pip install gymnasium[classic-control]
# !pip install torch

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

env = gym.make("CartPole-v1")      # 환경 생성
num_actions = int(env.action_space.n)   # 에이전트가 취할 수 있는 행동의 수
# CartPole에서는 2개 -- 0 : 왼쪽으로 힘을 가함, 1 : 오른쪽으로 힘을 가함

state_dim = int(env.observation_space.shape[0])
# 환경에서 반환하는 state 공간
# CartPole의 경우 상태는 연속값을 갖는 4차원 벡터  [카트 위치, 카트 속도, 막대 각도, 막대 각속도]
print(num_actions, state_dim)

# 2. DQN 모델 정의 : Q-value 예측용 신경망 모델
class DQN(nn.Module):
   def __init__(self, state_dim, num_actions):
      super().__init__()

      # PyTorch에서는 Sequential을 nn.Sequential로 구성
      # 입력층 : state_dim, CartPole에서는 4차원
      # 은닉층 : Dense에 해당하는 nn.Linear 사용
      # 출력층 : num_actions, 각 행동에 대한 Q값 출력
      self.net = nn.Sequential(
         nn.Linear(state_dim, 64),
         nn.ReLU(),

         nn.Linear(64, 64),
         nn.ReLU(),

         nn.Linear(64, num_actions)
         # 출력층: 각 행동의 Q값 출력
         # softmax 사용하지 않음.   예: num_actions=2이면 출력은 [Q(s, 0), Q(s, 1)]
         # Q값은 확률이 아니라 제한 없는 실수값이므로 activation 없이 linear 출력
      )

   def forward(self, x):
      return self.net(x)

# Main Q-Network
# 매 상태에서 Q(s, a)를 예측하고 optimizer.step()을 통해 가중치 갱신
model = DQN(state_dim, num_actions).to(device)

# Target Network
# 학습 중에는 고정된 가중치를 유지하며, 일정 주기마다 model의 가중치를 복사함
# 사용 목적 : Q-learning의 moving target 문제를 줄여 학습 안정성 향상
target_model = DQN(state_dim, num_actions).to(device)

# model의 현재 가중치를 target_model에 복사하는 작업
target_model.load_state_dict(model.state_dict())
target_model.eval()   # target_model은 직접 학습하지 않으므로 평가 모드로 둠

# 전체 흐름 요약
# model        : 현재 상태에서 Q(s, a)를 예측하고 학습함
# target_model : 미래 상태 s'에서의 Q(s', a')를 계산할 때 사용
# load_state_dict() : 일정 주기마다 현재 모델의 가중치를 타겟 모델로 복사

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()   # 예측 Q값과 target Q값의 차이를 최소화

optimizer = optim.Adam(model.parameters(), lr=0.0005) 
# 하이퍼 파라미터
gamma = 0.99   # 할인율 Discount Factor  미래 보상의 중요도를 결정함
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

# Replay buffer에서 꺼내 학습에 사용하는 샘플 수
# 작으면 학습이 불안정하고, 너무 크면 속도 느려짐. CartPole처럼 가벼운 환경에서는 32~64 정도가 적당
batch_size = 64

# deque : 양방향 큐 자료구조 : append()와 popleft()가 빠르고 효율적
# 경험 리플레이 버퍼 : (state, action, reward, next_state, done) 경험을 저장해서 랜덤 샘플링
# 오래된 경험은 자동으로 삭제됨
memory = deque(maxlen=5000)
episodes = 100      # 300 ~ 1000 정도 충분히 주면 더 안정적으로 학습됨

# 타겟 Q 네트워크 갱신 주기
# Q_target을 너무 자주 바꾸면 불안정하고, 너무 늦게 바꾸면 학습 속도가 느릴 수 있음
# 5~10 에피소드마다 target_model.load_state_dict(model.state_dict())로 갱신
update_target_every = 5

reward_list = []

# 학습
for ep in range(episodes):
   # 에피소드 초기화.  state는 현재 관측값:  CartPole의 state는 4개 float: 위치, 속도, 각도, 각속도
   state, _ = env.reset()
   total_reward = 0     # 이번 에피소드 동안 받은 보상의 총합
   done = False   # 에피소드 종료 조건. 막대가 너무 기울거나 카트가 벗어난 경우 True

   # DQN의 핵심 루프 : 한 에피소드 내에서 에이전트가 행동하고 학습하는 과정
   # state → action → next_state → reward 과정을 반복
   while not done:
      # PyTorch 신경망은 Tensor 입력을 원함.  shape: [batch_size, state_dim]
      # 현재 state 하나만 넣으므로 [1, state_dim] 형태로 변환
      state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

      # epsilon-greedy 행동 선택
      if np.random.rand() < epsilon:
         action = np.random.choice(num_actions)   # 탐험: 무작위 행동 선택
      else:
         # 활용: 현재 Q-network가 예측한 Q값 중 가장 큰 행동 선택
         with torch.no_grad():
               q_values = model(state_tensor)
               action = torch.argmax(q_values, dim=1).item()

      # 환경 실행 : 선택한 action을 환경에 전달하고 다음 상태, 보상, 종료 여부를 받음
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      # terminated : 막대가 넘어지거나 카트가 범위를 벗어난 경우
      # truncated  : 최대 step 수에 도달한 경우

      # 실패 시 보상 페널티 적용
      # 막대가 안 넘어진 상태(done=False) → reward = 1 유지
      # 막대가 넘어진 상태(terminated=True) → modified_reward = -10
      if terminated:
         modified_reward = -10
      else:
         modified_reward = reward

      # Replay Buffer에 경험 저장 :  에이전트의 (s, a, r, s', done) 경험을 저장
      # 나중에 무작위로 꺼내 학습함. 샘플 다양성 확보, 데이터 간 상관관계 완화
      memory.append((state, action, modified_reward, next_state, done))

      state = next_state  # 상태 갱신
      total_reward += reward   # 원래 환경 보상을 기준으로 총 보상 누적

      # Replay Buffer에서 batch 추출 후 학습
      if len(memory) >= batch_size:
         # 경험 버퍼에서 무작위로 batch_size개의 경험 선택
         minibatch = random.sample(memory, batch_size)

         # minibatch에 있는 경험을 학습용 배열로 가공
         states = np.array([sample[0] for sample in minibatch], dtype=np.float32)
         actions = np.array([sample[1] for sample in minibatch], dtype=np.int64)
         rewards = np.array([sample[2] for sample in minibatch], dtype=np.float32)
         next_states = np.array([sample[3] for sample in minibatch], dtype=np.float32)
         dones = np.array([sample[4] for sample in minibatch], dtype=np.float32)

         # numpy 배열을 PyTorch Tensor로 변환
         states_tensor = torch.FloatTensor(states).to(device)
         actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
         rewards_tensor = torch.FloatTensor(rewards).to(device)
         next_states_tensor = torch.FloatTensor(next_states).to(device)
         dones_tensor = torch.FloatTensor(dones).to(device)

         # Main Q-Network로 현재 상태의 Q값 예측
         # q_values shape: [batch_size, num_actions]
         q_values = model(states_tensor)

         # 선택한 행동의 Q값만 추출
         # 예: actions_tensor가 [[0], [1], [0]...]이면 각 샘플에서 해당 action의 Q값만 가져옴
         current_q = q_values.gather(1, actions_tensor).squeeze(1)

         # Target Network로 다음상태의 Q값예측. target_model은 직접학습하지않으므로 no_grad 사용
         with torch.no_grad():
               next_q_values = target_model(next_states_tensor)

               # 다음 상태에서 가능한 행동 중 가장 큰 Q값
               max_next_q = torch.max(next_q_values, dim=1)[0]

               # Bellman target 계산 : done이면 미래 Q값을 보지 않고 reward만 사용
               # done이 아니면 reward + gamma * max_next_q 사용
               target_q = rewards_tensor + gamma * max_next_q * (1 - dones_tensor)

         loss = criterion(current_q, target_q)  # 손실 계산 : 현재 Q값과 target Q값의 차이를 최소화

         # Main Q-Network 학습
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

   reward_list.append(total_reward)  # 한 에피소드 동안 받은 보상 합계를 저장

   # epsilon 감소 : 초반에는 탐험을 많이 하고, 학습이 진행되면 예측 Q값이 큰 행동을 더 많이 선택
   # epsilon_min에 도달하면 최소 탐험 확률 유지
   if epsilon > epsilon_min:
      epsilon *= epsilon_decay
      epsilon = max(epsilon, epsilon_min)

   # 타겟 모델 갱신
   # 왜 필요한가? DQN은 Q(s, a)를 업데이트할 때 target Q(s', a')를 사용하는데, 
   # 이 둘이 동시에 계속 바뀌면 학습이 불안정해짐
   # 해결 방법: model은 매번 학습하지만 target_model은 일정 주기마다만 따라가게 함
   if ep % update_target_every == 0:
      target_model.load_state_dict(model.state_dict())  # model의 가중치를 target_model에 복사
   # 학습 로그 출력 : 10 에피소드마다 현재 에피소드 보상과 epsilon 출력
   # Reward가 증가하고 Epsilon이 감소하면 좋은 학습 흐름
   if ep % 10 == 0:
      print(f'Episode {ep}: Reward = {total_reward:.1f}, Epsilon = {epsilon:.3f}')

# np.convolve() 이동 평균 구하기 알아보기 ---
# data = np.array([1,2,3,4,5])
# window_size = 3
# avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
# print(avg)

# 보상 시각화
plt.figure(figsize=(10, 4))

# data 배열에 대해 window_size 크기의 이동 평균 계산용 함수
# 이동 평균을 계산해서 학습 보상 그래프를 더 부드럽게 시각화
# np.ones(window_size) / window_size : 평균 필터
# np.convolve(data, ...) : 데이터를 슬라이딩하며 평균값 계산
# mode='valid' : 경계값을 제외한 완전한 평균만 반환
def moving_average(data, window_size=10):
   return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# reward_list: 각 에피소드에서 받은 총 보상 목록
# 일반적으로 200에 가까워질수록 학습 성공. 학습이 잘 되면 우상향 곡선을 그림
plt.plot(reward_list, label='Reward per episode')

if len(reward_list) >= 10:
   plt.plot(moving_average(reward_list), label='Moving Avg(10)', color='red')

plt.xlabel('episode') 
plt.ylabel('reward')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 모델 저장 : PyTorch에서는 torch.save() 사용. 일반적으로 state_dict만 저장하는 방식을 많이 사용
torch.save(model.state_dict(), 'cartpole_model.pth')
print('모델 저장 완료')

# 카트 애니메이션 생성 : 저장된 PyTorch 모델을 다시 로드해서 greedy 정책으로 CartPole 실행
# 1. 환경 및 모델 로드
env = gym.make("CartPole-v1", render_mode=None)
loaded_model = DQN(state_dim, num_actions).to(device)
loaded_model.load_state_dict(torch.load('cartpole_model.pth', map_location=device))
loaded_model.eval()
print(loaded_model)

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# 궤적 수집, 1회 실행
flat_states = []
episode_labels = []

state, _ = env.reset()
done = False
ep_num = 0

while not done:
   flat_states.append(state.copy())
   episode_labels.append(ep_num)

   # 현재 상태를 PyTorch Tensor로 변환
   state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

   # 학습된 모델로 Q값 예측 후 가장 큰 Q값의 행동 선택
   with torch.no_grad():
      q_values = loaded_model(state_tensor)
      action = torch.argmax(q_values, dim=1).item()

   next_state, reward, terminated, truncated, _ = env.step(action)
   state = next_state
   done = terminated or truncated

env.close()

frame_count = len(flat_states) 

fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title('Cartpole simulation')
ax.set_xlabel('Cart position')
ax.set_ylabel('height')

# 카트와 막대 표시
cart_width = 0.4
cart_height = 0.2
cart_y = 0.0

cart_rect = Rectangle((0, 0), cart_width, cart_height, color='black')
ax.add_patch(cart_rect)

pole_len = 1.0

line_list = ax.plot([], [], 'r-', lw=4)
pole_line = line_list[0]
episode_text = ax.text( 0.05, 1.4, '', transform=ax.transData, fontsize=12, color='blue' )

# 프레임 업데이트 함수
def updateFunc(frame):
   x = flat_states[frame][0]     # 카트 위치, x축 좌표

   theta = flat_states[frame][2]      # 막대 각도
   ep_num = episode_labels[frame]  # 현재 프레임의 에피소드 번호
   cart_rect.set_xy((x - cart_width / 2, cart_y))  # 카트 위치 이동

   # 막대 시작 좌표
   x_start = x
   y_start = cart_y + cart_height

   # 막대 끝 좌표
   x_end = x_start + pole_len * np.sin(theta)
   y_end = y_start + pole_len * np.cos(theta)

   pole_line.set_data([x_start, x_end], [y_start, y_end])
   episode_text.set_text(f'Episode : {ep_num}')

   return cart_rect, pole_line, episode_text

ani = FuncAnimation(fig, updateFunc, frames=frame_count, interval=50, repeat=False)
plt.close(fig)
display(HTML(ani.to_jshtml()))
