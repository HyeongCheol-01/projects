# 강화학습으로 미니카 학습 모델 생성 + 에니메이션 <== DQN
# MiniCar : (x, y, theta, 속도)의 상태를 가짐, action은 3가지(좌회전, 직진, 우회전)
# 보상은 중앙에서 멀어질수록 불이익. 목표 도착시 보상 +10, 도로 이탈 시 보상 -10
# 현재 정책학습은 main-model이 담당, 안정적인 목표값 제공은 target-model이 담당
# 메모리 버퍼에 경험을 저장하고 샘플링하여 학습(경험 재사용)

# !pip install gymnasium
# !pip install torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from collections import deque
import random
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim


# PyTorch 실행 장치 설정
# GPU가 있으면 cuda를 사용하고, 없으면 cpu를 사용한다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치 : ", device)


# 사용자 정의 환경
class MiniCarEnv(gym.Env):
    def __init__(self) -> None:
        super(MiniCarEnv, self).__init__()

        # 상태 공간 : x, y, theta, v
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0], dtype=np.float32),
            high=np.array([100, 100, np.pi, 5], dtype=np.float32)
        )

        # 행동 공간 : 0 - 좌회전, 1 - 직진, 2 - 우회전
        self.action_space = spaces.Discrete(3)

        # 환경 초기화
        self.reset()

    def reset(self, seed=None, options=None):
        # 시작 위치 x좌표
        x = 10.0

        # 시작 위치 y좌표
        y = 50.0

        # 시작 각도
        theta = 0.0

        # 시작 속도
        v = 1.0

        # 상태를 numpy 배열로 저장
        self.state = np.array([x, y, theta, v], dtype=np.float32)

        # Gymnasium 형식에 맞게 state와 info를 반환
        return self.state, {}

    def step(self, action):
        # 현재 상태값을 꺼낸다.
        x, y, theta, v = self.state.astype(np.float32)

        # 조향 갱신
        steer_step = 0.1

        if action == 0:  # 왼쪽
            theta -= steer_step
        elif action == 2:  # 오른쪽
            theta += steer_step

        # 조향 감쇠
        # 너무 급격하게 방향이 계속 꺾이지 않도록 자연스럽게 직진 방향으로 돌아오게 한다.
        theta *= 0.98

        # 각도 wrapping
        # 각도를 항상 [-π, +π](-180도 ~ 180도) 범위 안에 맞추기 - 각도 정규화
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # 이동 : 2차원 좌표상에서
        # x축 이동량 : x = x + v * cos(theta)
        # y축 이동량 : y = y + v * sin(theta)
        n = np.random.normal(0, 0.02, size=2)  # 강화학습에 반영할 노이즈(센서오차, 바람, 미끌어짐...)

        # 이전 x좌표 저장
        x_prev = x

        # 새로운 x좌표 계산
        x = x + v * np.cos(theta) + n[0]

        # 새로운 y좌표 계산
        y = y + v * np.sin(theta) + n[1]

        # 새로운 상태 저장
        self.state = np.array([x, y, theta, v], dtype=np.float32)

        # 보상 설계
        # 중앙선에서 벗어날 경우 감점 고려
        center_penalty = -0.05 * abs(y - 50.0)

        # 진행 보상 : 직진 한 만큼 보상
        # 뒤로 가거나 옆으로만 움직이는 행동을 억제하기 위해 x 증가량을 보상으로 사용
        process = max(0.0, x - x_prev) * 0.8

        # 생존 보상 - 너무 크지 않게
        alive = 0.2

        # 최종 보상
        reward = alive + center_penalty + process

        # 종료 조건
        terminated = False
        truncated = False

        if x > 90 and 0 <= y <= 100:  # 목표 달성인 경우
            reward += 50.0
            terminated = True
        elif not (0 <= x <= 100 and 0 <= y <= 100):  # 도로 밖으로 나간 경우
            reward -= 15.0
            terminated = True

        # Gymnasium step 반환 형식
        return self.state, float(reward), terminated, truncated, {}


# DQN 모델 생성
# Keras의 Sequential 대신 PyTorch에서는 nn.Module을 상속해서 신경망을 정의한다.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        # 신경망 계층 정의
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),   # 입력층 : 상태 4개 입력
            nn.ReLU(),                  # 활성화 함수
            nn.Linear(64, 64),          # 은닉층
            nn.ReLU(),                  # 활성화 함수
            nn.Linear(64, output_dim)   # 출력층 : 행동 개수만큼 Q값 출력
        )

    def forward(self, x):
        # 입력 x를 신경망에 통과시켜 Q값을 반환한다.
        return self.network(x)


# DQN 모델 생성 함수
def create_dqn(input_dim, output_dim):
    # DQN 모델 객체 생성
    model = DQN(input_dim, output_dim)

    # 모델을 GPU 또는 CPU로 이동
    model = model.to(device)

    return model


# 실험 설정
episodes = 100
runs = 1  # 전체 학습을 몇 번 반복할 지 설정  2 ~ 3 을 권장

all_run_rewards = []     # 에피소드별 총 보상 저장
all_run_deviations = []  # 도로 중앙(y=50)으로부터 평균 편차 저장
final_trajectories = []  # 에피소드별 이동 경로를 저장


# 학습 - 강화학습의 특성(무작위)상 여러번 실행해 평균 성능을 확인하길 권장
for run in range(runs):
    env = MiniCarEnv()

    state_dim = int(env.observation_space.shape[0])  # 관측 차원 수 : 4
    num_actions = int(env.action_space.n)            # 가능한 행동 수 : 3

    model = create_dqn(state_dim, num_actions)        # main model (Main Network)
    target_model = create_dqn(state_dim, num_actions) # target model (Target Network)

    # PyTorch에서는 Keras의 set_weights 대신 load_state_dict를 사용한다.
    target_model.load_state_dict(model.state_dict())

    # target_model은 학습용이 아니라 목표 Q값 계산용이므로 평가 모드로 둔다.
    target_model.eval()

    # model : 현재 Q값을 예측하고 학습하는 신경망
    # target_model : 비교 기준이 되는 고정된 Q값을 제공(학습 목표값 계산에 사용)

    # PyTorch 최적화 도구와 손실 함수 설정
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer 사용
    loss_fn = nn.MSELoss()  # Keras의 loss='mse'와 같은 평균제곱오차 사용

    # 하이퍼 파라미터 설정
    gamma = 0.99
    epsilon = 0.6
    epsilon_decay = 0.997
    epsilon_min = 0.02

    batch_size = 32  # 경험을 32개씩 샘플링해서 학습
    memory = deque(maxlen=2000)  # ReplayBuffer 크기

    reward_history = []     # 에피소드마다 받은 총보상을 저장
    deviation_history = []  # 주행 중 y값이 도로중심에서 얼마나 벗어났는지 평균편차 저장
    run_trajectories = []   # run별 이동 경로 저장

    CENTER_Y = 50.0

    for ep in range(episodes):
        state, _ = env.reset()

        print(f'Run {run} 에피소드 시작 : ep = {ep}, 초기 상태={state}')

        total_reward = 0
        trajectory = []
        done = False

        for step in range(200):
            # 현재 상태를 PyTorch Tensor로 변환
            # Keras에서는 np.reshape(state, [1, state_dim]) 형태를 사용했지만,
            # PyTorch에서는 torch.tensor로 변환한 뒤 device로 이동한다.
            state_input = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)  # 1차원 -> 2차원

            # 탐험, 이용 결정
            if np.random.rand() < epsilon:
                # 탐험 : 무작위 행동 선택
                action = np.random.choice(num_actions)
            else:
                # 이용 : 현재 main model이 예측한 Q값 중 가장 큰 행동 선택
                # 추론만 하므로 torch.no_grad()를 사용해 기울기 계산을 막는다.
                with torch.no_grad():
                    q_values = model(state_input)
                    action = torch.argmax(q_values[0]).item()

            # 선택한 행동을 환경에 적용
            next_state, reward, done, _, _ = env.step(action)

            # 경험 저장 : 현재상태, 행동, 보상, 다음상태, 종료여부
            memory.append((state, action, reward, next_state, done))

            # 경로 및 보상 누적
            trajectory.append(state)
            total_reward += reward

            # 다음 상태로 이동
            state = next_state

            if done:
                break

        # 하나의 에피소드가 끝난 뒤, 그 동안의 성과(보상, 이동경로, 편차 등)를 기록
        reward_history.append(total_reward)

        # 이동경로 배열로 변환
        traj = np.array(trajectory)

        # 평균 편차 계산
        mean_deviation = np.mean(np.abs(traj[:, 1] - CENTER_Y))

        # 평균 편차 저장
        deviation_history.append(mean_deviation)

        # 이동 경로 저장
        run_trajectories.append((ep, trajectory.copy()))

        print(
            f'Run {run} 에피소드 종료 : ep={ep}, '
            f'총보상={total_reward:.2f}, 편차={mean_deviation:.2f}, 종료상태={state}'
        )

        # 메모리에서 무작위 샘플링 후 Q-network 갱신
        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)

            states = []   # 학습에 사용할 상태 누적용
            targets = []  # 학습에 사용할 목표 Q값 누적용

            # 각 샘플에 대해 Q값 계산
            for s, a, r, s_next, d in minibatch:
                # 현재 상태를 Tensor로 변환
                s_input = torch.tensor(
                    s,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                # 다음 상태를 Tensor로 변환
                s_next_input = torch.tensor(
                    s_next,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                # 현재 상태 s에서 각 행동의 Q값을 예측
                # target 배열은 학습 정답 역할을 하므로 detach로 계산 그래프에서 분리한다.
                with torch.no_grad():
                    target = model(s_input)[0].detach().clone()

                if d:
                    # 종료된 상태면 단순히 보상이 정답
                    target[a] = r
                else:
                    # target_model로 다음 상태의 Q값을 예측
                    with torch.no_grad():
                        t_next = target_model(s_next_input)[0]

                    # DQN 목표값 계산
                    # Q(s, a) = r + gamma * max Q_target(s_next, a_next)
                    target[a] = r + gamma * torch.max(t_next)

                # 학습 입력/타겟 누적
                states.append(s)
                targets.append(target.cpu().numpy())

            # numpy 배열을 PyTorch Tensor로 변환
            states_tensor = torch.tensor(
                np.array(states),
                dtype=torch.float32,
                device=device
            )

            targets_tensor = torch.tensor(
                np.array(targets),
                dtype=torch.float32,
                device=device
            )

            # Q-Network 학습
            # PyTorch에서는 model.fit 대신 아래 과정을 직접 수행한다.
            model.train()  # 학습 모드

            optimizer.zero_grad()  # 이전 단계의 기울기 초기화

            predictions = model(states_tensor)  # 현재 model이 예측한 Q값

            loss = loss_fn(predictions, targets_tensor)  # 예측값과 목표값의 차이를 손실로 계산

            loss.backward()  # 역전파 수행

            optimizer.step()  # 가중치 갱신

        # epsilon 감소
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # target network 갱신
        if ep % 10 == 0:
            # PyTorch에서는 set_weights 대신 load_state_dict 사용
            target_model.load_state_dict(model.state_dict())

    # 하나의 run 안에서 각 에피소드별 보상 저장
    all_run_rewards.append(reward_history)

    # 하나의 run 안에서 각 에피소드별 평균 편차 저장
    all_run_deviations.append(deviation_history)

    if run == runs - 1:
        # 마지막 run의 경로 데이터 저장
        final_trajectories = run_trajectories


# 참고 : 각도 wrapping ----------------
def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


angle = np.deg2rad(370)   # 370도는 wrapping 후 10도
angle = np.deg2rad(-190)  # -190도는 wrapping 후 170도
angle = np.deg2rad(120)   # 120도는 그대로 120도

print(np.rad2deg(wrap_angle(angle)))