# 강화학습 : 기초 자율 주행 - Q-Learning 기반
# 환경은 1차선 도로, 에이전트(차량)는 좌회전,직진,우회전을 하며 중앙유지를 목표로 함

import numpy as np
import random
import matplotlib.pyplot as plt

# 상태(state) - 11개의 이산적인 구간으로 나눠서 표현
state_space = np.linspace(-1.0, 1.0, 11)
print(state_space)

action_space = [-1, 0, 1]
# Q[state_index, action_index]
q_table = np.zeros((len(state_space), len(action_space)))
# print(q_table)

# 하이퍼 파라미터
alpha = 0.1
gamma = 0.9
epsilon = 0.9
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 500

# 현재 위치를 이산화하여 상태 인덱스로 변환
def get_state_index(position):
    return np.argmin(np.abs(state_space - position))

print(get_state_index(-0.2), ' ', state_space[get_state_index(0.4)])

def get_reward(position):
    return -abs(position)

print(get_reward(0.5))

# 환경 동작 정의
# 현재 위치에서 어떤 행동을 했을 때 '새로운 위치와 보상을 계산'
def stepFunc(position, action):
    position += action * 0.1
    position = np.clip(position, -1.0, 1.0)
    reward = get_reward(position)  # 이동한 후의 위치에 대한 보상 반환
    return position, reward

reward_list = []

# 학습
for ep in range(episodes):
    # 매 에피소드 마다 에이전트는 임의의 위치에서 출발
    position = np.random.uniform(-1.0, 1.0)
    total_reward = 0

    for _ in range(50):
        state_idx = get_state_index(position)

        if random.random() < epsilon:
            action_idx = random.choice([0, 1, 2])
        else:
            action_idx = np.argmax(q_table[state_idx])

        action = action_space[action_idx]
        next_position, reward = stepFunc(position, action)

        next_state_idx = get_state_index(next_position)  # 다음 위치에 대한 이산 상태 인덱스 계산

        # Q값 갱신 : 다음 상태에서 선택 가능한 최대 Q값 계산
        best_next_q = np.max(q_table[next_state_idx])

        # Q 테이블 갱신
        q_table[state_idx, action_idx] += alpha * (reward + gamma * best_next_q - q_table[state_idx, action_idx])

        position = next_position
        total_reward += reward

    reward_list.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % 50 == 0:
        # 성능 지표 계산 및 출력
        initial_avg = np.mean(reward_list[:50])
        final_avg = np.mean(reward_list[-50:])

        total_max_reward = np.max(reward_list)
        total_min_reward = np.min(reward_list)

        recent_max_reward = np.max(reward_list[-50:])
        recent_min_reward = np.min(reward_list[-50:])

        print(f'===== Episode {ep + 1} Performance Summary =====')
        print(f'- initial 50 episodes average reward : {initial_avg:.3f}')
        print(f'- recent 50 episodes average reward  : {final_avg:.3f}')
        print(f'- best reward so far                 : {total_max_reward:.3f}')
        print(f'- worst reward so far                : {total_min_reward:.3f}')
        print(f'- recent 50 max reward               : {recent_max_reward:.3f}')
        print(f'- recent 50 min reward               : {recent_min_reward:.3f}')

        if final_avg > initial_avg:
            print(f'모델이 개선됨 (+{final_avg - initial_avg:.3f})\n')
        else:
            print('크게 개선되지 않음\n')

# 보상 변화 시각화 : reward_list
plt.figure(figsize = (10, 5))
plt.plot(reward_list, label = 'Episode reward')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('episode')
plt.ylabel('total reward')
plt.title('Total reward per episode')
plt.grid(True)
plt.legend()
plt.show()

# 에피소드 50 단위로 평균 보상 시각화
window = 50   # 평균을 낼 구간 크기
avg_rewards = []  # 구간별 평균 보상 저장

for i in range(0, len(reward_list), window):
    chunk = reward_list[i:i + window]
    avg = np.mean(chunk)
    avg_rewards.append(avg)

plt.figure(figsize = (10, 5))
plt.plot(range(0, len(reward_list), window), \
         avg_rewards, marker='o', label = 'average reward(50 episodes)')

plt.xlabel('episode')
plt.ylabel('average reward')
plt.title('average reward every 50 episodes')
plt.grid(True)
plt.legend()
plt.show()

# 위치 히스토그램
position_counts = np.zeros(len(state_space))  # 상태별 방문 수 배열 초기화

# 학습 실행하면서 위치 기록
for _ in range(100):
    position = np.random.uniform(-1.0, 1.0)
    for _ in range(50):
        state_idx = get_state_index(position)
        position_counts[state_idx] += 1   # 해당 상태의 방문 횟수 누적

        if np.random.rand() < epsilon:
            action_idx = np.random.choice(len(action_space))
        else:
            action_idx = np.argmax(q_table[state_idx])

        action = action_space[action_idx]
        position, _ = stepFunc(position, action)

plt.figure(figsize = (10, 3))
plt.bar(state_space, position_counts, width=0.15, align='center')
plt.xlabel('positiojn')
plt.ylabel('visit count')
plt.title('Agent position Frequency after training')
plt.axvline(0, color='red', linestyle='--', label='Center (0.0)')
plt.grid(True)
plt.legend()
plt.show()