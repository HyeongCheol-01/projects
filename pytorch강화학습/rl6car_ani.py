# 이전 코드 + 궤적 에니메이션 시각화
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

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

def get_reward(position):
    return -abs(position)


# 환경 동작 정의
# 현재 위치에서 어떤 행동을 했을 때 '새로운 위치와 보상을 계산'
def stepFunc(position, action):
    position += action * 0.1
    position = np.clip(position, -1.0, 1.0)
    reward = get_reward(position)  # 이동한 후의 위치에 대한 보상 반환
    return position, reward

reward_list = []
trajectories = []   # 에이전트 궤적 저장

# 학습
for ep in range(episodes):
    # 매 에피소드 마다 에이전트는 임의의 위치에서 출발
    position = np.random.uniform(-1.0, 1.0)
    total_reward = 0
    trajectory = []   # 각 에피소드 마다 이동 기록

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
        trajectory.append(position)

    reward_list.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % 10 == 0:    # 10 에피소드 마다 궤적 저장
        trajectories.append(trajectory)

    epsilon = max(0.05, epsilon * epsilon_decay)

# 궤적 평탄화
flat_positions = [pos for traj in trajectories for pos in traj]
frame_count = len(flat_positions)
print('flat_positions : ', flat_positions, '\n',frame_count)

from matplotlib import transforms
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-1.1, 1.0)
ax.set_ylim(-0.1, 1.05)
ax.set_title('agent simulation (Q-learning)')
ax.set_xlabel('Position on road(-1.0 ~ 1.0)')
ax.axvline(0, color='gray', linestyle='--', label='Center')
point, = ax.plot([],[], 'bo', markersize=8)
ax.legend()
# 에피소드 번호 화면에 표시
episode_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)

# 에니메이션 정의 : 프레임 수 만큼 호출되며, 각 프레임에서 파란점을 갱신
def updateFunc(frame):
    x = flat_positions[frame]   # 현재 프레임의 에이전트 위치
    y = (frame % 50) / 50  # 정규화된 스텝위치(y축)
    point.set_data([x],[y])
    episode_num = frame // 50 + 1
    episode_text.set_text(f'Episode : {episode_num}')

ani = FuncAnimation(fig, updateFunc, frames=frame_count, interval=100, repeat=False)
plt.close(fig)

HTML(ani.to_jshtml())

ani.save('ani_car.mp4', writer='ffmpeg', fps=10)
from google.colab import files
files.download('ani_car.mp4')

# Q-table 출력
print('\nfinal Q-table (rows=states, columns=actions[-1, 0, 1])\n')
header = f"{'state':>7} | {'-1':>7} {'0':>7} {'1':>7}"
print(header)
print('-' * len(header))
for i, state_val in enumerate(state_space):
    q_vals = q_table[i]
    print(f'{state_val:7.2f} | {q_vals[0]:7.3f} {q_vals[1]:7.3f} {q_vals[2]:7.3f}')