# 강화학습으로 cartPole 버티기
# 목표 : 200번의 step(시간) 단계 동안 Pole이 넘어지지 않고 유지하기
# 종료 : Pole이 (수직으로 부터 +-12 도 이상으로) 기울거나 카트가 환경 밖으로 벗어나거나
#        최대 시간 단계인 200에 도달한 경우

# !pip install gymnasium[classic-control]

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

# 환경 생성
env = gym.make("CartPole-v1")

print(env.observation_space)
# Box(
#   -4.8 ~ 4.8        : cart 위치
#   -inf ~ inf        : cart 속도
#   -0.418 ~ 0.418    : pole 각도
#   -inf ~ inf        : pole 각속도
# )

print(env.action_space.n)
# 2
# 0 : 왼쪽으로 힘을 가함
# 1 : 오른쪽으로 힘을 가함


# Q-table(observation_space) 공간 범위 인위적 실험공간 생성
# CartPole의 관측값은 연속형 실수이므로 Q-table에 바로 사용할 수 없음
# 따라서 관측값의 범위를 임의로 정하고 구간으로 나누어 이산화함
obs_space_low = np.array([-2.4, -3.0, -0.5, -2.0])
obs_space_high = np.array([2.4, 3.0, 0.5, 2.0])


# 상태공간 이산화 수준 결정
# 각 상태 차원을 몇 개의 구간으로 나눌지를 결정
# cart 위치        : 6개 구간
# cart 속도        : 12개 구간
# pole 각도        : 6개 구간
# pole 각속도      : 12개 구간
state_bins = [6, 12, 6, 12]


# q_table 초기화
# 상태는 4차원 이산 인덱스이고, 각 상태마다 가능한 행동은 2개
q_table = np.zeros(state_bins + [env.action_space.n])

# print(q_table)
# print(q_table.shape)     # (6, 12, 6, 12, 2)

print(6 * 12 * 6 * 12 * 2)
# 10368
# 전체 Q값 개수 = 상태 조합 수 * 행동 수


# 연속적인 값(실수)을 몇 개의 고정된 구간(bin)으로 나눠 이산적인(정수) 인덱스로 변환
def discretize_state(state):
    # state는 CartPole 환경에서 반환하는 관측값
    # state = [카트위치, 카트속도, 막대각도, 막대각속도]

    # state의 각 요소가 obs_space_low ~ obs_space_high 사이에서
    # 어느 정도 위치에 있는지 비율로 변환
    ratios = (state - obs_space_low) / (obs_space_high - obs_space_low)

    # 비율값에 구간 개수를 곱해서 이산 인덱스로 변환
    discrete = (ratios * state_bins).astype(int)

    # 인덱스가 범위를 벗어나지 않도록 clip 처리
    # 예: state_bins가 6이면 가능한 인덱스는 0 ~ 5
    return tuple(np.clip(discrete, 0, np.array(state_bins) - 1))


# clip : 제한된 범위 내의 수치 얻기
# a = np.array([-2, 0, 3, 7, 10])
# result = np.clip(a, 0, 5)
# print(result)   # [0 0 3 5 5]

# ex_state = np.array([1.0, 0.5, 0.5, -1.0])
# dis_index = discretize_state(ex_state)
# print('Q-Table 인덱스 : ', dis_index)


# 하이퍼 파라미터 설정
alpha = 0.1          # 학습률
gamma = 0.99         # 할인율
epsilon = 1.0        # 탐험 확률
epsilon_decay = 0.999
epsilon_min = 0.05
episodes = 1000


reward_list = []     # 에피소드에서 받은 총보상 기록
trajectories = []    # 위치정보(궤적)를 저장
best_reward = 0      # 지금까지 달성한 최고의 총 보상 누적


# 에피소드 만큼 반복
for ep in range(episodes):
    obs, _ = env.reset()

    # 연속형 관측값을 Q-table에서 사용할 수 있는 이산 상태로 변환
    state = discretize_state(obs)

    total_reward = 0
    trajectory = []

    # 하나의 에피소드 안에서 최대 200번 행동
    for step in range(200):
        # 행동선택 -> 새로운상태, 보상, 종료여부 반환 -> Q-table 갱신 -> 상태 갱신

        # epsilon-greedy 방식으로 행동 선택
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 무작위 행동 선택, 탐험
        else:
            action = np.argmax(q_table[state])  # Q값이 가장 큰 행동 선택, 이용

        # 주어진 action으로 환경 내에서 실행
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # terminated : pole이 넘어지거나 카트가 범위를 벗어나 종료
        # truncated  : 최대 step 수에 도달해서 종료
        done = terminated or truncated

        # 다음 관측값도 Q-table에서 사용할 수 있도록 이산 상태로 변환
        next_state = discretize_state(next_obs)

        # Q-learning target 계산
        # done이 True이면 에피소드가 끝난 상태이므로 미래 Q값을 보지 않음
        if done:
            target = reward
        else:
            best_next_q = np.max(q_table[next_state])  # 다음 상태에서 가능한 행동 중 가장 큰 Q값
            target = reward + gamma * best_next_q

        # Q-table 갱신 : MDP가 적용된 벨만 방정식 기반 업데이트
        # 기존 Q값
        old_q = q_table[state + (action,)]

        # TD Error = target - old_q
        td_error = target - old_q

        # Q값 갱신
        q_table[state + (action,)] = old_q + alpha * td_error

        # 상태 갱신
        state = next_state
        obs = next_obs  # 원래 상태값도 갱신, 시각화 용도

        # 보상 누적
        total_reward += reward

        # 애니메이션을 위해 관측값 저장
        trajectory.append(obs.copy())

        # 종료 조건이면 현재 에피소드 종료
        if done:
            break

    # 에피소드별 총 보상 저장
    reward_list.append(total_reward)

    # 향상된 보상 출력
    if total_reward > best_reward:
        best_reward = total_reward
        print(f'Episode : {ep} : Reward improved to {total_reward}')

    # 10 에피소드 마다 위치정보(궤적) 저장
    if ep % 10 == 0:
        trajectories.append(trajectory)

    # epsilon 감소
    if epsilon > epsilon_min:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)


# 보상 그래프 시각화
plt.figure(figsize=(10, 4))
plt.plot(reward_list, label='Episode reward')
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 저장된 궤적정보로 애니메이션 실행
flat_states = []  # 여러 에피소드의 궤적을 한 줄로 펼쳐 저장

episode_labels = []
episode_numbers = list(range(0, episodes, 10))

# 시각화 목적의 데이터 평탄화 라벨링
for i, traj in enumerate(trajectories):
    flat_states.extend(traj)
    episode_labels.extend([episode_numbers[i]] * len(traj))


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

episode_text = ax.text(
    0.05,
    1.4,
    '',
    transform=ax.transData,
    fontsize=12,
    color='blue'
)


def updateFunc(frame):
    # 현재 프레임의 카트 위치와 막대 각도 가져오기
    x = flat_states[frame][0]       # 카트 위치
    theta = flat_states[frame][2]   # 막대 각도
    ep_num = episode_labels[frame]  # 현재 프레임의 에피소드 번호

    # 카트 위치 갱신
    cart_rect.set_xy((x - cart_width / 2, cart_y))

    # 막대 시작 좌표
    x_start = x
    y_start = cart_y + cart_height

    # 막대 끝 좌표
    x_end = x_start + pole_len * np.sin(theta)
    y_end = y_start + pole_len * np.cos(theta)

    # 막대 선 갱신
    pole_line.set_data([x_start, x_end], [y_start, y_end])

    # 에피소드 번호 표시
    episode_text.set_text(f'Episode : {ep_num}')

    return cart_rect, pole_line, episode_text


# 애니메이션 생성
ani = FuncAnimation(
    fig,
    updateFunc,
    frames=frame_count,
    interval=50,
    repeat=False
)

plt.close(fig)

# ani.to_jshtml() : 애니메이션을 HTML + JavaScript로 변환
display(HTML(ani.to_jshtml()))


# 일반 파이썬 파일(.py)에서는 display가 바로 동작하지 않을 수 있음
# Jupyter Notebook 또는 Colab 환경에서 실행하는 것을 권장