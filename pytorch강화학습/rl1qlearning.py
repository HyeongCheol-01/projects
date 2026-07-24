# Q-learning의 구조를 이해하기 - 벨만 방정식 기반의 근사학습
# Q-learning에서 에이전트는 가장 Q값이 높은 행동을 선택(그리디한 행동)한다.

# 현재 위치(state)에서 어떤 행동(action)을 취할 것인가? 왼쪽, 오른쪽 ...

import numpy as np
import random

# 상태 공간 : 에이전트가 있을 수 있는 위치
state_space = [0, 1, 2, 3, 4]

# 행동 공간 : -1은 왼쪽, 1은 오른쪽 이동을 의미
action_space = [-1, 1]

# Q-table : (행 : 상태, 열 : 행동)
# Q[state][action] <== 특정 상태에서 특정 행동을 했을 때의 가치

Q = np.zeros((len(state_space), len(action_space)))
print(Q)

# 하이퍼 파라미터 설정
alpha = 0.1     # 학습률(새롭게 배운 값을 기존 Q값에 얼마나 반영할지 결정)
gamma = 0.9    # 할인률(discount factor). 미래 보상을 얼마나 중요하게 볼 것인지를 결정
epsilon = 1.0   # 탐험 확률
epsilon_decay = 0.99  # epsilon 감소율
epsilon_min = 0.1  # 탐험의 최소 확률
episodes = 500  # 전체 학습 횟수

# 보상 함수 : state가 4이면 목표에 도달
def get_reward(state):
    return 10 if state == 4 else 0

# 학습 시작 : episode는 하나의 학습 시도
for episode in range(episodes):
    state = 0  # 매 에피소드 마다 0번 위치에서 시작

    for step in range(20):   # step : 한번의 이동. 한 에피소드 안에서 행동은 20번으로 제한
        # 행동 선택
        if random.random() < epsilon:
            action_index = random.randint(0, 1)   # 탐험(Exploration) - 랜덤함
        else:
            action_index = np.argmax(Q[state])    # 이용(Exploitation) - 탐욕적 행동

        action = action_space[action_index]   # action_index를 실제 행동값(-1 or 1)으로 변환
        # print('action : ', action)

        # 다음 상태 계산
        next_state = state + action   # state=2, action=1 ==> next_state:3

        # 상태공간을 유지
        if next_state < 0 or next_state > 4:
            next_state = state

        # 보상 받기
        reward = get_reward(next_state)

        # Q값 갱신
        old_q = Q[state][action_index]

        next_max = np.max(Q[next_state])   # 다음 상태에서 가장 좋은 행동을 했을 때 기대되는 가치

        # Q-learning 갱신 식 - 벨만 방정식(off-policy 방식의 수식)
        Q[state][action_index] = old_q + alpha * (reward + gamma * next_max - old_q)

        # 상태 이동
        state = next_state  # 다음 상태를 현재 상태로 변경

        if reward == 10:
            break

    # epsilon 감소
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    # print('epsilon : ', epsilon)

print(Q)