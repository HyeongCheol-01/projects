# 강화학습
# 에이전트의 라벨이 없이 직접 행동해보고 보상을 받아 어떤 행동이 좋은지 점점 학습해나감
# 현재상태 -> 행동선택 -> 보상확인 -> 다음행동 개선 순으로 진행함

# 강화학습은 정답이 아니라 보상으로 배운다
# 순서
# 1) 상태는 현재 위치다
# 2) 행동은 위/아래/좌/우
# 3) 보상은 행동 결과에 대한 점수다
# 4) Q-table은 상태별 행동 점수표다
# 5) Q-learning은 Q-table을 조금씩 갱신해 나간다
# 6) epsilon-greedy는 탐험과 이용을 조절한다.
# 7) 학습 후 Q-table에서 가장 큰 행동을 선택하면 이것이 정책(policy)이 된다.
# 8) 이 구조가 Gymnasium의 step() 구조와 연결된다.

import numpy as np
import random

np.random.seed(42)
random.seed(42)

# GridWorld 환경 설정
ROWS = 3
COLS = 4

START = (0,0)  # 에이전트의 행동 시작위치
GOAL = (2,3)   # 목표
TRAP = (1,1)   # 함정

actions = {
    0:(-1, 0), # 상
    1:(1, 0),  # 하
    2:(0, -1), # 좌
    3:(0, 1),  # 우
}

action_names = {
    0:'위',
    1:'아래',
    2:'왼쪽',
    3:'오른쪽'
}

num_states = ROWS * COLS    # 상태수는 12
num_actions = len(actions)  # 가능한 행동수 4

Q = np.zeros((num_states, num_actions))
# print(Q)

# 하이퍼 파라미터
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

episodes = 1000
max_steps = 30   # 하나의 에피소드 내에서 최대 이동 가능 횟수

# 위치 정보를 Q-table에서 사용할 상태 번호로 변환 (0,0)->0 ,... (2,3)->11
def pos_to_state(pos):
    row, col = pos
    return row * COLS + col
print(pos_to_state((2, 3)))

def state_to_pos(state):  # 상태 번호를 위치정보로 변환
    row = state // COLS
    col = state % COLS
    return (row, col)
print(state_to_pos(11))

# 환경 이동 함수
def step(pos, action):  # 현재 위치에서 행동을 실행하고 결과를 반환하는 함수
    row, col = pos
    dr, dc = actions[action]  # 선택한 행동에 해당하는 행변화량, 열변화량 얻기

    next_row = row + dr   # 이동 후의 행 위치 계산
    next_col = col + dc   # 이동 후의 열 위치 계산

    # GridWorld 경계밖으로 나가면 제자리, 패널티 부여
    if next_row < 0 or next_row >= ROWS or next_col < 0 or next_col >= COLS:
        next_pos = pos
        reward = -2
        done = False    # 벽에 부딪혔다고 에피소드가 끝나지는 않음
        return next_pos, reward, done

    next_pos = (next_row, next_col)

    if next_pos == GOAL:
        reward = 10
        done = True
    elif next_pos == TRAP:
        reward = -10
        done = True
    else:
        reward = -1   # 일반 이동할 때 마다 -1 패널티, 짧은 경로를 유도
        done = False

    return next_pos, reward, done

print(step((0,0), 3))  # ((0, 1), -1, False)
print(step((1,0), 3))  # ((1, 1), -10, True)
print(step((2,2), 3))  # ((2, 3), 10, True)

# epsilon-greedy 행동 선택 : 탐험 또는 이용
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)   # 탐험
    else:
        return np.argmax(Q[state])   # 이용 : 현재 상태에서 Q값이 가장 큰 행동 선택

print('선택된 행동은 ', choose_action(0, 1.0))


# Q-learning 학습
for episode in range(episodes):
    pos = START
    state = pos_to_state(pos)   # (0,0) => 0

    # 하나의 에피소드 안에서 최대 행동 횟수
    for step_count in range(max_steps):
        # 행동 선택
        action = choose_action(state, epsilon)
        # print('action : ', action)

        # 행동 실행
        next_pos, reward, done = step(pos, action)
        # print(next_pos, reward, done)
        next_state = pos_to_state(next_pos)  # 다음 위치로 상태번호로 변환

        # 현재 Q값(벨만 방정식에 적용)
        old_q = Q[state][action]    # 현재 상태(state)에서 선택한 action의 기존 Q값
        # print('old_q : ', old_q)

        # Q-learning target을 계산(이번 경험으로 계산한 목표 Q값)
        if done:
            target = reward    # 함정 또는 목표 도착시 미래 Q값을 보지않고 현재 보상만 사용
        else:
            next_max = np.max(Q[next_state])   # 다음 상태에서 가능한 행동 중 가장 큰 Q값
            target = reward + gamma * next_max

        # Q-learning update
        td_error = target - old_q   # 목표값과 기존 Q값의 차이
        Q[state][action] = old_q + alpha * td_error  # 현재 상태의 가치를 보상과 다음상태의 가치로 표현한 식

        # 상태 이동
        pos = next_pos      # 현재 위치를 다음 위치로 갱신
        state = next_state  # 현재 상태번호를 다음 상태번호로 갱신

        if done:
            break

    # epsilon 감소
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    # print('epsilon : ', epsilon)

# 학습 결과 출력
print('학습된 Q-table : ', np.round(Q, 2))

print('\n각 상태에서 가장 좋은 행동 출력')
for state in range(num_states):
    pos = state_to_pos(state)

    if pos == GOAL:
        print(f'상태 {state} {pos} : 목표지점')
        continue

    if pos == TRAP:
        print(f'상태 {state} {pos} : 함정')
        continue

    best_action = np.argmax(Q[state])
    print(f'상태 {state} {pos} : {action_names[best_action]}')

print('학습된 정책으로 실제 이동 경로 확인')
pos = START
path = [pos]    # 시작위치부터 이동경로 저장

for i in range(20):
    state = pos_to_state(pos)
    action = np.argmax(Q[state])

    next_pos, reward, done = step(pos, action)
    path.append(next_pos)   # 이동한 다음위치를  경로에 추가

    pos = next_pos

    if done:
        break

print('이동 경로 : ', path)

arrow = {
    0:'⇡',
    1:'⇣',
    2:'⇠',
    3:'⇢'
}

print('이동 경로 화살표로 출력')
for r in range(ROWS):
    row_text = ''
    for c in range(COLS):
        pos = (r, c)

        if pos == START:
            row_text += ' S '
        elif pos == GOAL:
            row_text += ' G '
        elif pos == TRAP:
            row_text += ' X '
        else:
            state = pos_to_state(pos)
            best_action = np.argmax(Q[state])
            row_text += f' {arrow[best_action]} '

    print(row_text)