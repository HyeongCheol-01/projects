# 주식 트레이딩 에이전트 (DQN, PyTorch)
import math
import random
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

# 1. 데이터 준비
def load_returns(csv_path: str) -> np.ndarray:
    """
    CSV 파일의 종가(Close)를 읽어 일별 수익률 배열을 반환한다.
    수익률 r_t = (Close_t - Close_{t-1}) / Close_{t-1}
    """
    df = pd.read_csv(csv_path)                                  # CSV 파일 로드
    close = df['Close'].astype(float).values                    # 종가 열을 float 배열로 변환
    ret = np.zeros_like(close, dtype=np.float32)                # 수익률 배열 초기화 (종가와 동일 길이)
    ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)    # 일별 수익률 계산 (0나눗셈 방지용 1e-9)
    return ret                                                  # 첫 원소는 0.0 (기준일)

# 2. 트레이딩 환경
class TradingEnv:
    """
    단순 롱/현금/숏 3방향 포지션 트레이딩 환경.
    관측(obs) : 직전 window일 수익률 + 현재 포지션 → 길이 (window+1) 벡터
    행동(action) : 0=숏(-1), 1=현금(0), 2=롱(+1)
    보상(reward) : 이전 포지션 × 당일 수익률 − 거래 비용
    """

    # 포지션 매핑: 행동 인덱스 → 실제 포지션 값
    ACTION_TO_POS = [-1, 0, 1]

    def __init__(self, returns: np.ndarray, window: int = 20, cost_bps: float = 10.0):
        assert len(returns) > window + 1, '데이터가 너무 짧아요'  # 최소 길이 검증

        self.rets_all = returns.astype(np.float32)  # 전체 수익률 배열 저장
        self.window = window                        # 관측 윈도우 크기 (과거 N일)
        self.cost = cost_bps / 10_000.0   # bps → 소수 비율 변환 (10bp = 0.001)
        self.reset()   # 초기 상태 설정

    @property
    def obs_dim(self) -> int:
        #관측 벡터 차원 = 과거 수익률 window개 + 현재 포지션 1개
        return self.window + 1

    @property
    def n_actions(self) -> int:
        # 행동 공간 크기: 숏(0) / 현금(1) / 롱(2) → 3가지
        return 3

    def reset(self) -> np.ndarray:
        """에피소드를 초기화하고 첫 관측값을 반환한다."""
        self.t = self.window        # 타임스텝을 window 위치부터 시작
        self.pos = 0                # 초기 포지션: 현금
        return self._obs()          # 초기 관측값 반환

    def _obs(self) -> np.ndarray:
        # 현재 시점의 관측 벡터를 생성해 반환한다.
        past = self.rets_all[self.t - self.window: self.t]                 # 직전 window일 수익률 슬라이싱
        return np.concatenate([past, [float(self.pos)]]).astype(np.float32) # [수익률..., 포지션] 결합

    def step(self, action: int):
        # 행동을 받아 환경을 한 스텝 진행. 반환: (다음 관측, 보상, 종료 여부)
        new_pos = self.ACTION_TO_POS[action]                 # 행동 인덱스 → 실제 포지션 값 매핑
        trade_cost = self.cost * abs(new_pos - self.pos)     # 거래 비용 = 비율 × |포지션 변화량|

        # 보상 = 이전 포지션 × 당일 수익률 − 거래 비용
        reward = self.pos * self.rets_all[self.t] - trade_cost

        self.pos = new_pos                   # 포지션 업데이트
        self.t += 1                          # 타임스텝 전진
        done = self.t >= len(self.rets_all)  # 데이터 끝에 도달하면 에피소드 종료

        return self._obs(), float(reward), done


# 3. Q-네트워크 (PyTorch nn.Module)
class QNet(nn.Module):
    """
    2층 완전연결(FC) Q-네트워크를 정의한다.
    입력: 관측 벡터 (obs_dim,), 출력: 각 행동의 Q값 (n_actions,)
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super(QNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        # PyTorch 모델의 순전파 함수. 입력 x를 받아 각 행동에 대한 Q값을 반환한다.
        return self.net(x)


def build_qnet(obs_dim: int, n_actions: int) -> QNet:
    # PyTorch Q-network 객체를 생성하고 device로 이동해 반환한다.
    model = QNet(obs_dim, n_actions)      # QNet 객체 생성
    model = model.to(device)              # CPU 또는 GPU 장치로 이동
    return model


# 4. 리플레이 버퍼
class ReplayBuffer:
    """
    경험 재생 버퍼 (Experience Replay Buffer).
    (s, a, r, s', done) 튜플을 저장하고 미니배치를 무작위 샘플링한다.
    """
    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity) # 최대 capacity개 저장, 초과 시 오래된 것 제거

    def __len__(self) -> int:
        return len(self.buf) # 현재 저장된 경험 수 반환

    def push(self, obs, action, reward, next_obs, done):
        # 새 경험 튜플을 버퍼에 추가한다.
        self.buf.append((obs, action, reward, next_obs, done))   # (s, a, r, s', done) 저장

    def sample(self, n: int):
        """
        버퍼에서 n개의 경험을 무작위로 뽑아 numpy 배열로 반환한다.
        반환: (상태, 행동, 보상, 다음상태, 종료) 각각 (n, ...) 배열
        """
        batch = random.sample(self.buf, n)          # 무작위 미니배치 추출
        s, a, r, ns, d = zip(*batch)                # 각 필드를 분리

        return (
            np.array(s, dtype=np.float32),   # 상태 배열      (n, obs_dim)
            np.array(a, dtype=np.int64),     # 행동 배열      (n,) ← PyTorch gather용 int64
            np.array(r, dtype=np.float32),   # 보상 배열      (n,)
            np.array(ns, dtype=np.float32),  # 다음 상태 배열 (n, obs_dim)
            np.array(d, dtype=np.float32),   # 종료 배열      (n,)
        )


# 5. DQN 에이전트
class DQNAgent:
    """
    Deep Q-Network 에이전트.
    - Q-network      : 현재 상태 가치 추정 및 학습 대상
    - Target-network : 벨만 타깃 계산용 (주기적으로 Q-network 가중치 복사)
    - ε-greedy 탐험/이용 전략 사용
    """

    def __init__(self,
        obs_dim: int,
        n_actions: int,
        lr: float = 3e-4, 
        gamma: float = 0.99,
        batch: int = 128, 
        target_update_freq: int = 100,    # Target network 업데이트 주기 (스텝 수)
    ):
        self.q = build_qnet(obs_dim, n_actions)   # 학습용 Q-network
        self.tgt = build_qnet(obs_dim, n_actions) # 타깃 네트워크

        # PyTorch에서는 Keras의 set_weights 대신 load_state_dict를 사용한다.
        self.tgt.load_state_dict(self.q.state_dict())  # 초기에는 Q-network와 동일한 가중치 복사

        # Target network는 직접 학습하지 않으므로 평가 모드로 설정한다.
        self.tgt.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)  # PyTorch Adam optimizer
        self.loss_fn = nn.SmoothL1Loss()                   # Huber 손실과 동일한 역할

        self.gamma = gamma                 # 할인 계수
        self.batch = batch                 # 미니배치 크기
        self.buf = ReplayBuffer()          # 경험 재생 버퍼

        self.eps = 1.0                     # ε 초기값 (100% 탐험)
        self.eps_decay = 0.9995            # 매 스텝마다 ε에 곱할 감쇠율
        self.eps_min = 0.05                # ε 최솟값 (최소 5% 탐험 유지)
        self.n_actions = n_actions         # 행동 공간 크기

        self.step_count = 0                # 총 업데이트 스텝 카운터
        self.target_update_freq = target_update_freq  # Target network 교체 주기

    def act(self, obs: np.ndarray) -> int:
        """
        ε-greedy 정책으로 행동을 선택한다.
        - ε 확률   : 무작위 행동 (탐험, Exploration)
        - 1-ε 확률 : Q값이 가장 큰 행동 (이용, Exploitation)
        """
        if random.random() < self.eps:                  # 탐험 조건 확인
            return random.randrange(self.n_actions)     # 무작위 행동 인덱스 반환

        # numpy 관측값을 PyTorch Tensor로 변환
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device
        ).unsqueeze(0)     # (obs_dim,) → (1, obs_dim)

        # 행동 선택은 학습이 아니라 추론이므로 torch.no_grad() 사용
        with torch.no_grad():
            qv = self.q(obs_t)   # Q(s, ·) 계산 → (1, n_actions)

        return int(torch.argmax(qv, dim=1).item())  # 최대 Q값 행동 인덱스 반환

    def update(self):
        """
        리플레이 버퍼에서 미니배치를 샘플링해 Q-network 파라미터를 갱신한다.
        손실: Huber( r + γ·max_a' Q_tgt(s',a')·(1-done),  Q(s,a) )
        """
        if len(self.buf) < self.batch:    # 버퍼가 배치 크기보다 작으면 스킵
            return

        s, a, r, ns, d = self.buf.sample(self.batch)    # 미니배치 샘플링

        # numpy 배열을 PyTorch Tensor로 변환하고 device로 이동
        s_t = torch.tensor(s, dtype=torch.float32, device=device)       # 상태 Tensor      (batch, obs_dim)
        a_t = torch.tensor(a, dtype=torch.int64, device=device)         # 행동 Tensor      (batch,)
        r_t = torch.tensor(r, dtype=torch.float32, device=device)       # 보상 Tensor      (batch,)
        ns_t = torch.tensor(ns, dtype=torch.float32, device=device)     # 다음 상태 Tensor (batch, obs_dim)
        d_t = torch.tensor(d, dtype=torch.float32, device=device)       # 종료 Tensor      (batch,)

        # 현재 Q값: Q(s, a)
        # self.q(s_t)의 결과는 (batch, n_actions)
        # gather를 사용해 실제 선택한 행동 a에 해당하는 Q값만 추출한다.
        q_values = self.q(s_t)                                         # 현재 상태의 모든 행동 Q값
        q_sa = q_values.gather(1, a_t.unsqueeze(1)).squeeze(1)          # 선택 행동의 Q값만 추출 (batch,)

        # 벨만 타깃 계산
        # target network는 학습하지 않으므로 torch.no_grad()를 사용한다.
        with torch.no_grad():
            q_next = self.tgt(ns_t).max(dim=1)[0]                      # 다음 상태의 최대 Q값 (batch,)
            y = r_t + (1.0 - d_t) * self.gamma * q_next                # y = r + γ·max Q_tgt(s', a')

        # 손실 계산 : Huber(타깃, 예측)
        loss = self.loss_fn(q_sa, y)

        # 역전파 및 가중치 업데이트
        self.opt.zero_grad()      # 이전 기울기 초기화
        loss.backward()           # 손실에 대한 기울기 계산
        self.opt.step()           # optimizer로 Q-network 파라미터 갱신

        # ε 감쇠 : ε 감쇠 후 최솟값 이하로는 내려가지 않음
        self.eps = max(self.eps * self.eps_decay, self.eps_min)

        # Target network 주기적 동기화
        # PyTorch에서는 Keras의 set_weights 대신 load_state_dict를 사용한다.
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.tgt.load_state_dict(self.q.state_dict())              # Q-network 가중치를 타깃에 복사


# 학습 루프
def train_go(
    csv_path: str = 'prices.csv',
    window: int = 20,
    cost_bps: float = 10.0,
    episodes: int = 5,
):
    """
    전체 학습 파이프라인.
    1) 데이터 로드 → 2) 환경/에이전트 초기화 → 3) 에피소드 반복 학습 → 4) 최종 PnL 및 샤프 비율 출력
    """
    rets = load_returns(csv_path)                  # 일별 수익률 배열 로드
    env = TradingEnv(rets, window, cost_bps)       # 트레이딩 환경 초기화
    agent = DQNAgent(env.obs_dim, env.n_actions)   # DQN 에이전트 초기화
    equity = []                                    # 누적 PnL 기록 리스트 (equity curve)

    for ep in range(1, episodes + 1):              # 에피소드 반복
        obs = env.reset()                          # 환경 초기화 및 초기 관측값 획득
        done = False                               # 에피소드 종료 플래그
        ep_pnl = 0.0                               # 에피소드 누적 PnL 초기화

        while not done:                            # 에피소드가 끝날 때까지 반복
            act = agent.act(obs)                   # ε-greedy 행동 선택
            nobs, r, done = env.step(act)          # 환경 한 스텝 진행 → (다음관측, 보상, 종료)

            agent.buf.push(
                obs,
                act,
                r,
                nobs,
                float(done)
            )   # 경험 튜플을 리플레이 버퍼에 저장

            agent.update()          # 미니배치 샘플링 후 Q-network 학습

            obs = nobs              # 다음 관측값으로 전진
            ep_pnl += r             # 누적 PnL 갱신 (스텝 보상의 합)
            equity.append(ep_pnl)   # 전체 equity curve에 추가

        print(f'ep:{ep:3d}/{episodes}  PnL={ep_pnl:.4f}  ε={agent.eps:.3f}')

    # 결과 요약
    equity = np.array(equity)                  # 리스트 → numpy 배열로 변환
    daily_ret = np.diff(equity, prepend=0.0)   # 일별 PnL 증분(≈ 스텝별 보상) 계산

    # 샤프 비율 = 평균(일별수익) / 표준편차(일별수익) × √252
    # 252는 일반적으로 사용하는 연간 거래일 수
    sharpe = daily_ret.mean() / (daily_ret.std() + 1e-9) * math.sqrt(252)
    print('\n요약 결과')
    print(f'Final PnL   : {equity[-1]:.5f}')
    print(f'Sharpe Ratio: {sharpe:.3f}')


if __name__ == '__main__':
    train_go(
        csv_path='prices.csv', 
        window=20,       # 관측 윈도우 크기 (20 거래일)
        cost_bps=10.0,   # 거래 비용 (10 basis points = 0.1%)
        episodes=5
    )