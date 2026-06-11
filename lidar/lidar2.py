# pip install gymnasium

import math
import numpy as np
import gymnasium as gym  # 실습환경 제공(현재상태 제공 -> 행동선택 -> 환경이 행동을 반영)
from gymnasium import spaces # 행동 공간과 관측 공간을 정의
import matplotlib.pyplot as plt

# 환경/장애물/라이다 설정
WORLD_W, WORLD_H = 20.0, 15.0
OBSTACLES = [(6.0, 4.0, 0.5),(8.0, 10.0, 1.5),(15.0, 5.0, 1.5)]
