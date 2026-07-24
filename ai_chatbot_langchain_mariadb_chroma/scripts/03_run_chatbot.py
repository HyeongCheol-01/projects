"""
03_run_chatbot.py

교육기관 AI 상담 챗봇을 CLI로 실행합니다.

실행:
python scripts/03_run_chatbot.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.chatbot import answer_question


EXAMPLE_QUESTIONS = [
    "홍길동 교육생은 현재 수료 가능성이 있나요?",
    "김민수 교육생은 출석률 때문에 문제가 있나요?",
    "이영희 교육생에게 취업지원은 어떻게 안내하면 좋나요?",
]


def main() -> None:
    print("교육기관 AI 상담 챗봇")
    print("종료하려면 exit 입력")
    print("예시 질문:")
    for q in EXAMPLE_QUESTIONS:
        print("-", q)

    while True:
        question = input("\n질문 입력 > ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("종료합니다.")
            break

        if not question:
            continue

        answer = answer_question(question)
        print("\n답변:")
        print(answer)


if __name__ == "__main__":
    main()
