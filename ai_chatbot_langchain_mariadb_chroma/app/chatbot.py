"""
교육기관 AI 상담 챗봇 핵심 로직

전체 흐름:
1. 사용자 질문 분석
2. MariaDB에서 교육생 정형 데이터 조회
3. Chroma VectorDB에서 관련 문서 검색
4. LangChain Prompt + LLM으로 최종 답변 생성
"""

from langchain_openai import ChatOpenAI

from app.config import settings
from app.db import (
    fetch_all_student_names,
    fetch_student_report,
    format_student_report,
)
from app.vector_store import search_policy_documents
from app.prompts import EDU_COUNSELING_PROMPT


def classify_intent(question: str) -> str:
    """질문을 간단한 규칙으로 분류합니다. 교육용이라 일부러 단순하게 작성했습니다."""
    if any(word in question for word in ["수료", "출석", "출결", "결석", "지각"]):
        return "수료/출결 상담"

    if any(word in question for word in ["평가", "점수", "시험", "과제"]):
        return "평가 상담"

    if any(word in question for word in ["취업", "이력서", "면접", "포트폴리오"]):
        return "취업지원 상담"

    if any(word in question for word in ["공지", "일정", "운영", "규정"]):
        return "운영/공지 상담"

    return "일반 교육 상담"


def extract_student_name(question: str) -> str | None:
    """
    질문 안에 포함된 교육생 이름을 찾습니다.
    예: '홍길동 교육생 수료 가능해?' -> '홍길동'
    """
    student_names = fetch_all_student_names()

    for name in student_names:
        if name in question:
            return name

    return None


def build_fake_answer(question: str, intent: str, student_context: str, policy_context: str) -> str:
    """
    API Key 없이 수업 흐름을 확인하기 위한 샘플 답변입니다.
    실제 서비스에서는 ChatOpenAI를 사용하세요.
    """
    return f"""
[FAKE_LLM 샘플 답변]

1. 핵심 답변
제공된 MariaDB 데이터와 VectorDB 문서를 함께 검토하면, 질문은 '{intent}'에 해당합니다.

2. 판단 근거
아래 MariaDB 데이터에는 교육생의 출석률, 평가 점수, 상담 이력이 포함되어 있습니다.

{student_context}

아래 VectorDB 검색 결과에는 수료 기준, 운영계획, 취업지원 안내가 포함되어 있습니다.

{policy_context}

3. 추가 안내
실제 답변 품질을 확인하려면 .env에서 USE_FAKE_LLM=false로 설정하고 OPENAI_API_KEY를 입력하세요.
""".strip()


def answer_question(question: str) -> str:
    """사용자 질문 하나에 대해 최종 답변을 생성합니다."""
    # 1. 질문 의도 분석
    intent = classify_intent(question)

    # 2. 질문에서 교육생 이름 추출
    student_name = extract_student_name(question)

    # 3. MariaDB 조회
    if student_name:
        report = fetch_student_report(student_name)
        student_context = format_student_report(report)
    else:
        student_context = "질문에서 교육생 이름을 찾지 못했습니다."

    # 4. VectorDB 문서 검색
    policy_context = search_policy_documents(question, k=3)

    # 5-A. 교육용 fake 모드
    if settings.use_fake_llm:
        return build_fake_answer(question, intent, student_context, policy_context)

    # 5-B. 실제 LLM 답변 생성
    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    chain = EDU_COUNSELING_PROMPT | llm

    response = chain.invoke(
        {
            "question": question,
            "intent": intent,
            "student_context": student_context,
            "policy_context": policy_context,
        }
    )

    return response.content
