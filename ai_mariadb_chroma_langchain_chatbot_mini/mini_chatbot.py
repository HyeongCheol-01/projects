"""
미니 예제: LangChain + MariaDB + Chroma VectorDB 상담 챗봇

흐름:
1) MariaDB에서 교육생 정보 조회
2) Chroma VectorDB에서 수료/취업지원 기준 문서 검색
3) LangChain + LLM으로 상담 답변 생성
"""

import os
import shutil
from pathlib import Path

import pymysql
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# 1. 기본 설정
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "database": os.getenv("DB_NAME", "edu_ai"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

VECTOR_DIR = Path("./chroma_mini_db")

# VectorDB에 넣을 작은 문서들입니다.
# 실제 현업에서는 PDF, HWP, DOCX, 공지사항 등을 잘라서 넣습니다.
POLICY_DOCS = [
    Document(
        page_content="수료 기준: 출석률은 80% 이상이어야 하며, 평가 점수는 60점 이상이어야 한다.",
        metadata={"source": "수료기준"},
    ),
    Document(
        page_content="출석률이 80% 미만인 교육생은 보강 또는 상담이 필요하다.",
        metadata={"source": "출결관리"},
    ),
    Document(
        page_content="취업지원: 이력서 첨삭, 포트폴리오 점검, 모의면접을 제공한다.",
        metadata={"source": "취업지원"},
    ),
]


# ------------------------------------------------------------
# 2. MariaDB 샘플 데이터 생성
# ------------------------------------------------------------
def get_conn():
    """MariaDB 연결 객체를 반환합니다."""
    return pymysql.connect(**DB_CONFIG)


def init_mariadb():
    """수업용 샘플 테이블과 데이터를 만듭니다."""
    sql_list = [
        "DROP TABLE IF EXISTS students",
        """
        CREATE TABLE students (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(20) NOT NULL,
            course_name VARCHAR(100) NOT NULL,
            attendance_rate FLOAT NOT NULL,
            score INT NOT NULL,
            counseling_note TEXT
        )
        """,
        """
        INSERT INTO students
        (name, course_name, attendance_rate, score, counseling_note)
        VALUES
        ('홍길동', 'AI 서비스 개발자 과정', 82.5, 75, '프로젝트 참여도가 좋고 취업 의지가 높음'),
        ('김민수', 'AI 서비스 개발자 과정', 72.0, 68, '출석률 관리가 필요함'),
        ('이영희', 'AI 서비스 개발자 과정', 91.0, 88, '포트폴리오 완성도가 높음')
        """,
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            for sql in sql_list:
                cur.execute(sql)
        conn.commit()

    print("MariaDB 샘플 데이터 생성 완료")


# ------------------------------------------------------------
# 3. Chroma VectorDB 생성
# ------------------------------------------------------------
def init_vectordb():
    """작은 문서 3개를 Chroma VectorDB에 저장합니다."""
    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)

    embeddings = OpenAIEmbeddings()

    Chroma.from_documents(
        documents=POLICY_DOCS,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
        collection_name="edu_policy",
    )

    print("Chroma VectorDB 생성 완료")


# ------------------------------------------------------------
# 4. MariaDB 조회 함수
# ------------------------------------------------------------
def find_student_name(question: str) -> str | None:
    """질문 문장 안에 교육생 이름이 있는지 찾습니다."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM students")
            rows = cur.fetchall()

    for row in rows:
        name = row["name"]
        if name in question:
            return name

    return None


def get_student_info(name: str | None) -> str:
    """MariaDB에서 특정 교육생 정보를 조회합니다."""
    if not name:
        return "질문에서 교육생 이름을 찾지 못했습니다."

    sql = """
    SELECT name, course_name, attendance_rate, score, counseling_note
    FROM students
    WHERE name = %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (name,))
            row = cur.fetchone()

    if not row:
        return f"{name} 교육생 정보를 찾지 못했습니다."

    return f"""
교육생명: {row['name']}
과정명: {row['course_name']}
출석률: {row['attendance_rate']}%
평가점수: {row['score']}점
상담메모: {row['counseling_note']}
""".strip()


# ------------------------------------------------------------
# 5. VectorDB 검색 함수
# ------------------------------------------------------------
def search_policy_docs(question: str) -> str:
    """질문과 관련 있는 수료/취업지원 문서를 검색합니다."""
    embeddings = OpenAIEmbeddings()

    vectordb = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
        collection_name="edu_policy",
    )

    docs = vectordb.similarity_search(question, k=2)

    return "\n".join(
        [f"- {doc.page_content} 출처: {doc.metadata['source']}" for doc in docs]
    )


# ------------------------------------------------------------
# 6. LangChain으로 최종 답변 생성
# ------------------------------------------------------------
def answer(question: str) -> str:
    """DB 정보 + 문서 검색 결과를 LLM에게 전달해 답변을 만듭니다."""
    student_name = find_student_name(question)
    student_info = get_student_info(student_name)
    policy_context = search_policy_docs(question)

    prompt = ChatPromptTemplate.from_template(
        """
당신은 교육기관 상담 챗봇입니다.
아래의 MariaDB 조회 결과와 VectorDB 문서 검색 결과만 근거로 답변하세요.
답변은 교육생이 이해하기 쉽게 3~5문장으로 작성하세요.

[사용자 질문]
{question}

[MariaDB 조회 결과]
{student_info}

[VectorDB 문서 검색 결과]
{policy_context}

[답변]
"""
    )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "question": question,
            "student_info": student_info,
            "policy_context": policy_context,
        }
    )

    return result.content


# ------------------------------------------------------------
# 7. 실행부
# ------------------------------------------------------------
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(".env 파일에 OPENAI_API_KEY를 입력하세요.")

    # 수업에서는 매번 초기화되도록 구성했습니다.
    init_mariadb()
    init_vectordb()

    questions = [
        "홍길동 교육생은 현재 수료 가능성이 있나요?",
        "김민수 교육생은 출석률 때문에 상담이 필요한가요?",
        "이영희 교육생에게 어떤 취업지원을 안내하면 좋을까요?",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print("질문:", q)
        print("-" * 60)
        print(answer(q))
