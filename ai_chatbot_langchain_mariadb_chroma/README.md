# 교육기관 AI 상담 챗봇

LangChain, MariaDB, Chroma VectorDB, OpenAI LLM을 함께 사용하는 교육용/실무형 예제입니다.

## 1. 전체 구조

```text
사용자 질문
   ↓
질문 의도 분석
   ↓
MariaDB 조회
   - 교육생
   - 과정
   - 출결
   - 평가
   - 상담 이력
   ↓
Chroma VectorDB 검색
   - 수료 기준
   - 훈련 운영계획서
   - 공지사항
   - 취업지원 안내문
   ↓
LangChain Prompt + LLM
   ↓
최종 상담 답변
```

## 2. 프로젝트 구성

```text
edu_ai_chatbot_langchain_mariadb_chroma/
├─ app/
│  ├─ config.py          # .env 설정 읽기
│  ├─ db.py              # MariaDB 조회 함수
│  ├─ vector_store.py    # Chroma VectorDB 적재/검색 함수
│  ├─ prompts.py         # LangChain PromptTemplate
│  └─ chatbot.py         # 질문 분석 → DB 조회 → 문서 검색 → 답변 생성
│
├─ scripts/
│  ├─ 01_seed_mariadb.py # MariaDB 샘플 데이터 생성
│  ├─ 02_ingest_docs.py  # 문서를 VectorDB에 적재
│  └─ 03_run_chatbot.py  # 챗봇 실행
│
├─ sample_docs/          # VectorDB에 넣을 교육기관 문서
├─ requirements.txt
├─ .env.example
└─ README.md
```

## 3. 설치

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 4. 환경변수 설정

`.env.example` 파일을 복사해서 `.env`로 만듭니다.

```bash
copy .env.example .env
```

macOS/Linux는 다음처럼 실행합니다.

```bash
cp .env.example .env
```

`.env` 예시:

```env
OPENAI_API_KEY=여기에_API_KEY_입력
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=1234
DB_NAME=edu_ai_demo

CHROMA_DIR=./chroma_db
USE_FAKE_LLM=false
```

API Key 없이 수업 흐름만 확인하려면 다음처럼 설정합니다.

```env
USE_FAKE_LLM=true
```

단, 문서 임베딩 단계에서는 OpenAI Embedding API Key가 필요합니다.

## 5. 실행 순서

### 1단계: MariaDB 샘플 데이터 생성

```bash
python scripts/01_seed_mariadb.py
```

생성되는 테이블:

```text
courses
students
attendances
evaluations
counseling_notes
```

### 2단계: VectorDB 문서 적재

```bash
python scripts/02_ingest_docs.py
```

`sample_docs/` 폴더의 문서가 Chroma VectorDB에 저장됩니다.

### 3단계: 챗봇 실행

```bash
python scripts/03_run_chatbot.py
```

예시 질문:

```text
홍길동 교육생은 현재 수료 가능성이 있나요?
김민수 교육생은 출석률 때문에 문제가 있나요?
이영희 교육생에게 취업지원은 어떻게 안내하면 좋나요?
```

## 6. 예시 답변 방향

질문:

```text
홍길동 교육생은 현재 수료 가능성이 있나요?
```

답변 방향:

```text
홍길동 교육생은 현재 출석률이 약 82%이고,
과정 수료 기준인 80% 이상을 충족하고 있습니다.
평균 평가 점수도 기준인 60점 이상이므로,
현재 기준으로는 수료 가능성이 있습니다.

다만 출석률이 80% 초반이기 때문에
추가 결석이나 지각이 발생하면 수료 기준 미달 위험이 있습니다.
따라서 남은 기간에는 출결 관리가 중요합니다.
```

## 7. 현업형 포인트

이 예제는 일부러 LLM이 직접 SQL을 생성하지 않게 구성했습니다.

```text
사용자 질문 → LLM이 SQL 생성 → MariaDB 실행
```

이 구조는 편하지만 보안상 위험할 수 있습니다.

그래서 이 예제는 다음 구조를 사용합니다.

```text
사용자 질문
   ↓
파이썬 코드가 안전한 SELECT 쿼리 실행
   ↓
LLM은 조회된 결과만 보고 답변 생성
```

현업에서는 다음을 추가하는 것이 좋습니다.

- 읽기 전용 DB 계정 사용
- 접근 가능한 테이블 제한
- 개인정보 마스킹
- SQL 실행 로그 기록
- 답변 근거 문서 표시
- 관리자 검토 기능
- LangSmith 같은 추적/평가 도구 연결
