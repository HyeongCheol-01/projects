# 교육용 미니 예제: LangChain + MariaDB + Chroma VectorDB

이 예제는 교육기관 AI 상담 챗봇을 아주 작게 만든 코드입니다.

## 목표

질문 하나가 들어오면 아래 흐름으로 답변합니다.

```text
사용자 질문
  ↓
MariaDB에서 교육생 정보 조회
  ↓
Chroma VectorDB에서 수료/취업지원 문서 검색
  ↓
LLM이 상담 답변 생성
```

## 파일 구성

```text
mini_chatbot.py       # 전체 코드
create_database.sql   # DB 생성 SQL
.env.example          # 환경 변수 예시
requirements.txt      # 설치 패키지
README.md             # 설명 문서
```

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. MariaDB 데이터베이스 생성

MariaDB에서 아래 SQL을 실행합니다.

```sql
CREATE DATABASE IF NOT EXISTS edu_ai DEFAULT CHARACTER SET utf8mb4;
```

또는 터미널에서 실행합니다.

```bash
mysql -u root -p < create_database.sql
```

### 3. 환경 변수 파일 생성

```bash
copy .env.example .env
```

macOS/Linux는 다음과 같습니다.

```bash
cp .env.example .env
```

`.env` 파일에서 DB 비밀번호와 OpenAI API Key를 수정합니다.

### 4. 실행

```bash
python mini_chatbot.py
```

## 수업 설명 포인트

### 1. MariaDB

교육생의 정형 데이터를 저장합니다.

```text
이름, 과정명, 출석률, 평가점수, 상담메모
```

### 2. VectorDB

수료 기준, 출결 기준, 취업지원 안내문처럼 문서성 데이터를 저장합니다.

### 3. LangChain

DB 조회 결과와 문서 검색 결과를 LLM에게 전달해 최종 답변을 생성합니다.

## 예시 질문

```text
홍길동 교육생은 현재 수료 가능성이 있나요?
김민수 교육생은 출석률 때문에 상담이 필요한가요?
이영희 교육생에게 어떤 취업지원을 안내하면 좋을까요?
```
