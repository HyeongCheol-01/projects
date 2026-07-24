"""
프로젝트 설정 파일
- .env 파일에서 DB, OpenAI, VectorDB 설정을 읽어옵니다.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로: edu_ai_chatbot_langchain_mariadb_chroma/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# .env 파일 로드
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # MariaDB 설정
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "3306"))
    db_user: str = os.getenv("DB_USER", "root")
    db_password: str = os.getenv("DB_PASSWORD", "1234")
    db_name: str = os.getenv("DB_NAME", "edu_ai_demo")

    # OpenAI / LangChain 설정
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Chroma VectorDB 설정
    chroma_dir: str = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))

    # 교육용 옵션: true이면 실제 LLM API를 호출하지 않고 샘플 답변 생성
    use_fake_llm: bool = os.getenv("USE_FAKE_LLM", "false").lower() == "true"

    @property
    def sqlalchemy_uri(self) -> str:
        """LangChain SQLDatabase에서 사용할 SQLAlchemy URI입니다."""
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )


settings = Settings()
