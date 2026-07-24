"""
02_ingest_docs.py

sample_docs/ 폴더의 텍스트 문서를 Chroma VectorDB에 적재합니다.

실행:
python scripts/02_ingest_docs.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.vector_store import ingest_documents


def main() -> None:
    count = ingest_documents()
    print(f"VectorDB 문서 적재 완료: {count}개 chunk 저장")


if __name__ == "__main__":
    main()
