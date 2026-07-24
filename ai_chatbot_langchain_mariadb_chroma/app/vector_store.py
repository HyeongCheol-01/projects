"""
VectorDB 접근 모듈

- sample_docs/ 폴더의 텍스트 문서를 읽습니다.
- OpenAI Embedding으로 벡터화합니다.
- Chroma VectorDB에 저장합니다.
- 질문과 의미가 가까운 문서를 검색합니다.
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from app.config import PROJECT_ROOT, settings

COLLECTION_NAME = "edu_policy_docs"


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델 객체를 생성합니다."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vector_store() -> Chroma:
    """기존 Chroma VectorDB를 불러옵니다."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
    )


def load_text_documents(doc_dir: Path) -> list[Document]:
    """sample_docs 폴더의 .txt 파일을 LangChain Document로 변환합니다."""
    documents: list[Document] = []

    for path in sorted(doc_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name},
            )
        )

    return documents


def ingest_documents(doc_dir: Path | None = None) -> int:
    """문서를 잘게 나눈 뒤 Chroma VectorDB에 저장합니다."""
    doc_dir = doc_dir or PROJECT_ROOT / "sample_docs"

    raw_docs = load_text_documents(doc_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
    )
    split_docs = splitter.split_documents(raw_docs)

    # 기존 컬렉션에 계속 추가됩니다.
    # 수업 중 완전히 초기화하고 싶다면 chroma_db 폴더를 삭제한 뒤 다시 실행하세요.
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=settings.chroma_dir,
    )

    # langchain-chroma 최신 버전에서는 자동 저장되지만, 하위 호환용으로 남겨둡니다.
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    return len(split_docs)


def search_policy_documents(question: str, k: int = 3) -> str:
    """질문과 관련 있는 운영 문서/공지/취업지원 문서를 검색합니다."""
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(question, k=k)

    if not docs:
        return "관련 문서를 VectorDB에서 찾지 못했습니다."

    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(
            f"[VectorDB 검색 문서 {i}]\n"
            f"- 출처: {source}\n"
            f"- 내용:\n{doc.page_content}"
        )

    return "\n\n".join(blocks)
