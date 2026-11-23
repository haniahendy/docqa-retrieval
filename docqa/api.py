from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .retriever import TfidfRetriever

DATA = Path(__file__).resolve().parent.parent / "data"


def load_docs(folder: Path = DATA) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.txt"))}


def build_app(docs: dict[str, str] | None = None) -> FastAPI:
    retriever = TfidfRetriever()
    retriever.add_documents(docs if docs is not None else load_docs())
    app = FastAPI(title="docqa-retrieval")

    class Ask(BaseModel):
        question: str
        k: int = 3

    @app.post("/ask")
    def ask(body: Ask):
        return {"passages": [h.__dict__ for h in retriever.search(body.question, body.k)]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = build_app()
