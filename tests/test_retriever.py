import pytest
from fastapi.testclient import TestClient

from docqa.api import build_app, load_docs
from docqa.retriever import TfidfRetriever, chunk_text, normalize


def test_normalize_arabic_variants():
    assert normalize("إِنْتَاجُ الكهرباءِ") == normalize("انتاج الكهرباء")
    assert normalize("مدرسة") == normalize("مدرسه")


def test_chunk_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, max_words=30, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert chunks[-1].split()[-1] == "w99"


def test_chunk_invalid_args():
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=5, overlap=5)
    assert chunk_text("") == []


def test_search_returns_right_document():
    r = TfidfRetriever()
    r.add_documents(load_docs())
    assert r.search("how do vaccines work with antibodies", 1)[0].doc == "vaccines_en.txt"
    assert r.search("نهر النيل والزراعة", 1)[0].doc == "nile_ar.txt"


def test_search_without_index_raises():
    with pytest.raises(RuntimeError):
        TfidfRetriever().search("x")


def test_unrelated_query_returns_nothing():
    r = TfidfRetriever()
    r.add_documents(load_docs())
    assert r.search("zzzz qqqq", 3) == []


def test_api():
    client = TestClient(build_app())
    assert client.get("/health").json() == {"status": "ok"}
    res = client.post("/ask", json={"question": "why do solar panels need cleaning", "k": 2}).json()
    assert res["passages"][0]["doc"] == "solar_panels_en.txt"
