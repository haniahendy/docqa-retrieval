# docqa-retrieval

Passage retrieval service for English and Arabic documents: chunking, Arabic text normalization, TF-IDF (word 1-2 grams) ranking, and a small FastAPI endpoint.

**What it is not:** it does not generate answers, does not use embeddings or an LLM. It returns the most similar passages. It is a baseline you can later swap for a dense retriever or put in front of an LLM (the RAG "R" part).

## Run
```bash
pip install -r requirements.txt
python -m pytest -q
python evaluate.py
uvicorn docqa.api:app --reload      # POST /ask {"question": "...", "k": 3}
```
Put your own `.txt` files in `data/` to index them.

## Measured result (this repo, run on 23 Nov 2026)
- 7 unit/API tests pass.
- `evaluate.py`: hit@1 = 6/6 on a hand-written set of 6 questions over 3 short documents (2 Arabic questions both correct). Very small set, so treat as a sanity check, not a benchmark.
- Before adding crude plural stripping, hit@1 was 5/6 ("vaccine" did not match "vaccines"). The fix targets that failure on a tiny set, so 6/6 is not evidence of general quality.

## Limits and next steps
- Only crude English plural stripping; no real stemming or Arabic morphology beyond character normalization.
- Add dense embeddings (multilingual sentence model) and compare hit@k.
- Add reranking and answer generation with citations.

## Provenance
Built with AI assistance as a learning and portfolio project. Documents in `data/` were written for this repo.
