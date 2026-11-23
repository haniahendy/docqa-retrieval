"""Hit@1 on the small hand-written eval set in data/eval_set.json."""
import json
from docqa.api import DATA, load_docs
from docqa.retriever import TfidfRetriever

r = TfidfRetriever()
r.add_documents(load_docs())
items = json.loads((DATA / "eval_set.json").read_text(encoding="utf-8"))
hits = sum(1 for it in items if (res := r.search(it["q"], 1)) and res[0].doc == it["doc"])
print(f"hit@1 = {hits}/{len(items)}")
