"""Reproducible retrieval benchmark: Lynx hybrid search vs simulated agentic grep.

What this measures
------------------
For each natural-language task (e.g. "logic that decides whether saving a
model instance performs an INSERT or an UPDATE") with a known ground-truth
file, both pipelines return a ranked list of files:

  - LYNX:  one `search()` call against the hybrid (dense+BM25+RRF) index.
  - GREP:  a faithful simulation of what a grep-driven agent does first —
           extract content keywords from the question, match them across the
           tree, rank files by weighted match counts (rare words count more,
           an IDF-style weight). This is *generous* to grep: a real agent
           often needs several grep rounds to get this far.

Metrics per pipeline:
  - hit@K        ground-truth file present in the top-K results
  - MRR@K        mean reciprocal rank of the first ground-truth hit
  - tool tokens  size of the text the tool hands back to the model
                 (chars/4 approximation). NOTE: Lynx output *contains the
                 relevant code chunks*; grep output is only match lines —
                 the agent still has to open files afterwards, which costs
                 additional tokens not charged here. The comparison is
                 therefore conservative in grep's favor.
  - latency      median wall-clock seconds per query (Lynx timed after the
                 model is warm; index build time is reported separately).

Reproduce
---------
    git clone --depth 1 --branch 5.2 https://github.com/django/django.git \
        benchmarks/_target/django
    python benchmarks/run_benchmark.py

Outputs benchmarks/RESULTS.md and benchmarks/results.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent / "src"))

# Words that carry no signal for the grep keyword extraction.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "not", "is", "are", "was", "be", "been",
    "how", "where", "what", "when", "which", "that", "this", "it", "its",
    "of", "in", "on", "to", "for", "by", "with", "from", "into", "against",
    "after", "before", "between", "across", "per", "via", "like", "as",
    "does", "do", "gets", "get", "has", "have", "their", "them", "they",
    "code", "logic", "implementation", "back", "later", "same", "new",
    "whether", "instead", "keeping", "performs", "turned",
}


def _keywords(query: str) -> list[str]:
    """Content words a grep-driven agent would plausibly try."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", query.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


# ---------------------------------------------------------------------------
# Pipeline 1: simulated agentic grep
# ---------------------------------------------------------------------------


class GrepIndex:
    """All .py files read once; per-keyword counts computed on demand."""

    def __init__(self, root: Path, ext: str = ".py"):
        self.root = root
        self.files: dict[str, str] = {}
        for p in sorted(root.rglob(f"*{ext}")):
            rel = p.relative_to(root).as_posix()
            try:
                self.files[rel] = p.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue

    def search(self, query: str, top_k: int):
        """Rank files by IDF-weighted keyword frequency; emit grep-style
        match lines for the top files (the text an agent would see)."""
        kws = _keywords(query)
        n_files = len(self.files)

        counts: dict[str, dict[str, int]] = {}
        df: dict[str, int] = {}
        for kw in kws:
            df[kw] = 0
            for rel, text in self.files.items():
                c = text.count(kw)
                if c:
                    counts.setdefault(rel, {})[kw] = c
                    df[kw] += 1

        scores = {}
        for rel, kw_counts in counts.items():
            s = 0.0
            for kw, c in kw_counts.items():
                idf = math.log(n_files / (1 + df[kw]))
                s += idf * math.log(1 + c)
            scores[rel] = s

        ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]

        # Tool output the agent reads: grep-style "file:line: text" matches,
        # up to 5 lines per keyword per file (rg-like behavior, capped).
        out_lines = []
        for rel in ranked:
            text = self.files[rel]
            lines = text.splitlines()
            emitted = 0
            for i, line in enumerate(lines, 1):
                if emitted >= 5 * max(1, len(kws)):
                    break
                if any(kw in line for kw in kws):
                    out_lines.append(f"{rel}:{i}: {line.strip()[:200]}")
                    emitted += 1
        grep_output = "\n".join(out_lines)

        # The grep output contains NO code body — to actually answer, the
        # agent's next step is reading the promising region of the top file.
        # Model that as a 150-line window around the first keyword match in
        # the top-ranked file (a conservative single follow-up Read).
        follow_up = ""
        if ranked:
            lines = self.files[ranked[0]].splitlines()
            first = next(
                (i for i, line in enumerate(lines)
                 if any(kw in line for kw in kws)),
                0,
            )
            lo = max(0, first - 30)
            follow_up = "\n".join(lines[lo:lo + 150])
        return ranked, grep_output, follow_up


# ---------------------------------------------------------------------------
# Pipeline 2: Lynx
# ---------------------------------------------------------------------------


def build_lynx_backend(index_dir: Path, storage_dir: Path, ext: str = ".py"):
    """Stand up a CodebaseRAG over the target tree (builds the index if the
    storage dir is empty; reuses it otherwise thanks to the SHA cache)."""
    import os
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # Same offline policy as the server: never touch the network when the
    # embedding model is already cached (and the heavy imports must see the
    # env flags, so this runs before importing rag_manager).
    from lynx.config import _hf_model_cached
    if _hf_model_cached("BAAI/bge-small-en-v1.5"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from lynx.rag_manager import CodebaseRAG

    # The constructor itself performs the initial build when the storage is
    # empty, so the timer must wrap construction, not just update().
    t0 = time.time()
    rag = CodebaseRAG(
        codebase_path=str(index_dir),
        rag_storage_path=str(storage_dir),
        supported_extensions=[ext],
        embedding_model_name="BAAI/bge-small-en-v1.5",
        collection_name="bench",
        search_mode="hybrid",
        rrf_k=60,
        candidate_pool_size=30,
    )
    rag.update(force=rag.vector_store._collection.count() == 0)
    build_seconds = time.time() - t0
    return rag, build_seconds


def lynx_search(rag, query: str, top_k: int, index_dir: Path):
    """One search call; returns (ranked file list, tool output text)."""
    results = rag.search(query, top_k=top_k)
    ranked, seen = [], set()
    out_chunks = []
    for r in results:
        fp = (r.get("file_path") or r.get("file") or "").replace("\\", "/")
        # Normalize to path relative to the indexed tree.
        rel = fp.split(index_dir.name + "/", 1)[-1] if index_dir.name + "/" in fp else fp
        if rel not in seen:
            seen.add(rel)
            ranked.append(rel)
        out_chunks.append(
            f"--- {rel} L{r.get('start_line')}-{r.get('end_line')} "
            f"{r.get('symbol_name') or ''} (score {r.get('score', 0):.4f})\n"
            f"{r.get('content', '')}"
        )
    # Third element is the follow-up read cost: zero for Lynx, because the
    # tool output above already contains the code chunks themselves.
    return ranked, "\n".join(out_chunks), ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _hit_rank(ranked: list[str], ground_truth: list[str]):
    """1-based rank of the first ranked file matching any ground truth
    (suffix match, so both 'django/db/...' and 'db/...' forms work)."""
    for i, rel in enumerate(ranked, 1):
        for gt in ground_truth:
            if rel.endswith(gt):
                return i
    return None


def evaluate(name, search_fn, tasks, top_k):
    rows, latencies, token_counts, answer_tokens, ranks = [], [], [], [], []
    for task in tasks:
        t0 = time.time()
        ranked, tool_output, follow_up = search_fn(task["query"], top_k)
        dt = time.time() - t0
        rank = _hit_rank(ranked, task["ground_truth"])
        tool_tok = len(tool_output) // 4
        answer_tok = tool_tok + len(follow_up) // 4
        latencies.append(dt)
        token_counts.append(tool_tok)
        answer_tokens.append(answer_tok)
        ranks.append(rank)
        rows.append({
            "id": task["id"], "query": task["query"], "rank": rank,
            "latency_s": round(dt, 3), "tool_tokens": tool_tok,
            "answer_tokens": answer_tok, "top": ranked[:3],
        })
        flag = f"rank {rank}" if rank else "MISS"
        print(f"  [{name}] task {task['id']:>2}: {flag:>7}  "
              f"{dt:5.2f}s  {tool_tok:>5} tok (+{answer_tok - tool_tok} follow-up)",
              file=sys.stderr)
    n = len(tasks)
    return {
        "pipeline": name,
        "hit_at_k": sum(1 for r in ranks if r) / n,
        "hit_at_1": sum(1 for r in ranks if r == 1) / n,
        "mrr": sum((1 / r) for r in ranks if r) / n,
        "median_latency_s": round(statistics.median(latencies), 3),
        "median_tool_tokens": int(statistics.median(token_counts)),
        "median_answer_tokens": int(statistics.median(answer_tokens)),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(BENCH_DIR / "tasks_django.json"))
    ap.add_argument("--target-dir", default=str(BENCH_DIR / "_target" / "django"))
    ap.add_argument("--storage-dir", default=str(BENCH_DIR / "_storage"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--results-json", default=str(BENCH_DIR / "results.json"))
    ap.add_argument("--results-md", default=str(BENCH_DIR / "RESULTS.md"))
    args = ap.parse_args()

    spec = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = spec["tasks"]
    ext = spec["target"].get("ext", ".py")
    index_dir = Path(args.target_dir) / spec["target"]["index_subdir"]
    if not index_dir.is_dir():
        sys.exit(f"target not found: {index_dir}\nClone it first (see module docstring).")

    # Sanity: every ground-truth file must exist in the pinned checkout.
    missing = [
        gt for t in tasks for gt in t["ground_truth"]
        if not (index_dir / gt).is_file()
    ]
    if missing:
        sys.exit(f"ground-truth files missing from checkout: {missing}")

    n_files = sum(1 for _ in index_dir.rglob(f"*{ext}"))
    n_lines = sum(
        len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
        for p in index_dir.rglob(f"*{ext}")
    )
    print(f"target: {index_dir} ({n_files} files, {n_lines} lines)", file=sys.stderr)

    print("building grep corpus...", file=sys.stderr)
    grep = GrepIndex(index_dir, ext)

    print("building / loading Lynx index (first run embeds the corpus)...", file=sys.stderr)
    rag, build_seconds = build_lynx_backend(index_dir, Path(args.storage_dir), ext)
    print(f"index ready in {build_seconds:.0f}s "
          f"({rag.vector_store._collection.count()} chunks)", file=sys.stderr)
    rag.search("warmup query", top_k=1)  # exclude model warm-up from timings

    grep_summary = evaluate("grep", grep.search, tasks, args.top_k)
    lynx_summary = evaluate(
        "lynx", lambda q, k: lynx_search(rag, q, k, index_dir), tasks, args.top_k
    )

    results = {
        "target": spec["target"], "files": n_files, "lines": n_lines,
        "top_k": args.top_k, "index_build_seconds": round(build_seconds, 1),
        "chunks": rag.vector_store._collection.count(),
        "grep": grep_summary, "lynx": lynx_summary,
    }
    Path(args.results_json).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    tname = spec["target"].get("name", spec["target"]["index_subdir"])
    tref = spec["target"].get("ref", "")
    k = args.top_k
    md = [
        f"# Lynx retrieval benchmark — {tname}",
        "",
        f"Target: `{spec['target']['index_subdir']}` of "
        f"[{tname} {tref}]({spec['target']['repo']}) — "
        f"{n_files} files, {n_lines:,} lines, {results['chunks']:,} indexed chunks "
        f"(one-time index build: {build_seconds:.0f}s on CPU).",
        "",
        f"{len(tasks)} natural-language tasks with known ground-truth files. "
        "Both pipelines return a ranked file list; see `run_benchmark.py` for "
        "the exact methodology (the grep baseline is deliberately generous).",
        "",
        f"| Metric | Agentic grep (simulated) | Lynx hybrid search |",
        f"|---|---|---|",
        f"| hit@{k} (ground truth in top {k}) | {grep_summary['hit_at_k']:.0%} | {lynx_summary['hit_at_k']:.0%} |",
        f"| hit@1 (top result is correct) | {grep_summary['hit_at_1']:.0%} | {lynx_summary['hit_at_1']:.0%} |",
        f"| MRR | {grep_summary['mrr']:.2f} | {lynx_summary['mrr']:.2f} |",
        f"| median tool output (tokens) | {grep_summary['median_tool_tokens']:,} | {lynx_summary['median_tool_tokens']:,} |",
        f"| median tokens **to answer** (incl. required follow-up read) | {grep_summary['median_answer_tokens']:,} | {lynx_summary['median_answer_tokens']:,} |",
        f"| tool round-trips to reach code | 2 (grep, then read file) | 1 (chunks included) |",
        f"| median latency / query | {grep_summary['median_latency_s']}s | {lynx_summary['median_latency_s']}s |",
        "",
        "Notes:",
        "- Lynx tool output already CONTAINS the relevant code chunks with file:line.",
        "  The grep output is match lines only, so its 'to answer' figure adds ONE",
        "  conservative follow-up read (150 lines around the first match in the top",
        "  file); real agents often need several reads and several grep refinements.",
        "- The grep baseline is intentionally strong (IDF-weighted multi-keyword",
        "  ranking with ideal stopword removal — closer to BM25 than to what an",
        "  agent's first `rg` attempt looks like).",
        "- A well-documented, English-identifier corpus is a favorable case for",
        "  lexical search. Codebases with sparser comments or non-obvious naming",
        "  shift results further toward semantic retrieval.",
        "- Tokens approximated as chars/4.",
        "",
        "## Beyond ranking: what one tool call hands the model",
        "",
        "The ranking metrics above understate the practical gap, because the two",
        "outputs are not the same *kind* of thing:",
        "",
        "| Per result | grep | Lynx |",
        "|---|---|---|",
        "| matching line(s) | ✅ | ✅ (whole AST chunk: the full function/class body) |",
        "| file + line range of the enclosing symbol | ❌ | ✅ (`db/models/base.py L767-881`) |",
        "| qualified symbol name (`Model._save_table`) | ❌ | ✅ — citable directly in the answer |",
        "| relevance score (when to stop trusting results) | ❌ | ✅ |",
        "| works when the query shares NO words with the code | ❌ | ✅ (dense embeddings) |",
        "",
        "And there is a whole class of questions grep cannot answer at any number",
        "of calls, that Lynx answers in ONE (`graph_query` / `find_usages`):",
        "*who calls X? what implements this interface? what breaks if I change",
        "this?* — polymorphic dispatch leaves no textual trace, so grepping the",
        "method name finds the definition and the textual mentions, not the",
        "runtime callers through base classes.",
        "",
        "## The round-trip economy",
        "",
        "Tool execution time is noise (both pipelines answer in well under a",
        "second). The real cost is that **every tool round-trip is a full model",
        "inference over the entire growing context**: seconds of wall-clock and",
        "the whole conversation re-billed in input tokens, every time.",
        "",
        "| Typical flow | model inferences before the code is in context |",
        "|---|---|",
        "| grep: search → read top file → (often) refine search → read again | 3-5 |",
        "| Lynx `search`: chunks arrive with the first response | **1** |",
        "| grep: reconstruct callers of a symbol (N textual hits to triage) | 2 + N reads, dispatch still missed |",
        "| Lynx `graph_query(\"callers\")`: resolved edges with file:line | **1** |",
        "",
        "## Per-task results",
        "",
        "| # | Task | grep rank | Lynx rank |",
        "|---|---|---|---|",
    ]
    for g_row, l_row in zip(grep_summary["rows"], lynx_summary["rows"]):
        md.append(
            f"| {g_row['id']} | {g_row['query'][:70]} "
            f"| {g_row['rank'] or '—'} | {l_row['rank'] or '—'} |"
        )
    Path(args.results_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {args.results_md} and {args.results_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
