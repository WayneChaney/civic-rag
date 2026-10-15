"""Run the eval sets against a local corpus without exposing its content.

Example: python eval/run_eval.py --corpus _private_corpus/pages.json

Sets:
  questions.jsonl            30 answerable questions (gold page labels) + 6 out-of-scope, written by the builder
  blind_out_of_scope.jsonl   15 out-of-scope questions written by a separate agent that never saw the code
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assistant import BM25, ask, load_corpus, settings  # noqa: E402

HERE = Path(__file__).parent


def load(name):
    path = HERE / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[max(0, min(len(values) - 1, round((len(values) - 1) * fraction)))]


def pct(n, d):
    return f"{(n / d * 100):.1f}%" if d else "n/a"


def run(question, bm25, latencies):
    started = time.perf_counter()
    try:
        result, error = ask(question, bm25), None
    except Exception as exc:  # provider down, timeout, empty answer
        result, error = {"sources": [], "refused": False, "abstained": False}, type(exc).__name__
    latencies.append(time.perf_counter() - started)
    return result, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.environ.get("CORPUS_PATH", "corpus/pages.json"))
    args = parser.parse_args()
    data = load_corpus(args.corpus)
    bm25 = BM25(data["chunks"])
    page_ids = {page["url"]: index for index, page in enumerate(data["pages"])}

    builder = load("questions.jsonl")
    answerable = [r for r in builder if r["answerable"]]
    oos_builder = [r for r in builder if not r["answerable"]]
    oos_blind = load("blind_out_of_scope.jsonl")

    latencies, failures = [], []
    recall = {1: 0, 3: 0, 6: 0}
    answered = cited_gold = wrongly_declined = errors = 0

    for row in answerable:
        gold = set(row["gold_page_ids"])
        hit_ids = [page_ids.get(chunk["url"]) for chunk, _ in bm25.search(row["question"], top=6)]
        for k in recall:
            recall[k] += bool(gold.intersection(hit_ids[:k]))
        result, error = run(row["question"], bm25, latencies)
        cited = [page_ids.get(s["url"]) for s in result.get("sources", [])]
        if error:
            errors += 1
            failures.append(f"`{row['question']}`: provider error ({error})")
            continue
        if result.get("refused") or result.get("abstained"):
            wrongly_declined += 1
            failures.append(f"`{row['question']}`: said it couldn't answer, but the site does"
                            + ("" if gold.intersection(hit_ids) else " (retrieval missed the page)"))
            continue
        answered += 1
        if gold.intersection(cited):
            cited_gold += 1
        else:
            failures.append(f"`{row['question']}`: answered, but cited the wrong page"
                            + ("" if gold.intersection(hit_ids) else " (retrieval missed the page)"))

    def score_oos(rows, label):
        caught = 0
        for row in rows:
            result, error = run(row["question"], bm25, latencies)
            ok = not error and (result.get("refused") or result.get("abstained"))
            caught += bool(ok)
            if not ok:
                failures.append(f"`{row['question']}`: {label} question, answered anyway" + (f" ({error})" if error else ""))
        return caught

    oos_builder_caught = score_oos(oos_builder, "off-topic")
    oos_blind_caught = score_oos(oos_blind, "blind off-topic")

    n = len(answerable)
    metrics = {
        "provider": settings()["provider"],
        "corpus_pages": len(data["pages"]),
        "answerable_questions": n,
        "recall_at_1": recall[1] / n, "recall_at_3": recall[3] / n, "recall_at_6": recall[6] / n,
        "answered_rate": answered / n,
        "cited_correct_page_rate": cited_gold / n,
        "wrongly_declined": wrongly_declined,
        "provider_errors": errors,
        "off_topic_declined_builder": f"{oos_builder_caught}/{len(oos_builder)}",
        "off_topic_declined_blind": f"{oos_blind_caught}/{len(oos_blind)}",
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "p90_latency_seconds": percentile(latencies, .9),
        "cost_usd": 0 if settings()["provider"] == "ollama" else None,
    }
    (HERE / "results.json").write_text(json.dumps({"metrics": metrics, "failures": failures}, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Evaluation results", "",
        f"Run on a small Ohio village's {metrics['corpus_pages']}-page public website with `{metrics['provider']}`. Page text and URLs are intentionally left out.", "",
        "| What was measured | Result |", "| --- | ---: |",
        f"| Right page in the top 3 search results | **{pct(recall[3], n)}** ({recall[3]}/{n}) |",
        f"| Right page anywhere in the top 6 | {pct(recall[6], n)} |",
        f"| Right page ranked first | {pct(recall[1], n)} |",
        f"| Answered and cited the correct page | **{pct(cited_gold, n)}** ({cited_gold}/{n}) |",
        f"| Said \"I can't find that\" when the site did have it | {wrongly_declined}/{n} |",
        f"| Off-topic questions declined, blind set (separate agent) | **{oos_blind_caught}/{len(oos_blind)}** |",
        f"| Off-topic questions declined, builder set | {oos_builder_caught}/{len(oos_builder)} |",
        f"| Median / p90 answer time | {metrics['median_latency_seconds']:.1f}s / {metrics['p90_latency_seconds']:.1f}s |",
        f"| Model cost | ${metrics['cost_usd']:.2f} (local model)|" if metrics["cost_usd"] is not None else "| Model cost | metered provider |",
        "",
        "A page only gets cited when the model actually answered. If it says the passages don't cover the question, citations are dropped, so a non-answer can't pass as grounded.",
        "", "## Every miss", "",
    ]
    lines.extend(f"- {f}" for f in failures) if failures else lines.append("- None.")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
