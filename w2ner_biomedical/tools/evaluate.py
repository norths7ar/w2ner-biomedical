# =============================================================================
# tools/evaluate.py
#
# PURPOSE
#   Compute entity-level precision, recall, and F1 by comparing step06
#   PostprocessRecord predictions against gold annotation JSON files.
#
# MATCHING CRITERION
#   An entity is correct if ALL of the following match exactly:
#     - PMID (document identity)
#     - Frozenset of (start, end) character spans (fulltext-absolute)
#     - Entity type string
#   This is the standard strict NER evaluation (exact boundary + type match).
#   Partial span matches are not counted as correct.
#
# COORDINATE SYSTEM
#   Gold JSON offsets:  fulltext-absolute (title + " " + abstract, NFKC).
#   PostprocessRecord:  sentence_spans are relative to fulltext_offset.
#                       This script adds fulltext_offset before comparison.
#
# GOLD FORMAT
#   Each file in --gold is a JSON array of annotation dicts:
#     [{"PMID": "...", "label": [[ [[s,e],...], "Type" ], ...] }, ...]
#
# PRED FORMAT
#   Each file in --pred is a PostprocessRecord JSONL (step06 output).
#
# USAGE
#   python -m w2ner_biomedical.tools.evaluate \
#       --gold data/raw/biored_test \
#       --pred data/predictions/biored_test \
#       [--output results/biored_test_metrics.json] \
#       [--type-filter Chemical Disease]
# =============================================================================

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# An entity key: (frozenset of (start, end) tuples, type string)
EntityKey = tuple[frozenset, str]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_gold(gold_dir: Path) -> dict[str, set[EntityKey]]:
    """Load gold annotations from all JSON files in gold_dir.

    Returns {pmid: set of EntityKey}.
    """
    gold: dict[str, set[EntityKey]] = defaultdict(set)

    for path in sorted(gold_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            docs = json.load(f)

        for doc in docs:
            pmid = str(doc["PMID"])
            for label_entry in doc.get("label", []):
                spans_list, ent_type = label_entry
                # spans_list: [[start, end], ...] — one or more fragments
                span_set = frozenset(tuple(s) for s in spans_list)
                gold[pmid].add((span_set, ent_type))

    return dict(gold)


def load_predictions(pred_dir: Path) -> dict[str, set[EntityKey]]:
    """Load predictions from all PostprocessRecord JSONL files in pred_dir.

    RefinedEntity.sentence_spans are relative to PostprocessRecord.fulltext_offset;
    this function converts them to fulltext-absolute before returning.

    Returns {pmid: set of EntityKey}.
    """
    pred: dict[str, set[EntityKey]] = defaultdict(set)

    for path in sorted(pred_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pmid = str(rec["pmid"])
                offset = rec["fulltext_offset"]

                for ent in rec.get("entity", []):
                    # Convert sentence-relative spans to fulltext-absolute
                    abs_spans = frozenset(
                        (s[0] + offset, s[1] + offset)
                        for s in ent["sentence_spans"]
                    )
                    pred[pmid].add((abs_spans, ent["type"]))

    return dict(pred)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    gold: dict[str, set[EntityKey]],
    pred: dict[str, set[EntityKey]],
    type_filter: list[str] | None = None,
) -> dict:
    """Compute micro-averaged P/R/F1 overall and per entity type.

    Args:
        gold:        {pmid: set of EntityKey} from load_gold().
        pred:        {pmid: set of EntityKey} from load_predictions().
        type_filter: if given, only score entities whose type is in this list.

    Returns a dict with keys: overall, per_type, n_documents,
    n_gold_entities, n_pred_entities.
    """
    all_pmids = set(gold) | set(pred)

    # Per-type counters: {type: {"tp": int, "fp": int, "fn": int}}
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pmid in all_pmids:
        g = gold.get(pmid, set())
        p = pred.get(pmid, set())

        if type_filter:
            g = {e for e in g if e[1] in type_filter}
            p = {e for e in p if e[1] in type_filter}

        for ent in g & p:
            type_counts[ent[1]]["tp"] += 1
        for ent in p - g:
            type_counts[ent[1]]["fp"] += 1
        for ent in g - p:
            type_counts[ent[1]]["fn"] += 1

    # Per-type metrics
    per_type: dict[str, dict] = {}
    for ent_type, c in sorted(type_counts.items()):
        per_type[ent_type] = _prf(c["tp"], c["fp"], c["fn"])
        per_type[ent_type].update(c)

    # Micro overall
    total_tp = sum(c["tp"] for c in type_counts.values())
    total_fp = sum(c["fp"] for c in type_counts.values())
    total_fn = sum(c["fn"] for c in type_counts.values())
    overall = _prf(total_tp, total_fp, total_fn)
    overall.update({"tp": total_tp, "fp": total_fp, "fn": total_fn})

    # Summary counts
    n_gold = sum(len(v) for v in gold.values())
    n_pred = sum(len(v) for v in pred.values())
    if type_filter:
        n_gold = sum(len({e for e in v if e[1] in type_filter}) for v in gold.values())
        n_pred = sum(len({e for e in v if e[1] in type_filter}) for v in pred.values())

    return {
        "overall": overall,
        "per_type": per_type,
        "n_documents": len(all_pmids),
        "n_gold_entities": n_gold,
        "n_pred_entities": n_pred,
    }


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results(results: dict) -> None:
    overall = results["overall"]
    print(f"\n{'='*60}")
    print(f"  Documents : {results['n_documents']}")
    print(f"  Gold      : {results['n_gold_entities']} entities")
    print(f"  Predicted : {results['n_pred_entities']} entities")
    print(f"{'='*60}")
    print(f"  {'Type':<30} {'P':>7} {'R':>7} {'F1':>7}  {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7}  {'-'*6} {'-'*6} {'-'*6}")
    for ent_type, m in results["per_type"].items():
        print(f"  {ent_type:<30} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}"
              f"  {m['tp']:>6} {m['fp']:>6} {m['fn']:>6}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7}  {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'OVERALL (micro)':<30} {overall['precision']:>7.4f} {overall['recall']:>7.4f}"
          f" {overall['f1']:>7.4f}  {overall['tp']:>6} {overall['fp']:>6} {overall['fn']:>6}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate step06 predictions against gold annotation JSON files."
    )
    parser.add_argument(
        "--gold", required=True,
        help="Directory containing gold annotation *.json files.",
    )
    parser.add_argument(
        "--pred", required=True,
        help="Directory containing step06 PostprocessRecord *.jsonl files.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write metrics JSON (e.g. results/metrics.json).",
    )
    parser.add_argument(
        "--type-filter", nargs="+", default=None, metavar="TYPE",
        help="Score only these entity types (e.g. --type-filter Chemical Disease).",
    )
    args = parser.parse_args()

    gold = load_gold(Path(args.gold))
    pred = load_predictions(Path(args.pred))
    results = evaluate(gold, pred, type_filter=args.type_filter)

    print_results(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
