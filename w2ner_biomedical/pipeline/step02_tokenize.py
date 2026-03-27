# =============================================================================
# pipeline/step02_tokenize.py
#
# PURPOSE
#   Split each document into sentences, then into word-level tokens, then
#   into subword pieces (BioBERT tokenizer), and finally into length-bounded
#   sentence chunks that fit within the model's max_length.  Emits one
#   TokenRecord per chunk as JSONL.
#
# CORRESPONDS TO
#   step03_tokenize.py  +  step04_apply_token_limit.py  (original W2NER)
#
# KEY DESIGN CHANGES (ARCHITECTURE)
#   - Sentence splitter: spaCy's statistical senter component replaces the
#     LitcoinSentenceTokenizer (which was a company-internal dependency).
#     The senter is loaded via en_core_sci_sm with only ["senter"] enabled
#     so no tagger, parser, or NER overhead is incurred.
#
#   - Single-pass chunking replaces the original two-pass approach:
#     step03 used a heuristic "-6 buffer" for rough chunking; step04 re-
#     tokenized each chunk against a hard max_length=500 limit.  Word
#     boundaries could differ between passes, misaligning character spans.
#     This file merges both steps: tokenize all words once, walk word_ids()
#     once, greedily commit chunks at word boundaries.
#
#   - Fulltext-absolute offsets throughout:
#     All character spans in TokenRecord.spans are relative to
#     IngestRecord.fulltext (already NFKC-normalised in step01).  No
#     per-sentence rebasing, no second normalisation pass.
#
#   - Document-level parallelism via ProcessPoolExecutor:
#     spaCy and the HuggingFace tokenizer are not safe to share across
#     processes.  Workers load their own copies via an initializer function.
#     The main process distributes one document per Future.
#
# KEY DESIGN CHANGES (OFFSET HANDLING)
#   By consuming IngestRecord.fulltext (normalised once in step01), this
#   step operates on the same string coordinate system as the annotation
#   loader.  No second normalisation occurs here.
#
# BUGS ADDRESSED
#   [HIGH]   Two-pass chunking misaligns word boundaries (original step03+04):
#            Eliminated by merging into a single greedy pass over word_ids().
#   [HIGH]   Unicode normalisation shifts char offsets (Bug B):
#            Consumed fulltext is pre-normalised; no re-normalisation here.
# =============================================================================

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import regex

from myutils import load_jsonl, save_jsonl, get_logger

from ..specs.schemas import IngestRecord, TokenRecord
from ..guards.validators import check_record_count_parity
from ._utils import file_sha256, write_stage_manifest, build_base_parser

LOGGER: logging.Logger = logging.getLogger(__name__)

# spaCy model to load for sentence segmentation.  en_core_sci_sm contains a
# statistical senter trained on biomedical text.
_SPACY_MODEL: str = "en_core_sci_sm"

# Regex pattern for split_punct — further splits spaCy tokens at punctuation
# boundaries.  Identical to the original step03_tokenize.py pattern.
_SPLIT_PUNCT_PATTERN = regex.compile(
    r"[\w']+"
    r"|[\ue000-\uf8ff\u0084\u0086\u0088\u0089\u0090\u0095\u0097"
    r"\u202b\u202d\u2060\u2062\u2063\u2fff]"
    r"|[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]"
    r"|\p{P}|\p{S}"
)

# Worker-process globals — set by _worker_init, used by _process_document_worker.
_NLP = None
_TOKENIZER = None


# ---------------------------------------------------------------------------
# Worker initializer (called once per worker process)
# ---------------------------------------------------------------------------

def _worker_init(bert_name: str, cache_dir: str) -> None:
    """Load spaCy and HuggingFace tokenizer once per worker process.

    Called by ProcessPoolExecutor as its initializer, so models are loaded
    exactly once per worker rather than once per document.
    """
    global _NLP, _TOKENIZER
    import spacy
    from transformers import AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _NLP = spacy.load(_SPACY_MODEL, enable=["senter"])
    _TOKENIZER = AutoTokenizer.from_pretrained(bert_name, cache_dir=cache_dir)


def _process_document_worker(record_dict: dict, max_length: int) -> list[dict]:
    """Tokenize one document in a worker process.

    Uses module-level _NLP and _TOKENIZER set by _worker_init.
    Returns a list of TokenRecord dicts (model_dump) ready for JSONL write.
    """
    record = IngestRecord.model_validate(record_dict)
    token_records = tokenize_document(record, _NLP, _TOKENIZER, max_length)
    return [r.model_dump() for r in token_records]


# ---------------------------------------------------------------------------
# Core tokenization functions
# ---------------------------------------------------------------------------

def split_punct(text: str, char_offset: int):
    """Yield (token_str, start_char, end_char) for each sub-token in text.

    Further splits spaCy tokens at punctuation boundaries using the same
    regex pattern as the original step03_tokenize.py.  char_offset makes
    the yielded positions fulltext-absolute.
    """
    for match in _SPLIT_PUNCT_PATTERN.finditer(text):
        yield match.group(), match.start() + char_offset, match.end() + char_offset


def is_supported_language(
    tokenizer,
    words: list[str],
    threshold: float = 0.05,
) -> bool:
    """Return True if the word list appears to be in a tokenizer-supported language.

    Checks the ratio of UNK tokens in the flat word string.  A ratio above
    threshold (default 5%) signals a language the BioBERT tokenizer cannot
    handle (e.g. Chinese, Japanese) and the document should be skipped.

    Raises NotImplementedError if the tokenizer has no unk_token_id (cannot
    perform the check).
    """
    unk_token_id = tokenizer.unk_token_id
    if unk_token_id is None:
        raise NotImplementedError(
            "Tokenizer has no unk_token_id; cannot check language support."
        )
    text = " ".join(words)
    encoded = tokenizer(text, add_special_tokens=False, is_split_into_words=False)
    input_ids = encoded["input_ids"]
    if not input_ids:
        return True
    unk_rate = input_ids.count(unk_token_id) / len(input_ids)
    if unk_rate > threshold:
        LOGGER.warning(
            "Unsupported language detected (%.1f%% UNK tokens in document).",
            unk_rate * 100,
        )
        return False
    return True


def chunk_words(
    words: list[str],
    spans: list[tuple[int, int]],
    tokenizer,
    max_length: int = 500,
) -> list[tuple[list[str], list[tuple[int, int]]]]:
    """Greedily split a word list into chunks fitting within max_length subword tokens.

    Algorithm:
      1. Tokenize all words at once with add_special_tokens=False, truncation=False.
      2. Count subword pieces per word from word_ids().
      3. Walk words left-to-right.  When adding the next word would push the
         total (pieces + 2 for CLS + SEP) over max_length, commit the current
         chunk and start a new one.
      4. A word whose pieces alone exceed max_length - 2 is emitted as a
         single-word chunk with a WARNING (cannot be split further).

    Returns a list of (words_chunk, spans_chunk) pairs.  spans are fulltext-
    absolute tuples.  The list is never empty if words is non-empty.
    """
    if not words:
        return []

    # Tokenize without special tokens or truncation to get raw piece counts
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    word_ids = encoded.word_ids()

    # Count pieces per word (words that produce 0 pieces get 1 to avoid empty chunks)
    pieces_per_word: list[int] = [0] * len(words)
    for wid in word_ids:
        if wid is not None:
            pieces_per_word[wid] += 1
    pieces_per_word = [max(1, p) for p in pieces_per_word]

    chunks: list[tuple[list[str], list[tuple[int, int]]]] = []
    chunk_start = 0
    cumulative = 0

    for word_idx, n_pieces in enumerate(pieces_per_word):
        if cumulative + n_pieces + 2 > max_length:
            # Commit everything before this word as a chunk
            if chunk_start < word_idx:
                chunks.append((
                    words[chunk_start:word_idx],
                    spans[chunk_start:word_idx],
                ))
                chunk_start = word_idx
                cumulative = 0

            # Edge case: single word exceeds limit on its own
            if n_pieces + 2 > max_length:
                LOGGER.warning(
                    "Word %r at index %d has %d subword pieces, exceeding "
                    "max_length=%d (including CLS+SEP).  "
                    "Emitting as a single-word chunk.",
                    words[word_idx], word_idx, n_pieces, max_length,
                )
                chunks.append((
                    words[word_idx:word_idx + 1],
                    spans[word_idx:word_idx + 1],
                ))
                chunk_start = word_idx + 1
                cumulative = 0
                continue

        cumulative += n_pieces

    # Commit the final chunk
    if chunk_start < len(words):
        chunks.append((words[chunk_start:], spans[chunk_start:]))

    return chunks


def tokenize_document(
    record: IngestRecord,
    nlp,
    tokenizer,
    max_length: int = 500,
) -> list[TokenRecord]:
    """Full tokenization pipeline for one IngestRecord.

    Runs: spaCy senter -> split_punct word tokenization -> language check
    -> greedy chunking -> TokenRecord emission.

    Returns an empty list if the document fails the language support check.
    """
    fulltext = record.fulltext
    doc = nlp(fulltext)

    # Collect all words across the document for the language check
    all_words: list[str] = []
    for token in doc:
        for word, _, _ in split_punct(token.text, token.idx):
            all_words.append(word)

    if not all_words:
        LOGGER.warning("PMID %s: document produced no words after tokenization.", record.pmid)
        return []

    if not is_supported_language(tokenizer, all_words):
        LOGGER.warning("PMID %s: skipping document due to unsupported language.", record.pmid)
        return []

    token_records: list[TokenRecord] = []

    for sent_idx, sent in enumerate(doc.sents):
        words: list[str] = []
        word_spans: list[tuple[int, int]] = []

        for token in sent:
            # split_punct yields fulltext-absolute (start, end) because
            # token.idx is already relative to the full doc string.
            for word, start, end in split_punct(token.text, token.idx):
                words.append("LONG_WORD" if len(word) > 500 else word)
                word_spans.append((start, end))

        if not words:
            continue

        chunks = chunk_words(words, word_spans, tokenizer, max_length)

        for chunk_idx, (chunk_words_, chunk_spans) in enumerate(chunks):
            token_records.append(TokenRecord(
                id=f"{record.pmid}_{sent_idx}_{chunk_idx}",
                document_id=record.pmid,
                pmid=record.pmid,
                sentence=chunk_words_,
                spans=list(chunk_spans),
                ner=[],
            ))

    return token_records


# ---------------------------------------------------------------------------
# File-level processing and manifest writing
# ---------------------------------------------------------------------------

def _process_file(
    input_path: Path,
    output_dir: Path,
    bert_name: str,
    cache_dir: str,
    max_length: int,
    force: bool,
    workers: int,
    nlp=None,
    tokenizer=None,
) -> int:
    """Tokenize all IngestRecords from input_path; write TokenRecord JSONL.

    When workers > 1, nlp and tokenizer are None (loaded in worker processes).
    When workers == 1, nlp and tokenizer are passed from the caller.

    Returns the number of TokenRecords written, or -1 if skipped.
    """
    import hashlib

    output_path = output_dir / f"{input_path.stem}.jsonl"

    if output_path.exists() and not force:
        LOGGER.info("Skipping %s (output exists; use --force to overwrite)", input_path.stem)
        return -1

    ingest_records = list(load_jsonl(input_path))
    LOGGER.info("Processing %s: %d documents", input_path.name, len(ingest_records))

    # Load step01 manifest for record count parity check
    step01_manifest_path = input_path.with_suffix(input_path.suffix + ".meta.json")
    prev_count: int | None = None
    if step01_manifest_path.exists():
        import json
        prev_data = json.loads(step01_manifest_path.read_text(encoding="utf-8"))
        prev_count = prev_data.get("record_count")

    if workers > 1:
        futures = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(bert_name, cache_dir),
        ) as executor:
            for record_dict in ingest_records:
                futures.append(executor.submit(_process_document_worker, record_dict, max_length))
            token_record_dicts = []
            for future in futures:
                token_record_dicts.extend(future.result())
    else:
        token_record_dicts = []
        for record_dict in ingest_records:
            record = IngestRecord.model_validate(record_dict)
            for tr in tokenize_document(record, nlp, tokenizer, max_length):
                token_record_dicts.append(tr.model_dump())

    # Guard 1: records went up (documents -> chunks) so this is a no-op,
    # but documents the pipeline intent.
    check_record_count_parity(prev_count, len(token_record_dicts), "step02_tokenize")

    # Write atomically
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=output_dir, suffix=".jsonl.tmp")
    try:
        os.close(tmp_fd)
        save_jsonl(token_record_dicts, Path(tmp_path_str))
        os.replace(tmp_path_str, output_path)
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise

    write_stage_manifest(
        output_path=output_path,
        stage="step02_tokenize",
        input_files=[input_path.name],
        input_hash=file_sha256(input_path),
        record_count=len(token_record_dicts),
    )

    LOGGER.info(
        "%s: %d documents -> %d token chunks",
        input_path.name, len(ingest_records), len(token_record_dicts),
    )
    return len(token_record_dicts)


def main() -> None:
    global LOGGER

    parser = build_base_parser("Step 02: tokenize IngestRecord JSONL into TokenRecord JSONL.")
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing step01 IngestRecord *.jsonl files.",
    )
    parser.add_argument(
        "--bert-name", required=True,
        help="HuggingFace model identifier for the subword tokenizer "
             "(e.g. dmis-lab/biobert-base-cased-v1.1).",
    )
    parser.add_argument(
        "--cache-dir", default="cache",
        help="HuggingFace model cache directory (default: cache).",
    )
    parser.add_argument(
        "--max-length", type=int, default=500,
        help="Maximum subword token count per chunk including CLS+SEP (default: 500).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes for document-level tokenization "
             "(default: 1).  Workers > 1 load spaCy and the tokenizer fresh per "
             "process via an initializer.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = get_logger("step02_tokenize", log_dir=output_dir / "logs")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    input_files = sorted(input_dir.glob("*.jsonl"))
    if not input_files:
        LOGGER.warning("No *.jsonl files found in %s", input_dir)
        return

    LOGGER.info(
        "Found %d input files, bert=%s, max_length=%d, workers=%d",
        len(input_files), args.bert_name, args.max_length, args.workers,
    )

    # Single-worker: load models once in the main process
    nlp = None
    tokenizer = None
    if args.workers == 1:
        import spacy
        from transformers import AutoTokenizer
        nlp = spacy.load(_SPACY_MODEL, enable=["senter"])
        tokenizer = AutoTokenizer.from_pretrained(
            args.bert_name, cache_dir=args.cache_dir
        )

    total_chunks = 0
    total_skipped = 0

    for input_file in input_files:
        try:
            n = _process_file(
                input_path=input_file,
                output_dir=output_dir,
                bert_name=args.bert_name,
                cache_dir=args.cache_dir,
                max_length=args.max_length,
                force=args.force,
                workers=args.workers,
                nlp=nlp,
                tokenizer=tokenizer,
            )
            if n == -1:
                total_skipped += 1
            else:
                total_chunks += n
        except Exception as exc:
            LOGGER.error("Failed to process %s: %s", input_file.name, exc, exc_info=True)

    LOGGER.info(
        "Done. %d files processed (%d skipped), %d total token chunks written.",
        len(input_files) - total_skipped, total_skipped, total_chunks,
    )


if __name__ == "__main__":
    main()
