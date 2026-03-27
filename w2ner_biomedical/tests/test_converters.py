# =============================================================================
# tests/test_converters.py
#
# Tests for converters/_bioc_utils.py, biored_to_schema.py, bc5cdr_to_schema.py.
#
# Structure:
#   Unit tests — pure-function tests with synthetic mock objects, no file I/O.
#   Integration tests — marked with pytest.mark.integration; load real corpus
#     files from data/ and are automatically skipped when files are absent.
#
# Mock objects mirror the minimal BioC object interface used by the converters:
#   bioc_document.id, .passages[]
#   passage.infons["type"], .offset, .text, .annotations[]
#   annotation.id, .infons["type"], .locations[]
#   location.offset, .length
# =============================================================================

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from w2ner_biomedical.converters._bioc_utils import (
    bioc_offset_to_fulltext_offset,
    extract_passages,
    validate_bioc_offsets,
)
from w2ner_biomedical.converters.biored_to_schema import (
    BIORED_TYPE_MAP,
    convert_annotation as biored_convert_annotation,
    convert_document as biored_convert_document,
)
from w2ner_biomedical.converters.bc5cdr_to_schema import (
    BC5CDR_TYPE_MAP,
    convert_annotation as bc5cdr_convert_annotation,
    convert_document as bc5cdr_convert_document,
)


# ---------------------------------------------------------------------------
# Minimal BioC mock helpers
# ---------------------------------------------------------------------------

def _loc(offset: int, length: int):
    return SimpleNamespace(offset=offset, length=length)


def _ann(id_: str, type_: str, locations):
    return SimpleNamespace(
        id=id_,
        infons={"type": type_},
        locations=[_loc(*l) for l in locations],
    )


def _passage(type_: str, offset: int, text: str, annotations=None):
    return SimpleNamespace(
        infons={"type": type_},
        offset=offset,
        text=text,
        annotations=annotations or [],
    )


def _document(id_: str, passages):
    return SimpleNamespace(id=id_, passages=passages)


def _simple_doc(title="Title text.", abstract="Abstract text.", extra_ann=None):
    """Build a minimal 2-passage BioC document (standard layout)."""
    title_len = len(title)
    # BioC convention: abstract starts at title_len + 1 (single space separator)
    abstract_offset = title_len + 1
    title_passage = _passage("title", 0, title)
    abstract_passage = _passage("abstract", abstract_offset, abstract,
                                annotations=extra_ann or [])
    return _document("99999", [title_passage, abstract_passage])


# ---------------------------------------------------------------------------
# bioc_offset_to_fulltext_offset — pure function unit tests
# ---------------------------------------------------------------------------

class TestBiocOffsetToFulltextOffset:
    def test_title_region_unchanged(self):
        # offset < abstract_bioc_offset → no rebasing
        assert bioc_offset_to_fulltext_offset(5, title_len=10, abstract_bioc_offset=11) == 5

    def test_abstract_region_rebased(self):
        # title_len=10, abstract_bioc_offset=11, sep_width=1 (standard)
        # bioc_offset=15 → pos_within_abstract=4 → fulltext=(10+1)+4=15
        assert bioc_offset_to_fulltext_offset(15, title_len=10, abstract_bioc_offset=11) == 15

    def test_abstract_start_maps_to_title_len_plus_1(self):
        # The very start of the abstract should map to title_len+1
        assert bioc_offset_to_fulltext_offset(11, title_len=10, abstract_bioc_offset=11) == 11

    def test_wide_separator_rebased_correctly(self):
        # title_len=10, abstract_bioc_offset=13 (2-char separator)
        # bioc_offset=15 → pos_within_abstract=15-13=2 → fulltext=(10+1)+2=13
        assert bioc_offset_to_fulltext_offset(15, title_len=10, abstract_bioc_offset=13) == 13

    def test_zero_offset_in_title(self):
        assert bioc_offset_to_fulltext_offset(0, title_len=20, abstract_bioc_offset=21) == 0


# ---------------------------------------------------------------------------
# validate_bioc_offsets — unit tests
# ---------------------------------------------------------------------------

class TestValidateBiocOffsets:
    def test_valid_title_at_offset_zero(self):
        doc = _document("1", [_passage("title", 0, "Title."),
                                _passage("abstract", 7, "Abstract.")])
        validate_bioc_offsets(doc)   # should not raise

    def test_title_not_at_zero_raises(self):
        doc = _document("1", [_passage("title", 5, "Title."),
                                _passage("abstract", 12, "Abstract.")])
        with pytest.raises(ValueError, match="offset 0"):
            validate_bioc_offsets(doc)

    def test_no_title_passage_raises(self):
        doc = _document("1", [_passage("abstract", 0, "Abstract.")])
        with pytest.raises(ValueError, match="no title"):
            validate_bioc_offsets(doc)


# ---------------------------------------------------------------------------
# extract_passages — unit tests
# ---------------------------------------------------------------------------

class TestExtractPassages:
    def test_standard_layout(self):
        title = "My title."
        abstract = "My abstract."
        doc = _simple_doc(title, abstract)
        t, a, abs_off = extract_passages(doc)
        assert t == title
        assert a == abstract
        assert abs_off == len(title) + 1

    def test_missing_abstract_raises(self):
        doc = _document("1", [_passage("title", 0, "Title.")])
        with pytest.raises(ValueError, match="no abstract"):
            extract_passages(doc)

    def test_missing_title_raises(self):
        doc = _document("1", [_passage("abstract", 10, "Abstract.")])
        with pytest.raises(ValueError, match="no title"):
            extract_passages(doc)

    def test_wide_separator_logs_warning(self, caplog):
        import logging
        # abstract starts at offset 15, title is only 10 chars → sep_width=5
        title = "0123456789"   # len=10
        abstract_offset = 15
        doc = _document("1", [
            _passage("title", 0, title),
            _passage("abstract", abstract_offset, "Abstract."),
        ])
        with caplog.at_level(logging.WARNING, logger="w2ner_biomedical.converters._bioc_utils"):
            t, a, off = extract_passages(doc)
        assert any("separator" in r.message.lower() for r in caplog.records)
        assert off == abstract_offset


# ---------------------------------------------------------------------------
# BioRED convert_annotation — unit tests
# ---------------------------------------------------------------------------

class TestBioredConvertAnnotation:
    def test_known_type_maps_correctly(self):
        # BioC location: offset=5, length=5 → span [5, 10]
        ann = _ann("A1", "ChemicalEntity", [(5, 5)])
        result = biored_convert_annotation(ann, title_len=10, abstract_bioc_offset=11)
        assert result is not None
        spans_list, mapped_type = result
        assert mapped_type == "Chemical"
        assert spans_list == [[5, 10]]

    def test_all_six_types_mapped(self):
        for raw, expected in BIORED_TYPE_MAP.items():
            ann = _ann("A1", raw, [(0, 5)])   # offset=0, length=5 → [0, 5]
            result = biored_convert_annotation(ann, title_len=10, abstract_bioc_offset=11)
            assert result is not None
            assert result[1] == expected, f"Type {raw!r} should map to {expected!r}."

    def test_unknown_type_returns_none(self):
        ann = _ann("A1", "SomeUnknownType", [(0, 5)])
        result = biored_convert_annotation(ann, title_len=10, abstract_bioc_offset=11)
        assert result is None

    def test_degenerate_span_returns_none(self):
        # offset=5, length=0 → start==end → degenerate
        ann = _ann("A1", "ChemicalEntity", [(5, 0)])
        result = biored_convert_annotation(ann, title_len=10, abstract_bioc_offset=11)
        assert result is None

    def test_abstract_offset_rebased(self):
        # title_len=10, abstract_bioc_offset=11 (single-space separator)
        # bioc_offset=12, length=5 → pos_in_abstract=1 → fulltext=[12, 17]
        ann = _ann("A1", "ChemicalEntity", [(12, 5)])
        result = biored_convert_annotation(ann, title_len=10, abstract_bioc_offset=11)
        assert result is not None
        spans, _ = result
        assert spans == [[12, 17]]


# ---------------------------------------------------------------------------
# BC5CDR convert_annotation — unit tests
# ---------------------------------------------------------------------------

class TestBc5cdrConvertAnnotation:
    def test_chemical_maps_correctly(self):
        ann = _ann("A1", "Chemical", [(0, 8)])
        result = bc5cdr_convert_annotation(ann, title_len=20, abstract_bioc_offset=21)
        assert result is not None
        spans, type_ = result
        assert type_ == "Chemical"
        assert spans == [[0, 8]]

    def test_disease_maps_correctly(self):
        ann = _ann("A1", "Disease", [(10, 7)])
        result = bc5cdr_convert_annotation(ann, title_len=20, abstract_bioc_offset=21)
        assert result is not None
        assert result[1] == "Disease"

    def test_unknown_type_returns_none(self):
        ann = _ann("A1", "Mutation", [(0, 5)])
        result = bc5cdr_convert_annotation(ann, title_len=20, abstract_bioc_offset=21)
        assert result is None

    def test_multi_location_discontinuous(self):
        # Two BioC locations: (offset, length)
        # title_len=20, abstract_bioc_offset=21
        # loc1: offset=5, length=5  → span [5, 10]  (in title region, no rebase)
        # loc2: offset=25, length=4 → pos_in_abstract=25-21=4 → fulltext=[(20+1)+4, (20+1)+8]=[25,29]
        ann = _ann("A1", "Disease", [(5, 5), (25, 4)])
        result = bc5cdr_convert_annotation(ann, title_len=20, abstract_bioc_offset=21)
        assert result is not None
        spans, type_ = result
        assert len(spans) == 2, "Multi-location annotation must produce two spans."
        assert spans[0] == [5, 10]
        assert spans[1] == [25, 29]


# ---------------------------------------------------------------------------
# BioRED convert_document — unit tests with synthetic documents
# ---------------------------------------------------------------------------

class TestBioredConvertDocument:
    def test_minimal_document(self):
        ann = _ann("A1", "ChemicalEntity", [(0, 5)])
        doc = _simple_doc("Title text.", "Abstract.", extra_ann=[ann])
        rec = biored_convert_document(doc)
        assert rec is not None
        assert rec["PMID"] == "99999"
        assert rec["articleTitle"] == "Title text."
        assert len(rec["label"]) == 1
        assert rec["label"][0][1] == "Chemical"

    def test_no_annotations_produces_empty_label(self):
        doc = _simple_doc()
        rec = biored_convert_document(doc)
        assert rec is not None
        assert rec["label"] == []

    def test_unknown_type_annotation_skipped(self):
        ann = _ann("A1", "UnknownType", [(0, 5)])
        doc = _simple_doc(extra_ann=[ann])
        rec = biored_convert_document(doc)
        assert rec is not None
        assert rec["label"] == []

    def test_bad_document_returns_none(self):
        # Document with no title passage
        doc = _document("bad", [_passage("abstract", 0, "Abstract.")])
        rec = biored_convert_document(doc)
        assert rec is None


# ---------------------------------------------------------------------------
# BC5CDR convert_document — unit tests
# ---------------------------------------------------------------------------

class TestBc5cdrConvertDocument:
    def test_minimal_document(self):
        ann = _ann("A1", "Chemical", [(0, 8)])
        doc = _simple_doc("Naloxone reversal.", "Abstract text.", extra_ann=[ann])
        rec = bc5cdr_convert_document(doc)
        assert rec is not None
        assert rec["PMID"] == "99999"
        assert len(rec["label"]) == 1
        assert rec["label"][0][1] == "Chemical"

    def test_discontinuous_label_preserved(self):
        # Two-location Disease annotation
        ann = _ann("A1", "Disease", [(0, 6), (10, 8)])
        doc = _simple_doc("Title text full sentence here.", "More text.", extra_ann=[ann])
        rec = bc5cdr_convert_document(doc)
        assert rec is not None
        label = rec["label"][0]
        assert len(label[0]) == 2, "Discontinuous label must have 2 spans."


# ---------------------------------------------------------------------------
# Integration tests — real data files (skipped if absent)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR  = REPO_ROOT / "data"


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Data file not found: {path}")


@pytest.mark.integration
class TestBioredIntegration:
    def test_train_document_count(self):
        import bioc
        path = DATA_DIR / "BioRED" / "Train.BioC.XML"
        _skip_if_missing(path)
        with open(path, encoding="utf-8") as f:
            collection = bioc.load(f)
        records = [biored_convert_document(d) for d in collection.documents]
        records = [r for r in records if r is not None]
        assert len(records) == 400, f"BioRED train: expected 400, got {len(records)}."

    def test_train_all_types_known(self):
        import bioc
        path = DATA_DIR / "BioRED" / "Train.BioC.XML"
        _skip_if_missing(path)
        with open(path, encoding="utf-8") as f:
            collection = bioc.load(f)
        observed_types = set()
        for doc in collection.documents:
            rec = biored_convert_document(doc)
            if rec:
                for spans, type_ in rec["label"]:
                    observed_types.add(type_)
        unknown = observed_types - set(BIORED_TYPE_MAP.values())
        assert not unknown, f"Unexpected types in BioRED train output: {unknown}"

    def test_train_no_multi_location_annotations(self):
        """BioRED has no discontinuous annotations — all labels have exactly one span."""
        import bioc
        path = DATA_DIR / "BioRED" / "Train.BioC.XML"
        _skip_if_missing(path)
        with open(path, encoding="utf-8") as f:
            collection = bioc.load(f)
        multi_span = []
        for doc in collection.documents:
            rec = biored_convert_document(doc)
            if rec:
                for spans, _ in rec["label"]:
                    if len(spans) > 1:
                        multi_span.append((rec["PMID"], spans))
        assert not multi_span, (
            f"BioRED should have no multi-location annotations, found {len(multi_span)}."
        )


@pytest.mark.integration
class TestBc5cdrIntegration:
    def test_train_document_count(self):
        import bioc
        path = DATA_DIR / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml"
        _skip_if_missing(path)
        with open(path, encoding="utf-8") as f:
            collection = bioc.load(f)
        records = [bc5cdr_convert_document(d) for d in collection.documents]
        records = [r for r in records if r is not None]
        assert len(records) == 500, f"BC5CDR train: expected 500, got {len(records)}."

    def test_train_all_types_known(self):
        import bioc
        path = DATA_DIR / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml"
        _skip_if_missing(path)
        with open(path, encoding="utf-8") as f:
            collection = bioc.load(f)
        observed_types = set()
        for doc in collection.documents:
            rec = bc5cdr_convert_document(doc)
            if rec:
                for spans, type_ in rec["label"]:
                    observed_types.add(type_)
        unknown = observed_types - set(BC5CDR_TYPE_MAP.values())
        assert not unknown, f"Unexpected types in BC5CDR train output: {unknown}"

    def test_train_multi_location_count(self):
        """BC5CDR training set contains 73 multi-location annotations."""
        import bioc
        path = DATA_DIR / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml"
        _skip_if_missing(path)
        with open(path, encoding="utf-8") as f:
            collection = bioc.load(f)
        multi_count = 0
        for doc in collection.documents:
            rec = bc5cdr_convert_document(doc)
            if rec:
                for spans, _ in rec["label"]:
                    if len(spans) > 1:
                        multi_count += 1
        assert multi_count == 73, (
            f"BC5CDR train: expected 73 multi-location annotations, got {multi_count}."
        )

    def test_first_pmid_is_227508(self):
        """Spot-check: first document in the training file is PMID 227508."""
        import bioc
        path = DATA_DIR / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml"
        _skip_if_missing(path)
        with open(path, encoding="utf-8") as f:
            collection = bioc.load(f)
        first_rec = bc5cdr_convert_document(collection.documents[0])
        assert first_rec is not None
        assert first_rec["PMID"] == "227508"

    def test_first_doc_naloxone_annotation(self):
        """PMID 227508: 'Naloxone' at [0, 8] is the first Chemical annotation."""
        import bioc
        path = DATA_DIR / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TrainingSet.BioC.xml"
        _skip_if_missing(path)
        with open(path, encoding="utf-8") as f:
            collection = bioc.load(f)
        rec = bc5cdr_convert_document(collection.documents[0])
        assert rec is not None
        fulltext = rec["articleTitle"] + " " + rec["abstract"]
        chemicals = [(spans, t) for spans, t in rec["label"] if t == "Chemical"]
        assert chemicals, "No Chemical annotations in PMID 227508."
        # 'Naloxone' is the title's first word at offset [0,8]
        first_spans = chemicals[0][0]
        assert first_spans == [[0, 8]], (
            f"Expected first Chemical span [[0,8]], got {first_spans}. "
            f"Title starts with: {fulltext[:20]!r}"
        )
