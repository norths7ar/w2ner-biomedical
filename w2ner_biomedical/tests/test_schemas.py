# =============================================================================
# tests/test_schemas.py
#
# Unit tests for specs/schemas.py.
# Tests focus on the two model validators (IngestRecord fulltext consistency,
# LabelSpec sentinel check) and the label2id / id2label helpers.
# =============================================================================

from __future__ import annotations

import pytest
from pydantic import ValidationError

from w2ner_biomedical.specs.schemas import IngestRecord, LabelSpec, TokenRecord, NEREntry


# ---------------------------------------------------------------------------
# IngestRecord
# ---------------------------------------------------------------------------

class TestIngestRecord:
    def test_valid_title_and_abstract(self):
        rec = IngestRecord(
            pmid="12345",
            title="Hello world.",
            abstract="Some abstract text.",
            fulltext="Hello world. Some abstract text.",
        )
        assert rec.fulltext == "Hello world. Some abstract text."

    def test_fulltext_inconsistent_raises(self):
        with pytest.raises(ValidationError, match="inconsistent"):
            IngestRecord(
                pmid="12345",
                title="Hello world.",
                abstract="Some abstract text.",
                fulltext="Wrong text.",
            )

    def test_empty_title_fulltext_is_abstract_only(self):
        rec = IngestRecord(pmid="1", title="", abstract="Only abstract.", fulltext="Only abstract.")
        assert rec.fulltext == "Only abstract."

    def test_empty_abstract_fulltext_is_title_only(self):
        rec = IngestRecord(pmid="1", title="Only title.", abstract="", fulltext="Only title.")
        assert rec.fulltext == "Only title."

    def test_unicode_content_roundtrips(self):
        title = "Hépatocyte nucléaire."
        abstract = "Résultats significatifs."
        rec = IngestRecord(pmid="99", title=title, abstract=abstract,
                           fulltext=f"{title} {abstract}")
        assert rec.fulltext == f"{title} {abstract}"

    def test_trailing_space_in_fulltext_raises(self):
        with pytest.raises(ValidationError):
            IngestRecord(pmid="1", title="Title.", abstract="Abstract.",
                         fulltext="Title. Abstract. ")   # trailing space


# ---------------------------------------------------------------------------
# LabelSpec
# ---------------------------------------------------------------------------

class TestLabelSpec:
    def _spec(self, entity_types, **kw):
        return LabelSpec.model_validate({"entity_types": entity_types, **kw})

    def test_valid_spec(self):
        spec = self._spec(["Chemical", "Disease", "Species"])
        assert spec.sentinels == ["<pad>", "<suc>"]
        assert spec.entity_types == ["Chemical", "Disease", "Species"]

    def test_wrong_sentinels_raises(self):
        with pytest.raises(ValidationError, match="sentinels"):
            LabelSpec.model_validate({"entity_types": ["Chemical"],
                                      "sentinels": ["<pad>"]})   # missing <suc>

    def test_reversed_sentinels_raises(self):
        with pytest.raises(ValidationError, match="sentinels"):
            LabelSpec.model_validate({"entity_types": ["Chemical"],
                                      "sentinels": ["<suc>", "<pad>"]})

    def test_label2id_sentinel_ids(self):
        spec = self._spec(["Chemical", "Disease"])
        l2i = spec.label2id()
        assert l2i["<pad>"] == 0
        assert l2i["<suc>"] == 1

    def test_label2id_entity_ids_start_at_2(self):
        spec = self._spec(["Chemical", "Disease", "Species"])
        l2i = spec.label2id()
        assert l2i["Chemical"] == 2
        assert l2i["Disease"]  == 3
        assert l2i["Species"]  == 4

    def test_label2id_total_size(self):
        spec = self._spec(["Chemical", "Disease"])
        assert len(spec.label2id()) == 4   # 2 sentinels + 2 entity types

    def test_id2label_is_exact_inverse(self):
        spec = self._spec(["Chemical", "Disease"])
        l2i = spec.label2id()
        i2l = spec.id2label()
        assert all(i2l[v] == k for k, v in l2i.items())

    def test_entity_type_order_preserved(self):
        types = ["Species", "Chemical", "Disease", "CellLine"]
        spec = self._spec(types)
        l2i = spec.label2id()
        for offset, t in enumerate(types):
            assert l2i[t] == offset + 2   # 2 sentinel slots

    def test_aliases_loaded(self):
        spec = LabelSpec.model_validate({
            "entity_types": ["Chemical"],
            "aliases": {"ChemicalEntity": "Chemical"},
        })
        assert spec.aliases["ChemicalEntity"] == "Chemical"


# ---------------------------------------------------------------------------
# TokenRecord
# ---------------------------------------------------------------------------

class TestTokenRecord:
    def test_valid_record_default_empty_ner(self):
        rec = TokenRecord(
            id="12345_0_0", document_id="doc1", pmid="12345",
            sentence=["Hello", "world"], spans=[(0, 5), (6, 11)],
        )
        assert rec.ner == []

    def test_ner_entry_attached(self):
        rec = TokenRecord(
            id="12345_0_0", document_id="doc1", pmid="12345",
            sentence=["Hello", "world"], spans=[(0, 5), (6, 11)],
            ner=[NEREntry(indices=[0, 1], type="Chemical")],
        )
        assert len(rec.ner) == 1
        assert rec.ner[0].type == "Chemical"
        assert rec.ner[0].indices == [0, 1]
