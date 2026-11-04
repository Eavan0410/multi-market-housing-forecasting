"""Pipeline checks for the LLM policy index, using the keyword MockBackend (no model, no GPU).
All documents below are made up for testing."""

import json

import numpy as np
import pandas as pd
import pytest

from housing_forecast.policy.backends import Generation, MockBackend
from housing_forecast.policy.evaluate import event_match, quarter_agreement, score_spotcheck, spotcheck_sample
from housing_forecast.policy.extract import _first_json_object, _to_extraction, run_extraction, validate
from housing_forecast.policy.index import extractions_to_events, quarterly_index, timeline_to_events
from housing_forecast.policy.schema import Announcement, read_extractions


def toy_docs():
    return [
        Announcement("d1", "测试机关", "2010-03-20", "测试一", "", "理事会决议升息半码。"),
        Announcement("d2", "测试机关", "2010-06-10", "测试二", "", "决议放宽第二户贷款限制。"),
        Announcement("d3", "测试机关", "2010-07-01", "测试三", "", "本机关新任处长就职。"),
        Announcement("d4", "测试机关", "2011-01-05", "测试四", "", "宣布开征囤房税以抑制投机。"),
    ]


GOOD = {"relevant": True, "policy_type": "policy_rate", "direction": 1, "housing_specific": False,
        "effective_date": None, "confidence": 0.8, "evidence": "升息半码"}


def test_parser_tolerates_fences_and_extra_text():
    assert _first_json_object("```json\n" + json.dumps(GOOD) + "\n```")["direction"] == 1
    assert _first_json_object("好的：" + json.dumps(GOOD) + " 以上")["policy_type"] == "policy_rate"


@pytest.mark.parametrize("bad", [{"direction": 3}, {"direction": 1.5}, {"policy_type": "weather"},
                                 {"confidence": 2}, {"relevant": "yes"}])
def test_validate_rejects_bad_fields(bad):
    with pytest.raises(ValueError):
        validate({**GOOD, **bad})


def test_irrelevant_documents_get_zero_direction():
    assert validate({**GOOD, "relevant": False})["direction"] == 0


def test_bad_output_is_kept_as_failed_record():
    r = _to_extraction(toy_docs()[0], "m", Generation("not json"))
    assert not r.parse_ok and r.raw_output == "not json"


def test_evidence_check():
    doc = toy_docs()[0]
    assert _to_extraction(doc, "m", Generation(json.dumps(GOOD, ensure_ascii=False))).evidence_found
    fake = {**GOOD, "evidence": "降息一码"}
    assert not _to_extraction(doc, "m", Generation(json.dumps(fake, ensure_ascii=False))).evidence_found


def test_extract_end_to_end_and_cache(tmp_path):
    path, run = run_extraction(toy_docs(), MockBackend(), tmp_path, batch_size=2, log=lambda *_: None)
    assert run.n_docs == 4 and run.n_parse_ok == 4
    ex = read_extractions(path)
    assert ex.set_index("doc_id").loc[["d1", "d2", "d3", "d4"], "direction"].tolist() == [1, -1, 0, 1]
    _, run2 = run_extraction(toy_docs(), MockBackend(), tmp_path, log=lambda *_: None)
    assert run2.n_docs == 0                                   # everything cached
    changed = toy_docs()
    changed[0] = Announcement("d1", "测试机关", "2010-03-20", "测试一", "", "理事会决议降息半码。")
    _, run3 = run_extraction(changed, MockBackend(), tmp_path, log=lambda *_: None)
    assert run3.n_docs == 1                                   # changed text is re-run


def test_index_uses_publication_quarter_and_decays(tmp_path):
    path, _ = run_extraction(toy_docs(), MockBackend(), tmp_path, log=lambda *_: None)
    idx = quarterly_index(extractions_to_events(read_extractions(path)), "2010Q1", "2011Q2", half_life=1)
    flow = idx.set_index("quarter")["policy_flow"]
    assert flow["2010Q1"] == pytest.approx(0.6) and flow["2010Q2"] == pytest.approx(-0.6)
    assert flow["2010Q3"] == 0                                 # irrelevant document adds nothing
    stance = idx.set_index("quarter")["policy_stance"]
    assert stance["2010Q2"] == pytest.approx(-0.6 + 0.5 * 0.6)
    assert stance["2010Q4"] == pytest.approx(stance["2010Q3"] * 0.5)


def manual():
    return pd.DataFrame({"event_id": ["E1", "E2", "E3"],
                         "date": pd.to_datetime(["2010-03-25", "2010-06-30", "2012-01-01"]),
                         "policy_type": ["policy_rate", "selective_credit", "policy_rate"],
                         "direction": [1, -1, -1], "housing_specific": [False, True, False],
                         "description": ["", "", ""], "source_url": ["", "", ""]})


def test_event_matching(tmp_path):
    path, _ = run_extraction(toy_docs(), MockBackend(), tmp_path, log=lambda *_: None)
    llm = extractions_to_events(read_extractions(path))
    r = event_match(llm, timeline_to_events(manual()), window_days=14)
    assert r["n_manual_events"] == 3 and r["recall"] == pytest.approx(1 / 3)   # only E1 within 14 days
    assert r["missed_event_ids"] == ["E2", "E3"]
    r30 = event_match(llm, timeline_to_events(manual()), window_days=30)
    assert r30["recall"] == pytest.approx(2 / 3)


def test_quarter_agreement_identical_series():
    idx = quarterly_index(timeline_to_events(manual()), "2010Q1", "2012Q4")
    r = quarter_agreement(idx, idx)
    assert r["pearson"] == pytest.approx(1.0) and r["sign_agreement_active"] == 1.0


def test_spotcheck_roundtrip(tmp_path):
    path, _ = run_extraction(toy_docs(), MockBackend(), tmp_path, log=lambda *_: None)
    docs = pd.DataFrame([d.__dict__ for d in toy_docs()])
    sheet = spotcheck_sample(read_extractions(path), docs, n=4)
    assert len(sheet) == 4 and (sheet["human_relevant"] == "").all()
    sheet["human_relevant"] = sheet["llm_relevant"]
    sheet["human_direction"] = sheet["llm_direction"]
    sheet["human_policy_type"] = sheet["llm_policy_type"]
    r = score_spotcheck(sheet)
    assert r["relevance_accuracy"] == 1.0 and r["direction_exact_accuracy"] == 1.0
