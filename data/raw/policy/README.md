# Policy data (C collects, A's pipeline reads)

## announcements.jsonl

One JSON object per line, one line per announcement (central bank, Ministry of Finance, Ministry of the Interior, Financial Supervisory Commission):

```json
{"doc_id": "cbc-2023-06-15", "source": "中央銀行", "published_date": "2023-06-15", "title": "...", "url": "https://...", "text": "全文..."}
```

- `doc_id`: unique, stable; never reuse an id for a different text.
- `published_date`: the day the text was first made public (YYYY-MM-DD). This decides which quarter the document counts in, so it must not be a later revision date.
- `text`: the original wording (traditional Chinese is fine); keep the full text, not a summary.

A CSV with the same columns also works, but JSONL keeps line breaks inside the text safely.

## manual_timeline.csv (gold standard)

Copy `manual_timeline_template.csv` and fill one row per policy event:

| column | values |
|---|---|
| `event_id` | unique id, e.g. `E001` |
| `date` | announcement date, YYYY-MM-DD |
| `policy_type` | `policy_rate`, `selective_credit`, `bank_credit`, `housing_tax`, `housing_market`, `other` |
| `direction` | +2 strong tightening, +1 tightening, 0 neutral, -1 easing, -2 strong easing |
| `housing_specific` | true / false |
| `description` | one short sentence |
| `source_url` | link to the original announcement |

Fill the timeline from the announcements themselves, without looking at the LLM output, so it stays an independent check.
