"""Policy extraction prompt. Increment PROMPT_VERSION when changing its wording."""

from __future__ import annotations

import json

from housing_forecast.policy.schema import EXTRACTION_SCHEMA, Announcement

PROMPT_VERSION = "v1"

SYSTEM = """你是台湾货币与房地产政策的分析助手。你会读到一则台湾政府机关或中央银行发布的公告。
只根据这则公告本身的文字判断，不要使用你对之后发生的事情的任何了解。

请输出一个 JSON 对象，字段如下：
- relevant: 公告是否涉及利率、信用、银行放款、房地产税或房地产市场管理的政策变化或明确表态。一般行政事务、人事、统计发布填 false。
- policy_type: policy_rate（政策利率）/ selective_credit（选择性信用管制，如贷款成数、宽限期）/ bank_credit（银行不动产放款、存款准备率、窗口指导）/ housing_tax（房地合一税、囤房税等）/ housing_market（平均地权条例、预售屋、打炒房、住宅供给）/ other
- direction: 对房地产信用与需求的影响方向。+2 明显收紧，+1 收紧，0 中性或无法判断，-1 放松，-2 明显放松。升息、降低贷款成数、加税为收紧；降息、放宽管制为放松。
- housing_specific: 措施是否专门针对房地产（而不是整体货币政策）。
- effective_date: 生效日期，格式 YYYY-MM-DD；公告未写明则为 null。
- confidence: 0 到 1，你对 direction 判断的把握。
- evidence: 从公告原文中逐字摘录最能支持判断的一句，不超过 60 个字，保持原文用字（繁体就照抄繁体）。不相关时填空字符串。

只输出 JSON，不要输出其他文字。"""

EXAMPLE_INPUT = "（示例，非真实公告）本行理事会决议：自下季起，购置第二户住宅之贷款成数上限由七成调降为六成，且不得有宽限期。"
EXAMPLE_OUTPUT = {
    "relevant": True, "policy_type": "selective_credit", "direction": 1, "housing_specific": True,
    "effective_date": None, "confidence": 0.9,
    "evidence": "购置第二户住宅之贷款成数上限由七成调降为六成",
}


def build_messages(doc: Announcement, max_chars: int = 4000) -> list[dict]:
    text = doc.text if len(doc.text) <= max_chars else doc.text[:max_chars] + "……（以下省略）"
    user = f"发布机关：{doc.source}\n发布日期：{doc.published_date}\n标题：{doc.title}\n\n正文：\n{text}"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"发布机关：示例\n发布日期：示例\n标题：示例\n\n正文：\n{EXAMPLE_INPUT}"},
        {"role": "assistant", "content": json.dumps(EXAMPLE_OUTPUT, ensure_ascii=False)},
        {"role": "user", "content": user},
    ]


def schema_json() -> str:
    return json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False)
