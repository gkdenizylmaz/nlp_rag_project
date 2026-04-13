"""
CENG493 RAG: ortak QA / atıf / faithfulness metrikleri ve LLM baglam formati.
Notebook'lar: sys.path.insert(0, str(PROJECT_DIR)); import evaluation.metrics_utils as emu
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

# Ödev: grounded, source-supported, citation consistency
RAG_SYSTEM_PROMPT = """Sen bir Turk hukuku uzmanisin. Yalnizca asagida verilen baglam parcalarindaki bilgiyi kullan.

Kurallar (KESINLIKLE):
1) Once soruyu 1-3 cumle ile yanitla. Ilk paragraf dogrudan cevap olsun; istersen basina "Cevap:" yazabilirsin.
2) Bos satir birak, sonra tam olarak "Kaynaklar:" basligini yaz (iki nokta ust uste ile).
3) Her kullandigin parca icin ayri satirda su formatta yaz:
   - [chunk_id=PARCA_ID] KaynakAdi — en fazla 15 kelimelik kisa alinti
   PARCA_ID degerini baglam blogunun basindaki [chunk_id=...] satirindan AYNEN kopyala.
4) Baglamda cevap yoksa sadece soyle: "Bu bilgi verilen kaynaklarda bulunmamaktadir." ve "Kaynaklar:" bolumunu bos birak veya yazma.
5) Baglamda olmayan kanun, madde veya bilgi uydurma."""


def format_context_blocks_for_llm(retrieved: list[dict[str, Any]], max_chunks: int = 5) -> str:
    blocks = []
    for r in retrieved[:max_chunks]:
        cid = r.get("chunk_id", "")
        src = r.get("source", "")
        txt = r.get("text", "")
        blocks.append(f"[chunk_id={cid}]\nKaynak: {src}\n{txt}")
    return "\n\n---\n\n".join(blocks)


def short_context_quote(context_text: str, max_words: int = 15) -> str:
    """Tek satır, kelime bazlı kısa alıntı (RAG_SYSTEM_PROMPT ile uyumlu)."""
    t = " ".join(str(context_text).split())
    words = t.split()
    return " ".join(words[:max_words]) if words else ""


def format_sft_assistant_with_sources(
    answer_body: str,
    chunk_id: str,
    source: str,
    context_text: str,
    max_quote_words: int = 15,
) -> str:
    """
    SFT hedefi: cevap gövdesi + Kaynaklar bloğu (inference / ödev formatı).
    chunk_id ve source, chunks.jsonl ile birebir olmalı.
    """
    body = (answer_body or "").strip()
    cid = str(chunk_id or "").strip()
    src = (source or "").strip() or "Kaynak"
    if len(src) > 120:
        src = src[:117].rstrip() + "..."
    quote = short_context_quote(context_text, max_words=max_quote_words)
    if not cid:
        return body
    line = f"- [chunk_id={cid}] {src} — {quote}"
    return f"{body}\n\nKaynaklar:\n{line}"


def split_rag_response(raw: str) -> tuple[str, str]:
    """(cevap_govdesi, kaynaklar_blogu). Metrikler govde uzerinden."""
    if not raw or not str(raw).strip():
        return "", ""
    raw = str(raw).strip()
    m = re.search(r"(?i)\n\s*Kaynaklar\s*:\s*", raw)
    if m:
        body = raw[: m.start()].strip()
        src_block = raw[m.end() :].strip()
    else:
        body = raw
        src_block = ""
    body = re.sub(r"(?is)^\s*cevap\s*:\s*", "", body).strip()
    return body, src_block


def extract_cited_chunk_ids(text: str) -> list[str]:
    """Metinde gecen chunk_id=... ifadeleri (sirayi koru, tekrarsiz)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in re.finditer(r"chunk_id\s*=\s*([^\s\]\)\n]+)", text, flags=re.I):
        cid = m.group(1).strip()
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(pred: str, ref: str) -> float:
    p_tok = normalize(pred).split()
    r_tok = normalize(ref).split()
    if not r_tok:
        return 1.0 if not p_tok else 0.0
    if not p_tok:
        return 0.0
    common = sum((Counter(p_tok) & Counter(r_tok)).values())
    if common == 0:
        return 0.0
    prec = common / len(p_tok)
    rec = common / len(r_tok)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def retrieval_hit(retrieved: list[dict[str, Any]], gold_q: dict[str, Any]) -> dict[str, Any]:
    gold_tokens = set(re.findall(r"\w+", gold_q.get("gold_context", "").lower()))
    first_hit = None
    for r in retrieved:
        chunk_tokens = set(re.findall(r"\w+", r.get("text", "").lower()))
        overlap = len(gold_tokens & chunk_tokens) / max(len(gold_tokens), 1)
        if overlap >= 0.4:
            first_hit = r["rank"]
            break
    out: dict[str, Any] = {}
    for k in [1, 3, 5, 10]:
        out[f"hit@{k}"] = first_hit is not None and first_hit <= k
    out["mrr"] = (1.0 / first_hit) if first_hit else 0.0
    relevances = []
    for r in retrieved[:10]:
        chunk_tokens = set(re.findall(r"\w+", r.get("text", "").lower()))
        overlap = len(gold_tokens & chunk_tokens) / max(len(gold_tokens), 1)
        relevances.append(1.0 if overlap >= 0.4 else 0.0)
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    out["ndcg@10"] = dcg / idcg if idcg > 0 else 0.0
    return out


def sentence_bleu_smooth(pred: str, ref: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return 0.0
    ref_t = normalize(ref).split()
    pred_t = normalize(pred).split()
    if not ref_t:
        return 1.0 if not pred_t else 0.0
    if not pred_t:
        return 0.0
    sm = SmoothingFunction().method1
    return float(sentence_bleu([ref_t], pred_t, smoothing_function=sm))


def rouge_l_f1(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return 0.0
    if not pred.strip() or not ref.strip():
        return 0.0
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(sc.score(ref, pred)["rougeL"].fmeasure)


def rouge_all_f1(pred: str, ref: str) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if not str(pred).strip() or not str(ref).strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    s = sc.score(ref, pred)
    return {
        "rouge1": float(s["rouge1"].fmeasure),
        "rouge2": float(s["rouge2"].fmeasure),
        "rougeL": float(s["rougeL"].fmeasure),
    }


def faithfulness_token_recall(answer_body: str, context_texts: list[str]) -> float:
    """Cevap tokenlarinin ne kadari baglam kelime hazinesinde (basit operational faithfulness)."""
    ctx_words: set[str] = set()
    for t in context_texts:
        ctx_words.update(re.findall(r"\w+", t.lower()))
    toks = re.findall(r"\w+", answer_body.lower())
    if not toks:
        return 1.0
    hits = sum(1 for t in toks if t in ctx_words)
    return hits / len(toks)


def citation_metrics(
    raw_model_output: str,
    retrieved: list[dict[str, Any]],
    gold_reference_chunk_ids: list[str] | None,
) -> dict[str, float]:
    """
    citation_precision_retrieved: uretilen atiflarin ne kadari gercekten top-k icinde.
    citation_recall_gold: gold reference_chunk_ids ile ortusme (yoksa nan yerine -1.0 raporda filtrelenebilir).
    """
    ret_ids = {r.get("chunk_id") for r in retrieved if r.get("chunk_id")}
    cited = extract_cited_chunk_ids(raw_model_output)
    if not cited:
        prec = 0.0
    else:
        valid = sum(1 for c in cited if c in ret_ids)
        prec = valid / len(cited)
    refs = gold_reference_chunk_ids or []
    if not refs:
        rec_g = -1.0
    else:
        cited_set = set(cited)
        inter = sum(1 for r in refs if r in cited_set)
        rec_g = inter / len(refs)
    return {
        "citation_precision_retrieved": float(prec),
        "citation_recall_gold": float(rec_g),
        "num_citations": float(len(cited)),
    }


def aggregate_qa_row(
    raw_answer: str,
    gold_cevap: str,
    retrieved: list[dict[str, Any]],
    gold_q: dict[str, Any],
) -> dict[str, Any]:
    body, _src = split_rag_response(raw_answer)
    ctx = [r.get("text", "") for r in retrieved]
    refs = gold_q.get("reference_chunk_ids")
    if not isinstance(refs, list):
        refs = []
    cm = citation_metrics(raw_answer, retrieved, refs)
    rg = rouge_all_f1(body, gold_cevap)
    return {
        "pred_body": body,
        "f1": token_f1(body, gold_cevap),
        "em": exact_match(body, gold_cevap),
        "bleu": sentence_bleu_smooth(body, gold_cevap),
        "rougeL": rg["rougeL"],
        "rouge1": rg["rouge1"],
        "rouge2": rg["rouge2"],
        "faithfulness_token_recall": faithfulness_token_recall(body, ctx),
        **cm,
    }
