"""
gold_test_set.json içine reference_chunk_ids yazar.

Eşleme sırası (prepare_corpus normalize_whitespace ile):
  1) Tam metin eşleşmesi: gold_context == chunk.text
  2) gold_context, chunk.text içinde — en kısa içeren chunk (uzunluk artan sırada ilk eşleşme)
  3) chunk.text, gold_context içinde — en uzun uygun chunk (uzunluk azalan sırada ilk eşleşme)
  4) Token Jaccard (önceden hesaplı token kümeleri; son çare)

Çalıştır: proje kökünden  python scripts/annotate_gold_chunk_refs.py
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from prepare_corpus import normalize_whitespace  # noqa: E402


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def best_chunk_id_jaccard(
    g: set[str],
    chunk_token_sets: list[tuple[str, set[str]]],
    min_jaccard: float,
) -> list[str]:
    if not g:
        return []
    best_id, best_score = None, 0.0
    for cid, t in chunk_token_sets:
        if not t:
            continue
        inter = len(g & t)
        uni = len(g | t)
        j = inter / uni if uni else 0.0
        if j > best_score:
            best_score = j
            best_id = cid
    if best_id and best_score >= min_jaccard:
        return [str(best_id)]
    return []


@dataclass(frozen=True)
class ChunkIndex:
    exact_norm_to_id: dict[str, str]
    """normalize(chunk.text) -> chunk_id"""
    by_len_asc: list[tuple[str, str]]
    """(norm_text, chunk_id) kısadan uzuna"""
    by_len_desc: list[tuple[str, str]]
    """uzundan kısaya (chunk_in_gold)"""
    chunk_token_sets: list[tuple[str, set[str]]]


def build_chunk_index(chunks: list[dict]) -> ChunkIndex:
    pairs: list[tuple[str, str]] = []
    token_sets: list[tuple[str, set[str]]] = []
    exact: dict[str, str] = {}

    for c in chunks:
        cid = c.get("chunk_id")
        if not cid:
            continue
        cid_s = str(cid)
        nt = normalize_whitespace(c.get("text", ""))
        if not nt:
            continue
        exact[nt] = cid_s
        pairs.append((nt, cid_s))
        token_sets.append((cid_s, _tokens(nt)))

    pairs.sort(key=lambda x: len(x[0]))
    by_len_asc = pairs
    by_len_desc = sorted(pairs, key=lambda x: len(x[0]), reverse=True)

    return ChunkIndex(
        exact_norm_to_id=exact,
        by_len_asc=by_len_asc,
        by_len_desc=by_len_desc,
        chunk_token_sets=token_sets,
    )


def resolve_reference_chunk_ids(
    gold_context: str,
    idx: ChunkIndex,
    *,
    min_jaccard: float,
    min_substring_len: int = 24,
    min_chunk_in_gold_len: int = 80,
) -> tuple[list[str], str]:
    gc = normalize_whitespace(gold_context)
    if not gc:
        return [], "none"

    # 1) Tam metin
    cid = idx.exact_norm_to_id.get(gc)
    if cid:
        return [cid], "exact_text"

    # 2) gold, chunk içinde (kısadan uzuna; ilk eşleşme = en kısa içeren)
    if len(gc) >= min_substring_len:
        for ct, cid in idx.by_len_asc:
            if len(ct) < len(gc):
                continue
            if gc in ct:
                return [cid], "gold_in_chunk"

    # 3) chunk, gold içinde (uzundan kısa; ilk eşleşme = en uzun parça)
    for ct, cid in idx.by_len_desc:
        if len(ct) < min_chunk_in_gold_len:
            continue
        if ct in gc:
            return [cid], "chunk_in_gold"

    # 4) Jaccard
    got = best_chunk_id_jaccard(
        _tokens(gc), idx.chunk_token_sets, min_jaccard=min_jaccard
    )
    return (got, "jaccard" if got else "none")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=str, default="data/processed/chunks.jsonl")
    p.add_argument("--gold", type=str, default="evaluation/gold_test_set.json")
    p.add_argument("--min-jaccard", type=float, default=0.12)
    args = p.parse_args()
    root = _ROOT
    chunks_path = root / args.chunks
    gold_path = root / args.gold
    chunks: list[dict] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"[INFO] {len(chunks)} chunk indeksleniyor...")
    cidx = build_chunk_index(chunks)

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    n_ok = n_empty = 0
    by_method: dict[str, int] = {}

    for q in gold:
        if not q.get("answerable_from_corpus", True):
            q["reference_chunk_ids"] = []
            by_method["not_answerable"] = by_method.get("not_answerable", 0) + 1
            continue
        refs, method = resolve_reference_chunk_ids(
            q.get("gold_context", ""),
            cidx,
            min_jaccard=args.min_jaccard,
        )
        q["reference_chunk_ids"] = refs
        by_method[method] = by_method.get(method, 0) + 1
        if refs:
            n_ok += 1
        else:
            n_empty += 1

    gold_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {gold_path} güncellendi.")
    print(f"  answerable + ref atanmış: {n_ok}, atanamayan: {n_empty}, toplam kayıt: {len(gold)}")
    print("  yöntem sayıları:", dict(sorted(by_method.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
