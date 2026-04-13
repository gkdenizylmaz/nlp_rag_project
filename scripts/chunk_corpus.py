"""
articles.jsonl -> sliding-window / paragraf chunklari (ince parca deney hatti).

Varsayilan cikti: data/processed/chunks_article_window.jsonl
(data/processed/chunks.jsonl = prepare_corpus.py Kaggle-context hatti; uzerine yazmayin).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_SENT_BOUNDARY = re.compile(r"[.!?;…]\s")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_stats(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        stripped = line.strip()
        is_blank = stripped == ""
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(stripped if stripped else "")
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def split_paragraphs(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _find_break_point(text: str, ideal_pos: int, min_pos: int) -> int:
    """
    Find a clean break point at or before *ideal_pos* (never earlier than *min_pos*).
    Prefers sentence boundaries (.!?;…), falls back to word boundaries (space/newline).
    """
    if ideal_pos >= len(text):
        return len(text)

    search_floor = max(min_pos, ideal_pos - 250)
    region = text[search_floor : ideal_pos + 1]

    last_sent = None
    for m in _SENT_BOUNDARY.finditer(region):
        last_sent = m

    if last_sent is not None:
        return search_floor + last_sent.end()

    last_space = region.rfind(" ")
    if last_space >= 0:
        return search_floor + last_space + 1

    last_nl = region.rfind("\n")
    if last_nl >= 0:
        return search_floor + last_nl + 1

    return ideal_pos


def _snap_to_word_start(text: str, pos: int) -> int:
    """If *pos* lands in the middle of a word, snap forward to the next word start."""
    if pos <= 0 or pos >= len(text):
        return pos
    if text[pos - 1] in " \n":
        return pos
    for j in range(pos, min(len(text), pos + 80)):
        if text[j] in " \n":
            return j + 1
    return pos


def sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        ideal_end = min(len(text), start + chunk_size)

        if ideal_end == len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        min_end = start + int(chunk_size * 0.5)
        end = _find_break_point(text, ideal_end, min_end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = max(start + 1, end - overlap)
        next_start = _snap_to_word_start(text, next_start)
        start = next_start

    return chunks


def merge_small_chunks(chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
    """
    Merge overly small chunks with neighbors, especially heading-only orphan chunks.
    """
    if not chunks:
        return []

    merged = chunks[:]

    changed = True
    while changed and len(merged) > 1:
        changed = False
        new_chunks: list[str] = []
        i = 0

        while i < len(merged):
            current = merged[i].strip()

            # If this is the last chunk, just append
            if i == len(merged) - 1:
                if new_chunks and len(current) < min_chars:
                    candidate = new_chunks[-1] + "\n\n" + current
                    if len(candidate) <= max_chars + 250:
                        new_chunks[-1] = candidate.strip()
                        changed = True
                    else:
                        new_chunks.append(current)
                else:
                    new_chunks.append(current)
                i += 1
                continue

            # Small current chunk -> merge with next preferably
            if len(current) < min_chars:
                nxt = merged[i + 1].strip()
                candidate = current + "\n\n" + nxt

                if len(candidate) <= max_chars + 250:
                    new_chunks.append(candidate.strip())
                    changed = True
                    i += 2
                    continue

                # fallback: merge backward
                if new_chunks:
                    candidate_back = new_chunks[-1] + "\n\n" + current
                    if len(candidate_back) <= max_chars + 250:
                        new_chunks[-1] = candidate_back.strip()
                        changed = True
                    else:
                        new_chunks.append(current)
                else:
                    new_chunks.append(current)

                i += 1
                continue

            new_chunks.append(current)
            i += 1

        merged = new_chunks

    return merged


def paragraph_based_chunks(
    text: str,
    max_chars: int = 1200,
    min_chars: int = 350,
    overlap_chars: int = 180,
) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        chunks = sliding_window_chunks(text, chunk_size=max_chars, overlap=overlap_chars)
        return merge_small_chunks(chunks, min_chars=min_chars, max_chars=max_chars)

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Too long single paragraph -> flush current, split paragraph itself
        if len(para) > max_chars:
            if current.strip():
                chunks.append(current.strip())
                current = ""

            long_para_chunks = sliding_window_chunks(
                para,
                chunk_size=max_chars,
                overlap=overlap_chars,
            )
            chunks.extend(long_para_chunks)
            continue

        candidate = para if not current else f"{current}\n\n{para}"

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para

    if current.strip():
        chunks.append(current.strip())

    # Critical post-process: remove orphan short chunks
    chunks = merge_small_chunks(chunks, min_chars=min_chars, max_chars=max_chars)
    return chunks


def build_chunks(
    article_records: list[dict[str, Any]],
    max_chars: int = 1200,
    min_chars: int = 350,
    overlap_chars: int = 180,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_records: list[dict[str, Any]] = []

    total_articles = len(article_records)
    single_chunk_articles = 0
    multi_chunk_articles = 0
    total_chunks = 0

    chunk_lengths: list[int] = []

    for article in article_records:
        text = normalize_text(article.get("text", "") or "")
        if not text:
            continue

        chunks = paragraph_based_chunks(
            text=text,
            max_chars=max_chars,
            min_chars=min_chars,
            overlap_chars=overlap_chars,
        )

        if not chunks:
            continue

        if len(chunks) == 1:
            single_chunk_articles += 1
        else:
            multi_chunk_articles += 1

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{article['doc_id']}__chunk_{idx:03d}"

            chunk_record = {
                "chunk_id": chunk_id,
                "doc_id": article["doc_id"],
                "parent_doc_id": article.get("parent_doc_id"),
                "source": article.get("source"),
                "document_type": article.get("document_type"),
                "article_key": article.get("article_key"),
                "article_title": article.get("article_title"),
                "article_order": article.get("article_order"),
                "chunk_order": idx,
                "text": chunk_text,
                "char_length": len(chunk_text),
                "score": article.get("score"),
                "is_mulga": article.get("is_mulga", False),
            }

            chunk_records.append(chunk_record)
            total_chunks += 1
            chunk_lengths.append(len(chunk_text))

    sorted_lengths = sorted(chunk_lengths)
    median = sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0

    stats = {
        "total_articles": total_articles,
        "single_chunk_articles": single_chunk_articles,
        "multi_chunk_articles": multi_chunk_articles,
        "total_chunks": total_chunks,
        "avg_chunks_per_article": (total_chunks / total_articles) if total_articles else 0.0,
        "chunk_length_stats": {
            "min": min(chunk_lengths) if chunk_lengths else 0,
            "max": max(chunk_lengths) if chunk_lengths else 0,
            "mean": (sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else 0.0,
            "median": median,
        },
    }

    return chunk_records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk article-level legal corpus into retrieval-ready chunks."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/articles.jsonl",
        help="Path to article-level JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/chunks_article_window.jsonl",
        help=(
            "Path to output chunks JSONL. Default avoids overwriting "
            "data/processed/chunks.jsonl (prepare_corpus Kaggle-context output)."
        ),
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="data/processed/chunks_article_window_stats.json",
        help="Path to output chunk stats JSON.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=350,
        help="Minimum preferred characters per chunk before merging.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=180,
        help="Overlap used in sliding-window fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_path}")

    print(f"[INFO] Article corpus okunuyor: {input_path}")
    article_records = load_jsonl(input_path)

    print("[INFO] Chunking başlıyor...")
    chunk_records, stats = build_chunks(
        article_records=article_records,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        overlap_chars=args.overlap_chars,
    )

    print(f"[INFO] Chunk JSONL kaydediliyor: {output_path}")
    save_jsonl(chunk_records, output_path)

    print(f"[INFO] İstatistikler kaydediliyor: {stats_path}")
    save_stats(stats, stats_path)

    print("\n[OK] Chunking tamamlandı.")
    print(f"Total articles         : {stats['total_articles']}")
    print(f"Single chunk articles  : {stats['single_chunk_articles']}")
    print(f"Multi chunk articles   : {stats['multi_chunk_articles']}")
    print(f"Total chunks           : {stats['total_chunks']}")
    print(f"Avg chunks/article     : {stats['avg_chunks_per_article']:.2f}")
    print(f"Chunk length stats     : {stats['chunk_length_stats']}")


if __name__ == "__main__":
    main()