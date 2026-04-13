"""
cleaned_corpus.jsonl -> articles.jsonl (madde / baslik bazli metin bolme).

Girdi: prepare_corpus.py ciktisi (doc_id, source, document_type, text, score, ...).
Cikti: Her context icin bir veya cok article kaydi; sonrasinda chunk_corpus ile
kisa parcalara bolunur (varsayilan cikti: chunks_article_window.jsonl).

NOT: CENG493 RAG + Kaggle satir hizasi icin data/processed/chunks.jsonl dogrudan
prepare_corpus (--rag-chunks-output) ile uretilmelidir; bu script o akista zorunlu degildir.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ARTICLE_START_PATTERN = re.compile(
    r"^(?P<header>(?:EK\s+MADDE\s+\d+|Ek\s+Madde\s+\d+|GEÇİCİ\s+MADDE\s+\d+|Geçici\s+Madde\s+\d+|MADDE\s+\d+|Madde\s+\d+)\s*[-–—]?)\s*(?P<rest>.*)$"
)

BASLANGIC_PATTERN = re.compile(r"^BAŞLANGIÇ(?:\s*\[\d+\])?$", re.IGNORECASE)

SECTION_PATTERNS = [
    re.compile(r"^[A-ZÇĞİIÖŞÜ]+\s+KISIM$", re.IGNORECASE),
    re.compile(r"^[A-ZÇĞİIÖŞÜ]+\s+BÖLÜM$", re.IGNORECASE),
    re.compile(r"^[A-ZÇĞİIÖŞÜ]+\s+KİTAP$", re.IGNORECASE),
    re.compile(r"^[A-ZÇĞİIÖŞÜ]+\s+AYIRIM$", re.IGNORECASE),
]

SUBCLAUSE_PATTERN = re.compile(
    r"^(?:[a-zçğıöşü]\)|[A-ZÇĞİIÖŞÜ]\)|\d+\.\s*|\(\d+\)|\([a-zçğıöşü]\))",
    re.IGNORECASE,
)

MULGA_PATTERN = re.compile(r"\(Mülga\s*[,:;]", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", " ").replace("\xa0", " ")

    lines = [line.strip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank

    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text


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


def clean_article_key(header: str) -> str:
    text = normalize_whitespace(header).lower()
    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[-–—]+$", "", text).strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def is_section_line(line: str) -> bool:
    line = line.strip()
    return any(pattern.match(line) for pattern in SECTION_PATTERNS)


def is_subclause_line(line: str) -> bool:
    return bool(SUBCLAUSE_PATTERN.match(line.strip()))


def is_bad_heading_candidate(line: str) -> bool:
    """
    Satır teknik olarak kısa olsa bile heading olmamalı.
    Örn:
    - Cumhurbaşkanlığı Konseyinin görevleri şunlardır:
    - aşağıdaki bentlerin giriş cümleleri
    """
    line = line.strip().lower()
    if not line:
        return False

    bad_suffixes = [
        "şunlardır:",
        "aşağıdadır:",
        "aşağıdaki gibidir:",
    ]
    return any(line.endswith(sfx) for sfx in bad_suffixes)


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False

    if ARTICLE_START_PATTERN.match(line):
        return False

    if is_subclause_line(line):
        return False

    if is_bad_heading_candidate(line):
        return False

    if BASLANGIC_PATTERN.match(line):
        return True

    if is_section_line(line):
        return True

    if len(line) <= 120 and not line.endswith(".") and not line.endswith(";") and not line.endswith(":"):
        return True

    return False


def finalize_article(
    article_key: str,
    article_title: str,
    article_lines: list[str],
    base_hierarchy: list[str],
    local_heading: str | None,
) -> dict[str, str]:
    prefix_parts = [part for part in base_hierarchy if part.strip()]
    if local_heading and local_heading.strip():
        prefix_parts.append(local_heading.strip())

    body = "\n".join(article_lines).strip()

    if prefix_parts:
        full_text = "\n".join(prefix_parts) + "\n\n" + body
    else:
        full_text = body

    full_text = normalize_whitespace(full_text)

    return {
        "article_key": article_key,
        "article_title": article_title,
        "text": full_text,
    }


def split_articles(text: str) -> list[dict[str, str]]:
    text = normalize_whitespace(text)
    if not text:
        return []

    lines = text.split("\n")
    results: list[dict[str, str]] = []

    current_part_line: str | None = None
    current_chapter_line: str | None = None
    current_book_line: str | None = None
    current_section_line: str | None = None
    current_general_title: str | None = None

    pending_local_heading: str | None = None

    current_article_key: str | None = None
    current_article_title: str | None = None
    current_article_lines: list[str] = []
    current_article_base_hierarchy: list[str] = []
    current_article_local_heading: str | None = None

    baslangic_mode = False
    baslangic_lines: list[str] = []
    baslangic_hierarchy: list[str] = []

    expect_general_title_after_section = False

    def get_base_hierarchy() -> list[str]:
        parts: list[str] = []
        if current_book_line:
            parts.append(current_book_line)
        if current_part_line:
            parts.append(current_part_line)
        if current_chapter_line:
            parts.append(current_chapter_line)
        if current_section_line:
            parts.append(current_section_line)
        if current_general_title:
            parts.append(current_general_title)
        return parts

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            if current_article_key and current_article_lines:
                current_article_lines.append("")
            elif baslangic_mode:
                baslangic_lines.append("")
            i += 1
            continue

        if BASLANGIC_PATTERN.match(line):
            if current_article_key:
                results.append(
                    finalize_article(
                        article_key=current_article_key,
                        article_title=current_article_title or current_article_key,
                        article_lines=current_article_lines,
                        base_hierarchy=current_article_base_hierarchy,
                        local_heading=current_article_local_heading,
                    )
                )
                current_article_key = None
                current_article_title = None
                current_article_lines = []

            baslangic_mode = True
            baslangic_hierarchy = get_base_hierarchy()
            baslangic_lines = [line]
            i += 1
            continue

        article_match = ARTICLE_START_PATTERN.match(line)
        if article_match:
            if baslangic_mode:
                results.append(
                    finalize_article(
                        article_key="baslangic",
                        article_title="BAŞLANGIÇ",
                        article_lines=baslangic_lines,
                        base_hierarchy=baslangic_hierarchy,
                        local_heading=None,
                    )
                )
                baslangic_mode = False
                baslangic_lines = []
                baslangic_hierarchy = []

            if current_article_key:
                results.append(
                    finalize_article(
                        article_key=current_article_key,
                        article_title=current_article_title or current_article_key,
                        article_lines=current_article_lines,
                        base_hierarchy=current_article_base_hierarchy,
                        local_heading=current_article_local_heading,
                    )
                )

            header = article_match.group("header").strip()
            rest = article_match.group("rest").strip()

            current_article_key = clean_article_key(header)
            current_article_title = header
            current_article_lines = [line if rest else header]
            current_article_base_hierarchy = get_base_hierarchy()
            current_article_local_heading = pending_local_heading
            pending_local_heading = None

            i += 1
            continue

        if baslangic_mode:
            baslangic_lines.append(line)
            i += 1
            continue

        if is_section_line(line):
            upper_line = line.upper()
            if "KİTAP" in upper_line:
                current_book_line = line
                current_part_line = None
                current_chapter_line = None
                current_section_line = None
            elif "KISIM" in upper_line:
                current_part_line = line
                current_chapter_line = None
                current_section_line = None
            elif "BÖLÜM" in upper_line:
                current_chapter_line = line
                current_section_line = None
            elif "AYIRIM" in upper_line:
                current_section_line = line

            current_general_title = None
            pending_local_heading = None
            expect_general_title_after_section = True

            i += 1
            continue

        if looks_like_heading(line):
            if not current_article_key:
                if expect_general_title_after_section and current_general_title is None:
                    current_general_title = line
                    expect_general_title_after_section = False
                else:
                    pending_local_heading = line
            else:
                current_article_lines.append(line)
                if not is_subclause_line(line) and not is_bad_heading_candidate(line):
                    pending_local_heading = line
            i += 1
            continue

        if current_article_key:
            current_article_lines.append(line)

        i += 1

    if baslangic_mode:
        results.append(
            finalize_article(
                article_key="baslangic",
                article_title="BAŞLANGIÇ",
                article_lines=baslangic_lines,
                base_hierarchy=baslangic_hierarchy,
                local_heading=None,
            )
        )

    if current_article_key:
        results.append(
            finalize_article(
                article_key=current_article_key,
                article_title=current_article_title or current_article_key,
                article_lines=current_article_lines,
                base_hierarchy=current_article_base_hierarchy,
                local_heading=current_article_local_heading,
            )
        )

    if not results:
        return [
            {
                "article_key": "full_text",
                "article_title": "FULL_TEXT",
                "text": text,
            }
        ]

    return results


def build_article_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    article_records: list[dict[str, Any]] = []

    total_input_docs = len(records)
    docs_with_article_split = 0
    docs_fallback_full_text = 0
    total_output_articles = 0
    mulga_count = 0

    for record in records:
        source_doc_id = record["doc_id"]
        source = record.get("source", "")
        document_type = record.get("document_type", "")
        text = record.get("text", "")
        score = record.get("score", None)

        article_chunks = split_articles(text)

        if not article_chunks:
            continue

        if len(article_chunks) == 1 and article_chunks[0]["article_key"] == "full_text":
            docs_fallback_full_text += 1
        else:
            docs_with_article_split += 1

        id_parts = source_doc_id.split("__")
        source_slug = id_parts[0]
        parent_hash = id_parts[-1]

        for idx, chunk in enumerate(article_chunks):
            article_key = chunk["article_key"]
            article_title = chunk["article_title"]
            article_text = chunk["text"]
            is_mulga = bool(MULGA_PATTERN.search(article_text))

            if is_mulga:
                mulga_count += 1

            new_record = {
                "doc_id": f"{source_slug}__{parent_hash}__{article_key}",
                "parent_doc_id": source_doc_id,
                "source": source,
                "document_type": document_type,
                "article_key": article_key,
                "article_title": article_title,
                "text": article_text,
                "char_length": len(article_text),
                "score": score,
                "article_order": idx,
                "is_mulga": is_mulga,
            }
            article_records.append(new_record)
            total_output_articles += 1

    stats = {
        "total_input_docs": total_input_docs,
        "docs_with_article_split": docs_with_article_split,
        "docs_fallback_full_text": docs_fallback_full_text,
        "total_output_articles": total_output_articles,
        "mulga_articles": mulga_count,
        "avg_articles_per_doc": (
            total_output_articles / total_input_docs if total_input_docs > 0 else 0.0
        ),
    }

    return article_records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split cleaned legal corpus into article-level records."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/cleaned_corpus.jsonl",
        help="Path to cleaned corpus JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/articles.jsonl",
        help="Path to article-level JSONL output.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="data/interim/articles_stats.json",
        help="Path to article-level stats JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_path}")

    print(f"[INFO] Cleaned corpus okunuyor: {input_path}")
    records = load_jsonl(input_path)

    print("[INFO] Article-level parsing başlıyor...")
    article_records, stats = build_article_records(records)

    print(f"[INFO] Article JSONL kaydediliyor: {output_path}")
    save_jsonl(article_records, output_path)

    print(f"[INFO] İstatistikler kaydediliyor: {stats_path}")
    save_stats(stats, stats_path)

    print("\n[OK] Article parsing tamamlandı.")
    print(f"Input docs             : {stats['total_input_docs']}")
    print(f"Docs with split        : {stats['docs_with_article_split']}")
    print(f"Fallback full-text     : {stats['docs_fallback_full_text']}")
    print(f"Output article count   : {stats['total_output_articles']}")
    print(f"Avg articles per doc   : {stats['avg_articles_per_doc']:.2f}")


if __name__ == "__main__":
    main()