"""
Kaggle CSV -> temiz korpus + (isteğe bağlı) RAG chunks.

İki çıktı hattı:
  A) Ana RAG + Kaggle Q&A hizası (01-04, embedding triple):
     - data/interim/cleaned_corpus.jsonl  (tekil context kayıtları)
     - data/processed/chunks.jsonl        (notebook şeması; metin = tam context)
     build_articles / chunk_corpus kullanılmaz.

  B) Madde bazlı ince parçalama (deney / eski hat):
     cleaned_corpus.jsonl -> build_articles.py -> articles.jsonl
     -> chunk_corpus.py -> chunks_article_window.jsonl (sliding-window)
     Bu chunks, Kaggle satırındaki tam context ile bire bir olmaz.

Gold test atıfları: `chunks.jsonl` güncellendikten sonra
`python scripts/annotate_gold_chunk_refs.py` ile `evaluation/gold_test_set.json`
içine `reference_chunk_ids` yazılır (önce tam metin / içerme, son çare Jaccard).

normalize_whitespace: build_articles.py ile aynı mantık (satır + boşluk).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ["soru", "cevap", "veri türü", "kaynak", "context", "Score"]


def normalize_whitespace(text: str) -> str:
    """
    Normalize excessive whitespace while preserving line breaks enough
    to keep legal document structure readable.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", " ")
    text = text.replace("\xa0", " ")

    # Trim each line
    lines = [line.strip() for line in text.split("\n")]

    # Collapse multiple blank lines to max one blank line
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank

    text = "\n".join(cleaned_lines).strip()

    # Collapse repeated spaces/tabs inside lines
    text = re.sub(r"[ \t]+", " ", text)

    return text


def slugify_source(text: str) -> str:
    """
    Create a filesystem/id-friendly slug from Turkish legal source names.
    """
    if not isinstance(text, str):
        return "unknown_source"

    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "c",
        "Ğ": "g",
        "İ": "i",
        "I": "i",
        "Ö": "o",
        "Ş": "s",
        "Ü": "u",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text or "unknown_source"


def extract_article_hint(text: str) -> str | None:
    """
    Try to extract a useful legal structure hint such as:
    - MADDE 10
    - BAŞLANGIÇ
    - EK MADDE 1
    """
    if not isinstance(text, str):
        return None

    patterns = [
        r"\b(EK\s+MADDE\s+\d+)\b",
        r"\b(MADDE\s+\d+)\b",
        r"\b(GEÇİCİ\s+MADDE\s+\d+)\b",
        r"\b(BAŞLANGIÇ)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hint = match.group(1).strip()
            hint = normalize_whitespace(hint)
            hint = slugify_source(hint)
            return hint

    return None


def compute_doc_id(source: str, context: str) -> str:
    """
    Produce a stable doc_id using source + optional article hint + short hash.
    """
    source_slug = slugify_source(source)
    article_hint = extract_article_hint(context)

    short_hash = hashlib.md5(context.encode("utf-8")).hexdigest()[:10]

    if article_hint:
        return f"{source_slug}__{article_hint}__{short_hash}"
    return f"{source_slug}__{short_hash}"


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar var: {missing}")


def build_cleaned_corpus(
    df: pd.DataFrame,
    min_score: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Filter, clean, deduplicate by context, and prepare JSONL-ready records.
    """
    validate_columns(df)

    original_rows = len(df)

    # Basic type cleanup
    df = df.copy()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["context"] = df["context"].astype(str)
    df["kaynak"] = df["kaynak"].astype(str)
    df["veri türü"] = df["veri türü"].astype(str)

    # Filter by score
    df = df[df["Score"] >= min_score].copy()
    filtered_rows = len(df)

    # Normalize context and source
    df["context_clean"] = df["context"].apply(normalize_whitespace)
    df["kaynak_clean"] = df["kaynak"].apply(normalize_whitespace)
    df["veri_turu_clean"] = df["veri türü"].apply(normalize_whitespace)

    # Remove empty/too short contexts after cleaning
    df["context_len"] = df["context_clean"].str.len()
    df = df[df["context_clean"].str.strip() != ""].copy()
    df = df[df["context_len"] >= 100].copy()
    after_empty_short_filter = len(df)

    # Sort so that higher score rows are preferred when deduplicating
    df = df.sort_values(
        by=["context_clean", "Score", "kaynak_clean"],
        ascending=[True, False, True],
    )

    # Deduplicate by cleaned context
    dedup_df = df.drop_duplicates(subset=["context_clean"], keep="first").copy()
    unique_contexts = len(dedup_df)

    records: list[dict[str, Any]] = []
    for idx, row in dedup_df.reset_index(drop=True).iterrows():
        context = row["context_clean"]
        source = row["kaynak_clean"]
        record = {
            "doc_id": compute_doc_id(source=source, context=context),
            "source": source,
            "document_type": row["veri_turu_clean"],
            "text": context,
            "char_length": len(context),
            "score": int(row["Score"]),
            "article_hint": extract_article_hint(context),
            "record_index": idx,
        }
        records.append(record)

    stats = {
        "original_rows": original_rows,
        "after_score_filter": filtered_rows,
        "after_empty_short_filter": after_empty_short_filter,
        "unique_context_count": unique_contexts,
        "source_distribution": dedup_df["kaynak_clean"].value_counts().to_dict(),
        "char_length_stats": {
            "min": int(dedup_df["context_len"].min()) if not dedup_df.empty else 0,
            "max": int(dedup_df["context_len"].max()) if not dedup_df.empty else 0,
            "mean": float(dedup_df["context_len"].mean()) if not dedup_df.empty else 0.0,
            "median": float(dedup_df["context_len"].median()) if not dedup_df.empty else 0.0,
        },
    }

    return records, stats


def interim_record_to_rag_chunk(record: dict[str, Any]) -> dict[str, Any]:
    """
    Notebook'larin (01-04) bekledigi chunk semasina cevir.
    Her kayit = Kaggle CSV'deki tekillestirilmis tam context (chunk_corpus parcalari degil).
    article_key olarak doc_id kullanilir: hard negative icin benzersiz baglam ayirici.
    """
    doc_id = record["doc_id"]
    hint = record.get("article_hint") or ""
    title = hint.replace("_", " ").strip() if hint else ""
    return {
        "chunk_id": f"{doc_id}__chunk_000",
        "doc_id": doc_id,
        "parent_doc_id": doc_id,
        "source": record["source"],
        "document_type": record.get("document_type") or "hukuk",
        "article_key": doc_id,
        "article_title": title or record["source"][:80],
        "article_order": int(record.get("record_index", 0)),
        "chunk_order": 0,
        "text": record["text"],
        "char_length": int(record["char_length"]),
        "score": int(record["score"]),
        "is_mulga": False,
    }


def save_rag_chunks_jsonl(
    interim_records: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """data/processed/chunks.jsonl — RAG + Kaggle Q&A hizasi icin tek kaynak."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rag = [interim_record_to_rag_chunk(r) for r in interim_records]
    with output_path.open("w", encoding="utf-8") as f:
        for row in rag:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lengths = [row["char_length"] for row in rag]
    sorted_l = sorted(lengths)
    mid = sorted_l[len(sorted_l) // 2] if sorted_l else 0
    return {
        "pipeline": "prepare_corpus_kaggle_context_aligned",
        "total_chunks": len(rag),
        "chunk_length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": float(sum(lengths) / len(lengths)) if lengths else 0.0,
            "median": mid,
        },
    }


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_stats(stats: dict[str, Any], stats_path: Path) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cleaned, deduplicated legal retrieval corpus from Kaggle dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/turkish_law_dataset.csv",
        help="Path to raw Kaggle CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/cleaned_corpus.jsonl",
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="data/interim/cleaned_corpus_stats.json",
        help="Path to output stats JSON file.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=8,
        help="Minimum score threshold for filtering.",
    )
    parser.add_argument(
        "--rag-chunks-output",
        type=str,
        default="data/processed/chunks.jsonl",
        help=(
            "RAG icin notebook-uyumlu chunks.jsonl (Kaggle context ile bire bir; "
            "chunk_corpus sliding-window ile karistirmayin)."
        ),
    )
    parser.add_argument(
        "--rag-chunks-stats-output",
        type=str,
        default="data/processed/chunks_stats.json",
        help="rag-chunks-output icin kisa istatistik.",
    )
    parser.add_argument(
        "--skip-rag-chunks",
        action="store_true",
        help="Sadece interim cleaned_corpus yaz; processed/chunks uretme.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_path}")

    print(f"[INFO] CSV okunuyor: {input_path}")
    df = pd.read_csv(input_path)

    print("[INFO] Corpus hazırlanıyor...")
    records, stats = build_cleaned_corpus(df=df, min_score=args.min_score)

    print(f"[INFO] JSONL kaydediliyor: {output_path}")
    save_jsonl(records, output_path)

    print(f"[INFO] İstatistikler kaydediliyor: {stats_path}")
    save_stats(stats, stats_path)

    if not args.skip_rag_chunks:
        rag_out = Path(args.rag_chunks_output)
        rag_stats_path = Path(args.rag_chunks_stats_output)
        print(f"[INFO] RAG chunks (Kaggle-hizali) yaziliyor: {rag_out}")
        rag_stats = save_rag_chunks_jsonl(records, rag_out)
        save_stats(rag_stats, rag_stats_path)
        print(f"[INFO] RAG chunk sayisi: {rag_stats['total_chunks']}")

    print("\n[OK] Corpus hazırlama tamamlandı.")
    print(f"Original rows           : {stats['original_rows']}")
    print(f"After score filter      : {stats['after_score_filter']}")
    print(f"After empty/short clean : {stats['after_empty_short_filter']}")
    print(f"Unique contexts         : {stats['unique_context_count']}")
    print(f"Char length stats       : {stats['char_length_stats']}")


if __name__ == "__main__":
    main()