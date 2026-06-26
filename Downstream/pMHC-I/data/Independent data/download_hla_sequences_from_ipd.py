#!/usr/bin/env python3
"""Download HLA full sequences from IPD-IMGT/HLA for alleles in el_train/test.

Default behavior:
  - read el_train.csv and el_test.csv in this directory;
  - collect human HLA-A/B/C alleles from the `mhc` column;
  - normalize names such as HLA-A02:01 to IPD names such as A*02:01;
  - download protein sequences through the IPD REST API;
  - write one representative full sequence per input two-field HLA allele.

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
IPD_DOWNLOAD_URL = "https://www.ebi.ac.uk/cgi-bin/ipd/api/allele/download"


@dataclass(frozen=True)
class FastaRecord:
    header: str
    accession: str
    ipd_name: str
    sequence: str


@dataclass(frozen=True)
class MissingSequence:
    allele: str
    reason: str


def normalize_hla_name(raw_name: str) -> str | None:
    """Convert HLA-A02:01 / HLA-A*02:01 to A*02:01.

    Non-human or non-class-I names such as H-2-Kb, BoLA, DLA, Mamu are ignored.
    """
    name = raw_name.strip()
    if not name.startswith("HLA-"):
        return None

    name = name[4:]
    if len(name) < 2 or name[0] not in "ABC":
        return None

    if "*" in name:
        locus, fields = name.split("*", 1)
    else:
        locus, fields = name[0], name[1:]

    parts = fields.split(":")
    if len(parts) < 2:
        return None

    first, second = parts[0], parts[1]
    if not (first.isdigit() and second.isdigit() and len(first) == 2 and len(second) == 2):
        return None

    return f"{locus}*{first}:{second}"


def hla_display_name(ipd_two_field_name: str) -> str:
    locus, fields = ipd_two_field_name.split("*", 1)
    return f"HLA-{locus}{fields}"


def read_hla_alleles(input_dir: Path, filenames: list[str], mhc_column: str) -> tuple[list[str], list[str]]:
    hla = set()
    skipped = set()

    for filename in filenames:
        path = input_dir / filename
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if mhc_column not in (reader.fieldnames or []):
                raise ValueError(f"{path} does not contain column {mhc_column!r}")
            for row in reader:
                raw = row[mhc_column].strip()
                normalized = normalize_hla_name(raw)
                if normalized:
                    hla.add(normalized)
                else:
                    skipped.add(raw)

    return sorted(hla), sorted(skipped)


def fetch_ipd_fasta(two_field_name: str, sequence_type: str, timeout: int) -> str:
    query = f'startsWith(name,"{two_field_name}")'
    params = urllib.parse.urlencode(
        {
            "project": "HLA",
            "type": sequence_type,
            "query": query,
        }
    )
    request = urllib.request.Request(
        f"{IPD_DOWNLOAD_URL}?{params}",
        headers={"User-Agent": "AntigenLM-HLA-sequence-downloader/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_fasta(text: str) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal header, chunks
        if not header:
            return
        sequence = "".join(chunks).replace(" ", "").replace("\r", "")
        parts = header.split("|")
        accession = parts[0] if parts else ""
        ipd_name = parts[1] if len(parts) > 1 else ""
        records.append(FastaRecord(header=header, accession=accession, ipd_name=ipd_name, sequence=sequence))
        header = None
        chunks = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:]
        else:
            chunks.append(line)
    flush()

    return records


def expression_suffix(ipd_name: str) -> str:
    return ipd_name[-1] if ipd_name and ipd_name[-1].isalpha() else ""


def choose_representative(records: list[FastaRecord]) -> FastaRecord | None:
    """Choose a practical representative sequence for a two-field HLA allele.

    For pMHC work, two-field names identify the protein sequence. We prefer
    non-null/non-low-expression entries and then the longest available sequence.
    """
    if not records:
        return None

    bad_suffixes = {"N", "L", "S", "C", "A", "Q"}

    def natural_name_key(name: str) -> tuple[object, ...]:
        chunks: list[object] = []
        for part in re.split(r"(\d+)", name):
            if part.isdigit():
                chunks.append(int(part))
            elif part:
                chunks.append(part)
        return tuple(chunks)

    def rank(record: FastaRecord) -> tuple[int, int, int, tuple[object, ...]]:
        suffix = expression_suffix(record.ipd_name)
        suffix_bad = 1 if suffix in bad_suffixes else 0
        sequence_dirty = 1 if any(ch in record.sequence for ch in ".*") else 0
        return (suffix_bad, sequence_dirty, -len(record.sequence), natural_name_key(record.ipd_name))

    return min(records, key=rank)


def write_fasta(path: Path, records: list[tuple[str, FastaRecord]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for requested_name, record in records:
            handle.write(
                f">{hla_display_name(requested_name)}|{requested_name}|"
                f"{record.accession}|{record.ipd_name}|length={len(record.sequence)}\n"
            )
            for start in range(0, len(record.sequence), 80):
                handle.write(record.sequence[start : start + 80] + "\n")


def write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_hla",
        "ipd_query",
        "ipd_accession",
        "ipd_allele",
        "sequence_length",
        "sequence_type",
        "match_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HLA sequences from IPD-IMGT/HLA")
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--files", nargs="+", default=["el_train.csv", "el_test.csv"])
    parser.add_argument("--mhc-column", default="mhc")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "ipd_hla_sequences")
    parser.add_argument(
        "--sequence-type",
        choices=["protein", "coding", "genomic"],
        default="protein",
        help="IPD sequence type to download. protein is usually what pMHC models need.",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument(
        "--all-matching-ipd-alleles",
        action="store_true",
        help="Write every IPD allele returned for each two-field HLA query, not just one representative.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hla_alleles, skipped = read_hla_alleles(args.input_dir, args.files, args.mhc_column)

    representative_records: list[tuple[str, FastaRecord]] = []
    all_records: list[tuple[str, FastaRecord]] = []
    mapping_rows: list[dict[str, str]] = []
    missing: list[MissingSequence] = []

    for idx, allele in enumerate(hla_alleles, start=1):
        print(f"[{idx}/{len(hla_alleles)}] querying {hla_display_name(allele)} ...", flush=True)
        try:
            text = fetch_ipd_fasta(allele, args.sequence_type, args.timeout)
        except Exception as exc:
            missing.append(MissingSequence(allele, f"{type(exc).__name__}: {exc}"))
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
            continue

        records = parse_fasta(text)
        representative = choose_representative(records)

        print(f"  {len(records)} IPD records", flush=True)
        if not representative:
            missing.append(MissingSequence(allele, "no FASTA records returned"))
            continue

        representative_records.append((allele, representative))
        all_records.extend((allele, record) for record in records)
        mapping_rows.append(
            {
                "input_hla": hla_display_name(allele),
                "ipd_query": allele,
                "ipd_accession": representative.accession,
                "ipd_allele": representative.ipd_name,
                "sequence_length": str(len(representative.sequence)),
                "sequence_type": args.sequence_type,
                "match_count": str(len(records)),
            }
        )
        time.sleep(args.sleep)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    representative_fasta = args.output_dir / f"hla_{args.sequence_type}_representative.fasta"
    mapping_csv = args.output_dir / f"hla_{args.sequence_type}_representative_mapping.csv"
    skipped_txt = args.output_dir / "non_hla_or_unsupported_mhc_skipped.txt"
    missing_txt = args.output_dir / f"missing_{args.sequence_type}_sequences.txt"

    write_fasta(representative_fasta, representative_records)
    write_mapping(mapping_csv, mapping_rows)
    skipped_txt.write_text("\n".join(skipped) + ("\n" if skipped else ""))
    missing_txt.write_text(
        "\n".join(f"{hla_display_name(item.allele)}\t{item.reason}" for item in missing)
        + ("\n" if missing else "")
    )

    if args.all_matching_ipd_alleles:
        all_fasta = args.output_dir / f"hla_{args.sequence_type}_all_matching_ipd_alleles.fasta"
        write_fasta(all_fasta, all_records)
        print(f"Wrote all matching IPD records: {all_fasta}")

    print(f"HLA alleles found in input files: {len(hla_alleles)}")
    print(f"Representative sequences written: {len(representative_records)}")
    print(f"Missing sequences: {len(missing)}")
    print(f"Skipped non-HLA/unsupported MHC names: {len(skipped)}")
    print(f"Wrote FASTA: {representative_fasta}")
    print(f"Wrote mapping CSV: {mapping_csv}")


if __name__ == "__main__":
    main()
