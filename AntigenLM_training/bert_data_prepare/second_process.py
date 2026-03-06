import csv
from typing import List, Tuple


def read_fasta_as_list(file_path: str) -> List[Tuple[str, str]]:
    """
    以列表形式读取 FASTA，每个元素是一个 (seq_id, sequence) 元组。
    相同的 seq_id 会各占一条记录，互不影响。
    """
    records = []
    with open(file_path, "r") as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, "".join(seq_lines)))
                seq_id = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id is not None:
            records.append((seq_id, "".join(seq_lines)))
    return records


def parse_psipred_horiz(input_file: str) -> List[Tuple[str, str, str]]:
    """
    Parse PSIPRED .horiz output into (seq_id, sequence, secondary_structure).
    """
    with open(input_file, "r") as f:
        content = f.read()

    blocks = content.split("# PSIPRED HFORMAT (PSIPRED V4.0)")
    records = []
    seq_index = 1

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        aa_seq = []
        ss_seq = []

        for line in lines:
            line = line.strip()
            if line.startswith("AA:"):
                aa_seq.extend(line.replace("AA:", " ").strip().split())
            if line.startswith("Pred:"):
                ss_seq.extend(line.replace("Pred:", " ").strip().split())

        if aa_seq and ss_seq and len(aa_seq) == len(ss_seq):
            record_id = f"seq{seq_index}"
            seq_index += 1
            records.append((record_id, "".join(aa_seq), "".join(ss_seq)))

    return records


def write_fasta(records: List[Tuple[str, str]], output_path: str) -> None:
    with open(output_path, "w") as f_out:
        f_out.write("\n".join([f">{seq_id}\n{seq}" for seq_id, seq in records]))


def extract_psipred_to_fasta(
    input_file: str, sequence_output: str, structure_output: str
) -> None:
    records = parse_psipred_horiz(input_file)
    seq_records = [(seq_id, seq) for seq_id, seq, _ in records]
    ss_records = [(seq_id, ss) for seq_id, _, ss in records]

    write_fasta(seq_records, sequence_output)
    write_fasta(ss_records, structure_output)

    print("✅ 提取完成！")
    print(f"蛋白质序列保存至：{sequence_output}")
    print(f"二级结构保存至：{structure_output}")


def merge_fasta_to_csv(seq_fasta: str, ss_fasta: str, output_csv: str) -> None:
    seq_records = read_fasta_as_list(seq_fasta)
    ss_records = read_fasta_as_list(ss_fasta)

    print(f"序列文件条目数:        {len(seq_records)}")
    print(f"二级结构文件条目数:    {len(ss_records)}")

    if len(seq_records) != len(ss_records):
        print("⚠️ FASTA 条目数不一致，会按最少条目数对齐写入。")

    n = min(len(seq_records), len(ss_records))

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["seq_id", "sequence", "second_structure"])

        for i in range(n):
            seq_id, seq = seq_records[i]
            ss_id, ss = ss_records[i]

            if seq_id != ss_id:
                print(f"⚠️ Header 不匹配，第 {i + 1} 条: seq `[>{seq_id}]` vs ss `[>{ss_id}]`")

            if len(seq) != len(ss):
                print(
                    f"⚠️ 序列长度不匹配，第 {i + 1} 条 {seq_id} - sequence({len(seq)}) vs ss({len(ss)})"
                )

            writer.writerow([seq_id, seq, ss])

    print(f"✅ 已写入 {n} 条记录到 {output_csv}")


def main() -> None:
    psipred_input_file = (
        "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/"
        "UnifyImmun-main/data/data_HLA/output.fasta_all.horiz"
    )
    sequence_output = (
        "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/"
        "UnifyImmun-main/data/data_HLA/antigen.fasta"
    )
    structure_output = (
        "/data0/chenkai/data/microLM-main/downstream/Code_antigen_MHC/"
        "UnifyImmun-main/data/data_HLA/antigen_secondary_structures.fasta"
    )
    output_csv = (
        "/data0/chenkai/data/microLM-main/dataset/antigen_second/"
        "antigen_seq_ss_duotai.csv"
    )

    extract_psipred_to_fasta(psipred_input_file, sequence_output, structure_output)
    merge_fasta_to_csv(sequence_output, structure_output, output_csv)


if __name__ == "__main__":
    main()
