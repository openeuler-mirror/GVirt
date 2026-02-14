import os
import re
import argparse
from collections import defaultdict

def extract_metrics_from_log(log_path):
    """从单个log文件中提取关键指标"""
    metrics = {
        "ttft_avg": None,
        "ttft_p99": None,
        "tpot_avg": None,
        "tpot_p99": None,
        "qps": None,
        "output_speed": None
    }

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # 提取TTFT指标
        ttft_avg_match = re.search(r'Mean TTFT \(ms\):\s+(\d+\.\d+)', content)
        ttft_p99_match = re.search(r'P99 TTFT \(ms\):\s+(\d+\.\d+)', content)
        # 提取TPOT指标
        tpot_avg_match = re.search(r'Mean TPOT \(ms\):\s+(\d+\.\d+)', content)
        tpot_p99_match = re.search(r'P99 TPOT \(ms\):\s+(\d+\.\d+)', content)
        # 提取QPS和输出速度
        qps_match = re.search(r'Request throughput \(req/s\):\s+(\d+\.\d+)', content)
        output_speed_match = re.search(r'Output token throughput \(tok/s\):\s+(\d+\.\d+)', content)

        if ttft_avg_match:
            metrics["ttft_avg"] = float(ttft_avg_match.group(1))
        if ttft_p99_match:
            metrics["ttft_p99"] = float(ttft_p99_match.group(1))
        if tpot_avg_match:
            metrics["tpot_avg"] = float(tpot_avg_match.group(1))
        if tpot_p99_match:
            metrics["tpot_p99"] = float(tpot_p99_match.group(1))
        if qps_match:
            metrics["qps"] = float(qps_match.group(1))
        if output_speed_match:
            metrics["output_speed"] = float(output_speed_match.group(1))

    return metrics

def parse_concurrency_from_filename(filename):
    """从文件名中提取concurrency后的数字（用于排序）"""
    match = re.search(r'concurrency(\d+)\.log', filename)
    return int(match.group(1)) if match else 0

def calculate_diff(new_val, base_val):
    """计算差异百分比（保留2位小数）"""
    if base_val == 0 or new_val is None or base_val is None:
        return "N/A"
    diff = ((new_val - base_val) / base_val) * 100
    return f"{diff:.2f}%"

def validate_folder_path(path):
    """验证文件夹路径是否存在"""
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"无效的文件夹路径：{path}（路径不存在或不是文件夹）")
    return path

def main():
    # -------------------------- 命令行参数解析 --------------------------
    parser = argparse.ArgumentParser(description="提取三个文件夹的日志指标并对比，生成汇总log")

    # 必选参数：三个文件夹路径（按顺序对应 baseline、xlite-full、xlite-decode-only）
    parser.add_argument(
        "folder1",
        type=validate_folder_path,
        help="基准文件夹（对应 item: baseline-aclgraph）"
    )
    parser.add_argument(
        "folder2",
        type=validate_folder_path,
        help="对比文件夹1（对应 item: xlite-full）"
    )
    parser.add_argument(
        "folder3",
        type=validate_folder_path,
        help="对比文件夹2（对应 item: xlite-decode-only）"
    )

    # 可选参数：输出log路径（默认当前目录下的 benchmark_comparison.log）
    parser.add_argument(
        "-o", "--output",
        default="./benchmark_comparison.log",
        help="输出对比结果的log文件路径（默认：./benchmark_comparison.log）"
    )

    # 解析参数
    args = parser.parse_args()

    # 赋值给变量（与原逻辑保持一致）
    FOLDER1 = args.folder1
    FOLDER2 = args.folder2
    FOLDER3 = args.folder3
    OUTPUT_LOG = args.output
    # --------------------------------------------------------------------------------

    # 文件夹名称映射（用于表格中的item列，固定对应关系）
    folder_to_item = {
        FOLDER1: "baseline-aclgraph",
        FOLDER2: "xlite-full",
        FOLDER3: "xlite-decode-only"
    }

    # 收集所有文件夹中的log文件（key：文件名，value：{文件夹路径: 指标}）
    file_metrics = defaultdict(dict)
    folders = [FOLDER1, FOLDER2, FOLDER3]

    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".log") and "concurrency" in filename:
                log_path = os.path.join(folder, filename)
                metrics = extract_metrics_from_log(log_path)
                if any(metrics.values()):
                    file_metrics[filename][folder] = metrics

    # 筛选出三个文件夹都存在的同名文件
    valid_files = [
        filename for filename, folder_data in file_metrics.items()
    ]

    if not valid_files:
        print("错误：未找到三个文件夹共有的log文件（需文件名完全一致，且包含concurrency关键字）")
        return

    valid_files.sort(key=parse_concurrency_from_filename)

    # 生成表格内容
    table_lines = []
    # 表格头部
    table_lines.append("| maxconcurrency | item | TTFT(ms) |  | TPOT(ms) |  | QPS (req/s) | OutputSpeed (token/s) |")
    table_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    table_lines.append("|  |  | Avg | P99 | Avg | P99 |  |  |")

    for filename in valid_files:
        concurrency = parse_concurrency_from_filename(filename)
        folder_data = file_metrics[filename]

        # 获取三个文件夹的指标
        base_metrics = folder_data.get(FOLDER1, {})
        full_metrics = folder_data.get(FOLDER2, {})
        decode_only_metrics = folder_data.get(FOLDER3, {})

        # 1. 写入baseline-aclgraph行
        table_lines.append(
            f"| {concurrency} | baseline-aclgraph | "
            f"{base_metrics.get('ttft_avg', 'N/A'):.2f} | {base_metrics.get('ttft_p99', 'N/A'):.2f} | "
            f"{base_metrics.get('tpot_avg', 'N/A'):.2f} | {base_metrics.get('tpot_p99', 'N/A'):.2f} | "
            f"{base_metrics.get('qps', 'N/A'):.2f} | {base_metrics.get('output_speed', 'N/A'):.2f} |"
        )

        # 2. 写入xlite-full行
        table_lines.append(
            f"| {concurrency} | xlite-full | "
            f"{full_metrics.get('ttft_avg', 'N/A'):.2f} | {full_metrics.get('ttft_p99', 'N/A'):.2f} | "
            f"{full_metrics.get('tpot_avg', 'N/A'):.2f} | {full_metrics.get('tpot_p99', 'N/A'):.2f} | "
            f"{full_metrics.get('qps', 'N/A'):.2f} | {full_metrics.get('output_speed', 'N/A'):.2f} |"
        )

        # 3. 写入xlite-decode-only行
        table_lines.append(
            f"| {concurrency} | xlite-decode-only | "
            f"{decode_only_metrics.get('ttft_avg', 'N/A'):.2f} | {decode_only_metrics.get('ttft_p99', 'N/A'):.2f} | "
            f"{decode_only_metrics.get('tpot_avg', 'N/A'):.2f} | {decode_only_metrics.get('tpot_p99', 'N/A'):.2f} | "
            f"{decode_only_metrics.get('qps', 'N/A'):.2f} | {decode_only_metrics.get('output_speed', 'N/A'):.2f} |"
        )

        # 4. 计算并写入diff1（xlite-full 相对于 baseline-aclgraph）
        diff1_ttft_avg = calculate_diff(full_metrics.get('ttft_avg'), base_metrics.get('ttft_avg'))
        diff1_ttft_p99 = calculate_diff(full_metrics.get('ttft_p99'), base_metrics.get('ttft_p99'))
        diff1_tpot_avg = calculate_diff(full_metrics.get('tpot_avg'), base_metrics.get('tpot_avg'))
        diff1_tpot_p99 = calculate_diff(full_metrics.get('tpot_p99'), base_metrics.get('tpot_p99'))
        diff1_qps = calculate_diff(full_metrics.get('qps'), base_metrics.get('qps'))
        diff1_output_speed = calculate_diff(full_metrics.get('output_speed'), base_metrics.get('output_speed'))

        table_lines.append(
            f"| {concurrency} | diff1 | "
            f"{diff1_ttft_avg} | {diff1_ttft_p99} | "
            f"{diff1_tpot_avg} | {diff1_tpot_p99} | "
            f"{diff1_qps} | {diff1_output_speed} |"
        )

        # 5. 计算并写入diff2（xlite-decode-only 相对于 baseline-aclgraph）
        diff2_ttft_avg = calculate_diff(decode_only_metrics.get('ttft_avg'), base_metrics.get('ttft_avg'))
        diff2_ttft_p99 = calculate_diff(decode_only_metrics.get('ttft_p99'), base_metrics.get('ttft_p99'))
        diff2_tpot_avg = calculate_diff(decode_only_metrics.get('tpot_avg'), base_metrics.get('tpot_avg'))
        diff2_tpot_p99 = calculate_diff(decode_only_metrics.get('tpot_p99'), base_metrics.get('tpot_p99'))
        diff2_qps = calculate_diff(decode_only_metrics.get('qps'), base_metrics.get('qps'))
        diff2_output_speed = calculate_diff(decode_only_metrics.get('output_speed'), base_metrics.get('output_speed'))

        table_lines.append(
            f"| {concurrency} | diff2 | "
            f"{diff2_ttft_avg} | {diff2_ttft_p99} | "
            f"{diff2_tpot_avg} | {diff2_tpot_p99} | "
            f"{diff2_qps} | {diff2_output_speed} |"
        )

        table_lines.append("|  |  |  |  |  |  |  |  |")

    # 写入输出log文件
    with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
        f.write("## Qwen3 32B TPS 910B3(A2) Online Inference Performance Comparison\n")
        f.write("- aclgraph: main\n")
        f.write("- xlite-full: main + xlite-full\n")
        f.write("- xlite-decode-only: main + xlite-decode-only\n")
        f.write("- diff1: Performance comparison between xlite-full and aclgraph\n")
        f.write("- diff2: Performance comparison between xlite-decode-only and aclgraph\n\n")
        f.write('\n'.join(table_lines).rstrip('\n'))

    print(f"对比结果已成功写入：{OUTPUT_LOG}")
    print(f"共处理 {len(valid_files)} 组同名文件")
    print(f"文件夹对应关系：")
    print(f"   - baseline-aclgraph: {FOLDER1}")
    print(f"   - xlite-full: {FOLDER2}")
    print(f"   - xlite-decode-only: {FOLDER3}")

if __name__ == "__main__":
    main()