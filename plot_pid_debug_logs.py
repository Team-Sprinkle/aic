import argparse
import re
from pathlib import Path

PID_PATTERN = re.compile(
    r"Z-Offset:\s+([-?\d.]+)\s+\|\s+ErrX:\s+([-?\d.]+)\s+\(Adj:\s+([-?\d.]+)\)\s+\|\s+ErrY:\s+([-?\d.]+)\s+\(Adj:\s+([-?\d.]+)\)"
)


def parse_pid_log(log_path: Path) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    z_offsets: list[float] = []
    err_x: list[float] = []
    adj_x: list[float] = []
    err_y: list[float] = []
    adj_y: list[float] = []

    with log_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            match = PID_PATTERN.search(line)
            if not match:
                continue
            z_offsets.append(float(match.group(1)))
            err_x.append(float(match.group(2)))
            adj_x.append(float(match.group(3)))
            err_y.append(float(match.group(4)))
            adj_y.append(float(match.group(5)))

    return z_offsets, err_x, adj_x, err_y, adj_y


def plot_robot_logs(log_file: Path, output_file: Path) -> None:
    z_offsets, err_x, adj_x, err_y, adj_y = parse_pid_log(log_file)

    if not z_offsets:
        raise SystemExit(f"No PID log entries found in {log_file}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        summary_file = output_file.with_suffix(".summary.txt")
        summary_lines = [
            f"matplotlib is not installed; skipped PNG generation for {log_file}",
            f"samples={len(z_offsets)}",
            f"z_offset_range=[{min(z_offsets):.6f}, {max(z_offsets):.6f}]",
            f"err_x_range=[{min(err_x):.6f}, {max(err_x):.6f}]",
            f"err_y_range=[{min(err_y):.6f}, {max(err_y):.6f}]",
            f"adj_x_range=[{min(adj_x):.6f}, {max(adj_x):.6f}]",
            f"adj_y_range=[{min(adj_y):.6f}, {max(adj_y):.6f}]",
        ]
        summary_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"matplotlib not available; wrote PID summary to {summary_file}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(z_offsets, err_x, label="Error X", color="red", linestyle="--")
    ax1.plot(z_offsets, err_y, label="Error Y", color="blue", linestyle="--")
    ax1.set_ylabel("Error (meters)")
    ax1.set_title("Trajectory Error vs. Z-Offset")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(z_offsets, adj_x, label="Adj X (Output)", color="red")
    ax2.plot(z_offsets, adj_y, label="Adj Y (Output)", color="blue")
    ax2.set_xlabel("Z-Offset (Insertion Progress)")
    ax2.set_ylabel("PID Adjustment")
    ax2.set_title("Controller Output vs. Z-Offset")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.invert_xaxis()

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PID error logs.")
    parser.add_argument("--log-file", required=True, help="Path to the policy log file")
    parser.add_argument("--output", required=True, help="Path to write the PNG plot")
    args = parser.parse_args()

    log_file = Path(args.log_file)
    output_file = Path(args.output)

    plot_robot_logs(log_file, output_file)
    print(f"Wrote PID debug plot to {output_file}")


if __name__ == "__main__":
    main()
