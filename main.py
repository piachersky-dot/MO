import argparse
from pathlib import Path

from lab1.logistic_regression_task import run_lab1
from lab2.mlp_network import run_lab2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["lab1", "lab2"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "lab1":
        run_lab1(str(data_dir), str(out_dir))

    elif args.task == "lab2":
        run_lab2(str(data_dir), str(out_dir))


if __name__ == "__main__":
    main()