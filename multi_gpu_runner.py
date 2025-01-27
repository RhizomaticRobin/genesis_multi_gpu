import argparse
import os
import re
import subprocess
import multiprocessing
import torch

def run_script_on_gpu(script_path: str, gpu_index: int, fps_dict):
    """
    Launches the child script on the given GPU index,
    prints its output to the console, and parses lines matching
    'Running at ... FPS' for the final GPU FPS value.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    print(f"Launching '{script_path}' on GPU {gpu_index} ...")

    # Capture both stdout and stderr
    result = subprocess.run(
        ["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False
    )

    raw_output = result.stdout

    # 1) Print the child's console output so the user sees original messages
    print(raw_output, end='')  # Use "end=''" so we don't add extra newlines

    # 2) Clean up ANSI escape sequences for the parsing step
    ansi_escape_pattern = r'\x1b\[[0-9;]*m'
    cleaned_output = re.sub(ansi_escape_pattern, '', raw_output)

    # 3) Parse for pattern: "Running at ... FPS"
    fps_pattern = r"Running at\s*([\d,\.]+)\s*FPS"
    fps_matches = re.findall(fps_pattern, cleaned_output)

    if not fps_matches:
        fps_dict[gpu_index] = 0.0
    else:
        last_fps_str = fps_matches[-1].replace(",", "")
        try:
            last_fps_value = float(last_fps_str)
        except ValueError:
            last_fps_value = 0.0
        fps_dict[gpu_index] = last_fps_value
        print(f"GPU {gpu_index} last reported {last_fps_value:.2f} FPS.")

    print(f"Done running '{script_path}' on GPU {gpu_index}.\n")

def main():
    parser = argparse.ArgumentParser(
        description="Run a script on multiple GPUs in parallel, display each script's output, and parse the last FPS line."
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Path/name of the script that prints lines containing 'Running at XXX FPS'."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default uses all detected GPUs)."
    )

    args = parser.parse_args()

    gpu_count_available = torch.cuda.device_count()
    if gpu_count_available == 0:
        print("######### oh noes, no GPU(s) detected!!!###########\n")
    else:
        if args.gpus is None or args.gpus > gpu_count_available:
            gpus_to_use = gpu_count_available
        else:
            gpus_to_use = args.gpus

    manager = multiprocessing.Manager()
    fps_dict = manager.dict()

    processes = []

    for gpu_idx in range(gpus_to_use):
        p = multiprocessing.Process(
            target=run_script_on_gpu,
            args=(args.script, gpu_idx, fps_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Gather results and compute final metric
    all_fps_values = list(fps_dict.values())

    if all_fps_values:
        last_gpu_idx = gpus_to_use - 1
        last_gpu_fps = fps_dict[last_gpu_idx]
        final_value = last_gpu_fps * gpus_to_use
        print(f"\nLast GPU's FPS × Number of GPUs ≈ {final_value:,.2f} FPS\n")
    else:
        print("\nNo FPS values were collected from any process.\n")

if __name__ == "__main__":
    main()
