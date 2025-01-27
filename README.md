# genesis_multi_gpu

#### Description
Contains python helper script allowing for completely overkill multi-GPU use of the Genesis physics engine, given a genesis python file of scripts across multiple GPUs while aggregating their FPS (Frames Per Second) performance metrics 

# Performance Metrics
- 192 *million* FPS on 8x RTX 4070 Ti Super build compared to the project's stated 43 million FPS single RTX 4090 build benchmark, which would most likely yield 344 *million* FPS if done with 8x RTX 4090 GPUs.

Link to the Genesis Project: https://github.com/Genesis-Embodied-AI/Genesis


#### Key Features
- **Parallel GPU Execution**: Runs the same script simultaneously across multiple GPUs to turn each genesis python file to become a paralleized worker
- **Automatic GPU Detection**: Uses PyTorch to detect available GPUs and distribute workload
- **FPS Monitoring**: Captures and parses FPS metrics from script outputs
- **Flexible GPU Selection**: Allows specifying number of GPUs or defaults to all available

#### Usage
```bash
python script_runner.py --script <path_to_your_script> [--gpus <number_of_gpus>]
```
[small notes]
- The GPU argument is not necessary to use all the time, by default the script will use all GPUs that are visable to NVIDIA on your system and treat them as the same type of GPU
- The tuning involving how you maximize your FPS is up to you regarding how many parallel environments and other parameters you delegate to each GPU (I used ```watch -n 1 nvidia-smi``` to monitor how many resources were being consumed)
- Upon starting up, there may be some strange errors that probably have to do with a race condition somewhere (like ```pickle truncated``` or something esoteric like that), but these went away for me when I tried the script again (make sure you clear out the processes taking up VRAM beforehand if that's a case by either killing those processes with kill -9 *process id* or just rebooting).

#### How It Works

1. **Initialization**:
- Parses command line arguments using `argparse`
- Detects available GPUs using `torch.cuda.device_count()`
- Sets up multiprocessing management for parallel execution

2. **Parallel Execution**:
- Creates separate processes for each GPU Uses environment variables to assign specific GPUs to each process
- Captures both stdout and stderr from child processes

3. **Output Processing**:
- Monitors real-time output from each GPU process
- Parses FPS information using regex
- Cleans ANSI escape sequences from output
- Aggregates FPS metrics across all GPUs

4. **Results**:
- Displays individual GPU performance
- Calculates and shows combined (approximate) performance metric (Last GPU's FPS × Number of GPUs)

#### Requirements
- Python 3.x
- PyTorch
- CUDA-compatible GPU(s)

#### Notes
- The script being run must output performance in the format "Running at XXX FPS"
- Handles errors gracefully if no GPUs are detected
- Supports both single and multi-GPU configurations
- This creates [n] different scenes based on how many GPUs you have, which may be an issue if a single scene containing all simulations is needed to do other processes. This needs a single scene to be built into this that the other scripts can access in order for that to work if that is an issue (not sure yet if that would be an issue)
