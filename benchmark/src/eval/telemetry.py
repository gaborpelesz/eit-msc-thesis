#!/usr/bin/env python3

import time
import psutil
import GPUtil
import subprocess
import matplotlib.pyplot as plt
import sys
import os
import csv
import argparse

def get_metrics(psutil_process: psutil.Process | None = None):
    # Get CPU usage and memory usage
    if psutil_process:
        cpu_usage = psutil_process.cpu_percent()
        memory_usage = psutil_process.memory_info().rss
    else:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().used
    
    # Get GPU usage and memory usage (assuming at least one GPU is present)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_usage = gpus[0].load * 100
        gpu_memory = gpus[0].memoryUsed
    else:
        gpu_usage, gpu_memory = None, None
    
    return cpu_usage, memory_usage, gpu_usage, gpu_memory

def monitor_process(command: list[str]):
    stdout, stderr = sys.stdout, sys.stderr
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=os.environ, cwd=os.getcwd())
    psutil_process = psutil.Process(process.pid)

    try:
        cpu_usage_data = []
        memory_usage_data = []
        gpu_usage_data = []
        gpu_memory_data = []
        timestamps = []
        start_time = time.time()
        
        try:
            while process.poll() is None:  # While process is running
                cpu, mem, gpu, gpu_mem = get_metrics(psutil_process)

                elapsed_time = time.time() - start_time
                timestamps.append(elapsed_time)
                cpu_usage_data.append(cpu)
                memory_usage_data.append(mem / 1024 / 1024)
                gpu_usage_data.append(gpu if gpu is not None else 0)
                gpu_memory_data.append(gpu_mem if gpu_mem is not None else 0)
                
                time.sleep(0.1)
        except (KeyboardInterrupt, psutil.NoSuchProcess) as n:
            pass
        finally:
            process.terminate()
            process.wait()
    except Exception as e:
        process.kill()
        process.wait()
        raise e

    return timestamps, cpu_usage_data, memory_usage_data, gpu_usage_data, gpu_memory_data

def plot_results(outimage, timestamps, cpu_usage, memory_usage, gpu_usage, gpu_memory):
    plt.figure(figsize=(10, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(timestamps, cpu_usage, label='CPU Usage (%)', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.title('CPU Usage Over Time')

    plt.subplot(4, 1, 2)
    plt.plot(timestamps, memory_usage, label='Memory Usage (Mb)', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Usage (Mb)')
    plt.legend()
    plt.title('Memory Usage Over Time')
    
    plt.subplot(4, 1, 3)
    plt.plot(timestamps, gpu_usage, label='GPU Usage (%)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.title('GPU Usage Over Time')

    plt.subplot(4, 1, 4)
    plt.plot(timestamps, gpu_memory, label='GPU Memory Usage (Mb)', color='m')
    plt.xlabel('Time (s)')
    plt.ylabel('Usage (Mb)')
    plt.legend()
    plt.title('GPU Memory Usage Over Time')
    
    plt.tight_layout()
    plt.savefig(outimage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", default="telemetry", help="output folder where the results will be generated")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="additional arguments to pass to the executable")

    args = parser.parse_args()

    exe = os.path.abspath(args.args[0])

    command: list[str] = args.args

    timestamps, cpu_usage, memory_usage, gpu_usage, gpu_memory = monitor_process(command)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "telemetry.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Time(s)","CPU_Usage(%)","CPU_Memory(Mb)","GPU_Usage(%)","GPU_Memory(Mb)"])
        writer.writerows(zip(timestamps, cpu_usage, memory_usage, gpu_usage, gpu_memory))
    plot_results(os.path.join(args.output, "telemetry.png"), timestamps, cpu_usage, memory_usage, gpu_usage, gpu_memory)
