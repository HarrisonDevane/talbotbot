import os
import sys
import time
import shutil
import subprocess
import logging
import psutil

try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


# =============================================================================
# CONFIG -- flip these to choose what gets logged. All are cheap at a 60s
# sampling interval; USS and disk I/O are marginally more expensive than RSS
# but still negligible unless you drop the interval very low.
# =============================================================================
CONFIG = {
    # CPU
    "CPU_TOTAL":        True,   # summed % across tracked processes (100% = 1 core)
    "CPU_NORMALIZED":   True,   # same, divided by logical core count (0-100% of machine)
    "SYSTEM_CPU":        True,   # whole-machine CPU%, not just this process tree
    "THREAD_COUNT":       True,   # summed thread count across tracked processes

    # RAM
    "RAM_RSS":           True,   # resident set size, sum across tracked processes
    "RAM_USS":            True,   # unique set size: TRUE private memory, excludes
    "RAM_RSS_GB":         True,   # RSS also expressed in GB (pure convenience)
    "SYSTEM_RAM":         True,   # system-wide RAM used% and available MB
    "SYSTEM_SWAP":       True,   # system-wide pagefile/swap usage

    # Disk
    "DISK_IO":            True,   # cumulative read/write MB, tracked processes
    "PROCESS_COUNT":     True,   # number of processes in the tracked tree

    # GPU (auto-disabled if no pynvml / nvidia-smi found)
    "GPU":                 True,
    "GPU_MEM_UTIL":       True,   # memory controller utilization % (distinct from mem used)
    "GPU_PROCESS_MEM":   True,   # GPU memory attributed to tracked PIDs specifically
    "GPU_TEMP":           True,
    "GPU_POWER":          True,
    "GPU_CLOCKS":         True,

    "TRACK_BUFFER_SIZE": False,
    "BUFFER_PATH":         None,
}


# =============================================================================
# GPU helpers
# =============================================================================
def init_gpu():
    """
    Returns (mode, ctx):
      mode='pynvml', ctx=list of NVML device handles  (richer: temp/power/clocks/proc-mem)
      mode='smi',    ctx=int GPU count                (fallback via nvidia-smi subprocess)
      mode=None,     ctx=None                         (no GPU monitoring available)
    """
    if _HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
                return 'pynvml', handles
        except Exception:
            pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, text=True
            )
            count = len([l for l in out.strip().splitlines() if l.strip()])
            if count > 0:
                return 'smi', count
        except Exception:
            pass
    return None, None


def get_gpu_stats(gpu_mode, gpu_ctx, config):
    """
    Returns a list of dicts, one per GPU, with whatever subset of keys the
    config/mode combination supports:
      util_pct, mem_util_pct, mem_used_mb, mem_total_mb, temp_c, power_w,
      sm_clock_mhz, mem_clock_mhz
    Missing/unsupported fields are simply absent from the dict (never guessed).
    """
    stats = []
    if gpu_mode == 'pynvml':
        for h in gpu_ctx:
            d = {}
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                d['util_pct'] = float(util.gpu)
                if config.get("GPU_MEM_UTIL"):
                    d['mem_util_pct'] = float(util.memory)
            except Exception:
                pass
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                d['mem_used_mb'] = mem.used / (1024 * 1024)
                d['mem_total_mb'] = mem.total / (1024 * 1024)
            except Exception:
                pass
            if config.get("GPU_TEMP"):
                try:
                    d['temp_c'] = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    pass
            if config.get("GPU_POWER"):
                try:
                    d['power_w'] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    pass
            if config.get("GPU_CLOCKS"):
                try:
                    d['sm_clock_mhz'] = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                    d['mem_clock_mhz'] = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
                except Exception:
                    pass
            stats.append(d)
    elif gpu_mode == 'smi':
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,"
                 "temperature.gpu,power.draw,clocks.sm,clocks.mem",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True
            )
            for line in out.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                d = {}
                try:
                    d['util_pct'] = float(p[0])
                    if config.get("GPU_MEM_UTIL"):
                        d['mem_util_pct'] = float(p[1])
                    d['mem_used_mb'] = float(p[2])
                    d['mem_total_mb'] = float(p[3])
                    if config.get("GPU_TEMP"):
                        d['temp_c'] = float(p[4])
                    if config.get("GPU_POWER"):
                        d['power_w'] = float(p[5])
                    if config.get("GPU_CLOCKS"):
                        d['sm_clock_mhz'] = float(p[6])
                        d['mem_clock_mhz'] = float(p[7])
                except (ValueError, IndexError):
                    pass
                stats.append(d)
        except Exception:
            stats = [{} for _ in range(gpu_ctx or 0)]
    return stats


def get_gpu_process_mem(gpu_mode, gpu_ctx, tracked_pids):
    """
    GPU memory attributed to specific PIDs. Reliable via NVML (per-GPU).
    In nvidia-smi fallback mode, per-GPU attribution isn't cleanly available
    from one query, so this returns a single combined total instead --
    caller should treat that case as one aggregate figure, not per-GPU.
    Returns (mode2, result):
      mode2='per_gpu' -> result = list of MB, one per GPU index
      mode2='total'   -> result = single MB float across all GPUs
      mode2=None      -> unavailable
    """
    if gpu_mode == 'pynvml':
        per_gpu = []
        for h in gpu_ctx:
            total_mb = 0.0
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except Exception:
                procs = []
            for proc in procs:
                if proc.pid in tracked_pids and proc.usedGpuMemory:
                    total_mb += proc.usedGpuMemory / (1024 * 1024)
            per_gpu.append(total_mb)
        return 'per_gpu', per_gpu
    elif gpu_mode == 'smi':
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True
            )
            total_mb = 0.0
            for line in out.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2:
                    try:
                        pid = int(parts[0])
                        mem_mb = float(parts[1])
                        if pid in tracked_pids:
                            total_mb += mem_mb
                    except ValueError:
                        pass
            return 'total', total_mb
        except Exception:
            return 'total', 0.0
    return None, None


# =============================================================================
# Process discovery / sizing helpers
# =============================================================================
def find_process_by_name(name_fragment):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any(name_fragment in arg for arg in cmdline):
                if 'performance_metrics.py' not in "".join(cmdline):
                    return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None


def get_path_size_mb(path):
    """Size of a file, or recursive total of a directory, in MB. None on error."""
    try:
        if os.path.isfile(path):
            return os.path.getsize(path) / (1024 * 1024)
        elif os.path.isdir(path):
            total = 0
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
            return total / (1024 * 1024)
    except OSError:
        pass
    return None


# =============================================================================
# Log message construction -- pipe-delimited groups, matching the trainer's
# own log line style (e.g. "Step X | Loss: ... | Data=Xms FW=Yms BW=Zms").
# Built entirely from CONFIG + num_gpus so it can't drift from what's enabled.
# =============================================================================
def setup_logger(log_path):
    logger = logging.getLogger("performance_metrics")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def build_log_message(config, num_gpus, gpu_mode, ctx):
    """ctx is a dict of precomputed values for this sample; see the main loop."""
    parts = []

    cpu_bits = []
    if config["CPU_TOTAL"]:
        cpu_bits.append(f"All={ctx['cpu_total']:.2f}%")
    if config["CPU_NORMALIZED"]:
        cpu_bits.append(f"Norm={ctx['cpu_normalized']:.2f}%")
    if config["SYSTEM_CPU"]:
        cpu_bits.append(f"Sys={ctx['system_cpu']:.2f}%")
    if cpu_bits:
        parts.append("CPU: " + " ".join(cpu_bits))

    proc_bits = []
    if config["PROCESS_COUNT"]:
        proc_bits.append(f"Count={ctx['process_count']}")
    if config["THREAD_COUNT"]:
        proc_bits.append(f"Threads={ctx['thread_count']}")
    if proc_bits:
        parts.append("Proc: " + " ".join(proc_bits))

    ram_bits = []
    if config["RAM_RSS"]:
        ram_bits.append(f"RSS={ctx['ram_rss_mb']:.2f}MB")
    if config["RAM_USS"]:
        ram_bits.append(f"USS={ctx['ram_uss_mb']:.2f}MB")
    if config["RAM_RSS_GB"]:
        ram_bits.append(f"RSS_GB={ctx['ram_rss_mb'] / 1024:.2f}")
    if config["SYSTEM_RAM"]:
        ram_bits.append(f"SysUsed={ctx['sys_ram_pct']:.2f}% SysAvail={ctx['sys_ram_avail_mb']:.2f}MB")
    if config["SYSTEM_SWAP"]:
        ram_bits.append(f"Swap={ctx['sys_swap_mb']:.2f}MB")
    if ram_bits:
        parts.append("RAM: " + " ".join(ram_bits))

    if config["DISK_IO"]:
        parts.append(f"Disk: R={ctx['disk_read_mb']:.2f}MB W={ctx['disk_write_mb']:.2f}MB")

    if config["GPU"] and num_gpus > 0:
        gpu_stats = ctx.get("gpu_stats", [])
        proc_mem_mode, proc_mem_val = ctx.get("gpu_proc_mem", (None, None))
        for i in range(num_gpus):
            g = gpu_stats[i] if i < len(gpu_stats) else {}
            bits = []
            if 'util_pct' in g:
                bits.append(f"Util={g['util_pct']:.2f}%")
            if config["GPU_MEM_UTIL"] and 'mem_util_pct' in g:
                bits.append(f"MemUtil={g['mem_util_pct']:.2f}%")
            if 'mem_used_mb' in g and 'mem_total_mb' in g:
                bits.append(f"Mem={g['mem_used_mb']:.2f}/{g['mem_total_mb']:.2f}MB")
            if config["GPU_TEMP"] and 'temp_c' in g:
                bits.append(f"Temp={g['temp_c']:.1f}C")
            if config["GPU_POWER"] and 'power_w' in g:
                bits.append(f"Power={g['power_w']:.1f}W")
            if config["GPU_CLOCKS"] and 'sm_clock_mhz' in g:
                bits.append(f"SM={g['sm_clock_mhz']:.0f}MHz MemClk={g.get('mem_clock_mhz', 0):.0f}MHz")
            if config["GPU_PROCESS_MEM"]:
                if gpu_mode == 'pynvml' and proc_mem_mode == 'per_gpu' and i < len(proc_mem_val):
                    bits.append(f"ProcMem={proc_mem_val[i]:.2f}MB")
                elif gpu_mode == 'smi' and proc_mem_mode == 'total' and i == 0:
                    bits.append(f"ProcMemTotal={proc_mem_val:.2f}MB")
            if bits:
                parts.append(f"GPU{i}: " + " ".join(bits))

    if config["TRACK_BUFFER_SIZE"] and config["BUFFER_PATH"]:
        size = get_path_size_mb(config["BUFFER_PATH"])
        parts.append(f"Buffer={size:.2f}MB" if size is not None else "Buffer=N/A")

    return " | ".join(parts)


# =============================================================================
# Main
# =============================================================================
def monitor_process(interval=60.0, train_dir=None, config=None):
    config = config or CONFIG
    if not train_dir:
        raise ValueError("train_dir is required (pass it in, or run this script with the train_dir as its first argument).")

    train_dir = os.path.abspath(train_dir)
    os.makedirs(train_dir, exist_ok=True)
    log_path = os.path.join(train_dir, "performance_metrics.log")
    logger = setup_logger(log_path)

    logger.info(f"=== PERFORMANCE METRICS MONITOR STARTED === (train_dir: {train_dir})")
    logger.info("Searching for running training process (main_train.py)...")

    target_pid = None
    while target_pid is None:
        target_pid = find_process_by_name("main_train.py")
        if target_pid is None:
            logger.info("Training process not found yet. Retrying in 5 seconds...")
            time.sleep(5)

    logger.info(f"Found training process with native PID: {target_pid}")

    try:
        main_proc = psutil.Process(target_pid)
    except psutil.NoSuchProcess:
        logger.critical(f"Process with PID {target_pid} disappeared before monitoring could start.")
        return

    logical_cores = psutil.cpu_count(logical=True) or 1

    gpu_mode, gpu_ctx = init_gpu() if config["GPU"] else (None, None)
    num_gpus = len(gpu_ctx) if gpu_mode == 'pynvml' else (gpu_ctx if gpu_mode == 'smi' else 0)
    if config["GPU"] and gpu_mode is None:
        logger.warning("GPU monitoring unavailable (no pynvml and no nvidia-smi found on PATH). Skipping GPU stats.")
    elif config["GPU"]:
        logger.info(f"GPU monitoring enabled via {gpu_mode} ({num_gpus} GPU(s) detected). "
                    f"Utilization/temp/power/clocks are whole-GPU (all processes); "
                    f"GPU_ProcMem is attributed to the tracked process tree specifically.")

    if config["TRACK_BUFFER_SIZE"] and not config["BUFFER_PATH"]:
        logger.warning("TRACK_BUFFER_SIZE is on but BUFFER_PATH is not set -- buffer size will be skipped.")

    logger.info(f"Monitoring PID {target_pid} + children (sampling every {interval:.0f}s)")

        # Persistent registry of tracked Process objects, keyed by pid.
        # cpu_percent(interval=None) keeps its delta baseline on the Process
        # *object*, not the pid -- so children must be tracked as the same
        # object across iterations, or every reading for them is stuck at 0.0.
    tracked = {main_proc.pid: main_proc}
    main_proc.cpu_percent(interval=None)  # prime baseline
    for child in main_proc.children(recursive=True):
        try:
            child.cpu_percent(interval=None)  # prime baseline
            tracked[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.cpu_percent(interval=None)  # prime system-wide baseline

    try:
        while main_proc.is_running():
            # Snapshot pids eligible for a CPU reading THIS iteration --
            # i.e. pids that already had a ~60s-old baseline going into
            # this loop body. Pids discovered below get primed but are
            # deliberately excluded from this round's CPU sum, because
            # their only baseline would be ~1s old (set right before the
            # sleep below) instead of ~60s old, which would make their
            # contribution to cpu_total wildly overweighted relative to
            # everyone else's honestly-windowed reading.
            eligible_for_cpu = set(tracked.keys())

            # Refresh tracked-process registry: add new children (primed,
            # but not yet eligible), drop ones that no longer exist.
            try:
                current_children = main_proc.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                current_children = []

            current_pids = {main_proc.pid}
            for child in current_children:
                current_pids.add(child.pid)
                if child.pid not in tracked:
                    try:
                        child.cpu_percent(interval=None)  # prime baseline only
                        tracked[child.pid] = child
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            for dead_pid in (set(tracked.keys()) - current_pids):
                del tracked[dead_pid]
                eligible_for_cpu.discard(dead_pid)

            # Sleep for the measurement window so cpu_percent has a delta window
            time.sleep(1.0)

            cpu_total = 0.0
            ram_rss_bytes = 0
            ram_uss_bytes = 0
            thread_count = 0
            disk_read_bytes = 0
            disk_write_bytes = 0

            for pid, p in list(tracked.items()):
                if pid in eligible_for_cpu:
                    try:
                        cpu_total += p.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                # else: skip the cpu_percent call entirely this round --
                # its baseline stays at priming time, so next iteration
                # it will have a proper ~60s window instead of ~1s.
                try:
                    ram_rss_bytes += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if config["RAM_USS"]:
                    try:
                        ram_uss_bytes += p.memory_full_info().uss
                    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                        pass
                if config["THREAD_COUNT"]:
                    try:
                        thread_count += p.num_threads()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if config["DISK_IO"]:
                    try:
                        io = p.io_counters()
                        disk_read_bytes += io.read_bytes
                        disk_write_bytes += io.write_bytes
                    except (psutil.NoSuchProcess, psutil.AccessDenied, NotImplementedError):
                        pass

            ctx = {
                "timestamp": time.strftime("%H:%M:%S"),
                "cpu_total": cpu_total,
                "cpu_normalized": cpu_total / logical_cores,
                "system_cpu": psutil.cpu_percent(interval=None) if config["SYSTEM_CPU"] else 0.0,
                "thread_count": thread_count,
                "ram_rss_mb": ram_rss_bytes / (1024 * 1024),
                "ram_uss_mb": ram_uss_bytes / (1024 * 1024),
                "disk_read_mb": disk_read_bytes / (1024 * 1024),
                "disk_write_mb": disk_write_bytes / (1024 * 1024),
                "process_count": len(tracked),
            }

            if config["SYSTEM_RAM"]:
                vm = psutil.virtual_memory()
                ctx["sys_ram_pct"] = vm.percent
                ctx["sys_ram_avail_mb"] = vm.available / (1024 * 1024)
            if config["SYSTEM_SWAP"]:
                sw = psutil.swap_memory()
                ctx["sys_swap_mb"] = sw.used / (1024 * 1024)

            if config["GPU"] and num_gpus > 0:
                ctx["gpu_stats"] = get_gpu_stats(gpu_mode, gpu_ctx, config)
                if config["GPU_PROCESS_MEM"]:
                    ctx["gpu_proc_mem"] = get_gpu_process_mem(gpu_mode, gpu_ctx, set(tracked.keys()))

            message = build_log_message(config, num_gpus, gpu_mode, ctx)
            logger.info(message)

            # Remainder of the interval
            time.sleep(max(0.0, interval - 1.0))

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        logger.warning(f"Process {target_pid} has terminated. Stopping monitor.")
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user (KeyboardInterrupt).")
    finally:
        if gpu_mode == 'pynvml':
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: performance_metrics.py <train_dir>", file=sys.stderr)
        sys.exit(1)
    monitor_process(interval=60.0, train_dir=sys.argv[1])