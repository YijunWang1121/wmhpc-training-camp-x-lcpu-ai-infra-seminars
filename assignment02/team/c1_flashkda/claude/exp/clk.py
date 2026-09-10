import subprocess
def sm_mhz():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]).decode().split()[0])
    except Exception:
        return -1
