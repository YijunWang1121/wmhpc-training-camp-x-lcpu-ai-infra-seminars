#!/bin/bash
# E0: environment probe on the GPU node
cd "$(dirname "$0")/../../../.."   # assignment02/
echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,compute_cap,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,memory.total,driver_version,pstate --format=csv
nvidia-smi -q -d CLOCK | sed -n 1,40p
.venv/bin/python - <<'PY'
import torch
p=torch.cuda.get_device_properties(0)
print("torch",torch.__version__,"cuda",torch.version.cuda)
print("name",p.name,"cc",p.major,p.minor,"SMs",p.multi_processor_count,"smem/SM",p.shared_memory_per_multiprocessor,"L2",p.L2_cache_size, "regs/SM", p.regs_per_multiprocessor)
PY
which ncu nsys; ncu --version | tail -1
