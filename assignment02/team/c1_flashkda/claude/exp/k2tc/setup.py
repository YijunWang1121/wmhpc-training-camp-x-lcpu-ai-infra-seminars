import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# build (login node, no GPU needed):  cd exp/k2tc && ../../../../../.venv/bin/python setup.py build_ext --inplace
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
setup(
    name="k2_tc",
    ext_modules=[CUDAExtension(
        name=os.environ.get("K2TC","k2_tc2"),
        sources=[os.environ.get("K2TC_SRC", os.environ.get("K2TC","k2_tc2")+".cu")],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-std=c++20", "-lineinfo", "--ptxas-options=-v"] + os.environ.get("K2TC_DEFS", "").split()},
    )],
    cmdclass={"build_ext": BuildExtension},
)
