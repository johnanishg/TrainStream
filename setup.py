"""Build script for TrainStream.

    pip install -e . --no-build-isolation        # or
    python setup.py build_ext --inplace           # (Windows: build.bat)

Target architectures come from TORCH_CUDA_ARCH_LIST when set; otherwise the
compute capability of every visible GPU is used (e.g. "12.0" for Blackwell).
"""

import os
import sys

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as exc:  # pragma: no cover - build-time guard
    raise SystemExit("TrainStream needs PyTorch installed before building: " + str(exc))


def _arch_list():
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    if torch.cuda.is_available():
        caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
        return ";".join(f"{major}.{minor}" for major, minor in caps)
    return "8.0;8.6;8.9;9.0;12.0"


os.environ["TORCH_CUDA_ARCH_LIST"] = _arch_list()

if sys.platform == "win32":
    cxx_flags = ["/O2", "/EHsc", "/W3", "/wd4244", "/wd4267", "/wd4251", "/wd4275", "/wd4996"]
else:
    cxx_flags = ["-O3", "-Wall", "-Wno-unused-function", "-fvisibility=hidden"]

nvcc_flags = ["-O3", "--use_fast_math", "-lineinfo", "--expt-relaxed-constexpr"]
if sys.platform == "win32":
    nvcc_flags += ["-Xcompiler", "/wd4819"]

ext = CUDAExtension(
    name="trainstream._C",
    sources=[
        "csrc/bindings.cpp",
        "csrc/memory_pinner.cpp",
        "csrc/offload_adamw.cpp",
        "csrc/offload_adamw_kernel.cu",
    ],
    include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")],
    extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
)

setup(
    name="trainstream",
    version="0.1.0",
    author="John Anish G",
    license="Apache-2.0",
    description="Zero-dependency PCIe offloading engine for PyTorch training (optimizer, gradients, activations).",
    packages=find_packages(include=["trainstream", "trainstream.*"]),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.9",
    install_requires=[],  # torch is a build+runtime prerequisite, not a declared dependency
    zip_safe=False,
)
