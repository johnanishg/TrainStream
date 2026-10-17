"""Build script for TrainStream.

    pip install trainstream                      # pure-Python wheel; _C compiles on first import
    python setup.py build_ext --inplace          # from a checkout (Windows: build.bat)
    pip install . --no-build-isolation           # compile _C ahead of time into the install

The C++/CUDA sources ship inside the package (trainstream/csrc). Without
PyTorch at build time (pip's default isolated build, CI) this produces a
pure-Python wheel and trainstream/_build.py compiles the extension against the
user's PyTorch on first import. With PyTorch available it compiles _C now.

Target architectures come from TORCH_CUDA_ARCH_LIST when set; otherwise the
compute capability of every visible GPU is used (e.g. "12.0" for Blackwell).
"""

import os
import sys

from setuptools import find_packages, setup

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    torch = None
if os.environ.get("TRAINSTREAM_PURE_WHEEL"):  # force the compile-on-import wheel
    torch = None


def _arch_list():
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    if torch is not None and torch.cuda.is_available():
        caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
        return ";".join(f"{major}.{minor}" for major, minor in caps)
    return "8.0;8.6;8.9;9.0;12.0"


ext_modules, cmdclass = [], {}
if torch is not None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = _arch_list()
    if sys.platform == "win32":
        cxx_flags = ["/O2", "/EHsc", "/W3", "/wd4244", "/wd4267", "/wd4251", "/wd4275", "/wd4996"]
    else:
        cxx_flags = ["-O3", "-Wall", "-Wno-unused-function", "-fvisibility=hidden"]
    nvcc_flags = ["-O3", "--use_fast_math", "-lineinfo", "--expt-relaxed-constexpr"]
    if sys.platform == "win32":
        nvcc_flags += ["-Xcompiler", "/wd4819"]
    ext_modules = [CUDAExtension(
        name="trainstream._C",
        sources=[
            "trainstream/csrc/bindings.cpp",
            "trainstream/csrc/memory_pinner.cpp",
            "trainstream/csrc/offload_adamw.cpp",
            "trainstream/csrc/offload_adamw_kernel.cu",
            "trainstream/csrc/quant.cpp",
            "trainstream/csrc/quant_kernel.cu",
        ],
        include_dirs=[os.path.join(HERE, "trainstream", "csrc")],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    )]
    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}

with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trainstream",
    version="0.4.0",
    author="John Anish G",
    license="Apache-2.0",
    description="Zero-dependency PyTorch training offload: optimizer, gradients, activations and weights to host RAM, with a memory planner and built-in int8/NF4 quantization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnanishg/TrainStream",
    project_urls={"Source": "https://github.com/johnanishg/TrainStream",
                  "Issues": "https://github.com/johnanishg/TrainStream/issues"},
    packages=find_packages(include=["trainstream", "trainstream.*"]),
    package_data={"trainstream": ["csrc/*.h", "csrc/*.cpp", "csrc/*.cu"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    # torch is not declared: users must pick the CUDA build that matches their driver.
    # ninja drives the compile-on-first-import build.
    install_requires=["ninja"],
    extras_require={"hf": ["transformers", "accelerate", "peft"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Environment :: GPU :: NVIDIA CUDA",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
