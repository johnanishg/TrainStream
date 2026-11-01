"""Compile-on-first-import for trainstream._C.

`pip install trainstream` (or `uv pip install trainstream`, `uv add trainstream`)
installs a pure-Python wheel that carries the C++/CUDA sources in trainstream/csrc. The first `import trainstream` compiles them
against the PyTorch that is installed (a few minutes) and caches the result;
later imports load the cached module directly, without a compiler.

The cache lives in %LOCALAPPDATA%\\trainstream (Windows) or
~/.cache/trainstream (Linux); TRAINSTREAM_CACHE_DIR overrides it. Builds are
keyed by the sources, Python, PyTorch, CUDA and GPU architectures, so upgrading
any of them triggers a fresh build instead of loading a stale one.
"""

import glob
import hashlib
import importlib.machinery
import importlib.util
import os
import re
import shutil
import subprocess
import sys

import torch

NAME = "trainstream_C"
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
SOURCES = ["bindings.cpp", "memory_pinner.cpp", "offload_adamw.cpp", "offload_adamw_kernel.cu",
           "quant.cpp", "quant_kernel.cu", "shm.cpp", "peer.cpp"]
HEADERS = ["kernels.h", "offload_adamw.h", "trainstream.h"]

if sys.platform == "win32":
    CXX_FLAGS = ["/O2", "/EHsc", "/W3", "/wd4244", "/wd4267", "/wd4251", "/wd4275", "/wd4996"]
    NVCC_FLAGS = ["-O3", "--use_fast_math", "-lineinfo", "--expt-relaxed-constexpr", "-Xcompiler", "/wd4819"]
    LIB_EXT = ".pyd"
else:
    CXX_FLAGS = ["-O3", "-Wall", "-Wno-unused-function", "-fvisibility=hidden"]
    NVCC_FLAGS = ["-O3", "--use_fast_math", "-lineinfo", "--expt-relaxed-constexpr"]
    LIB_EXT = ".so"


class BuildError(ImportError):
    pass


def _cache_root():
    if os.environ.get("TRAINSTREAM_CACHE_DIR"):
        return os.environ["TRAINSTREAM_CACHE_DIR"]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, "trainstream", "build")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "trainstream")


def _arch_list():
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
    return ";".join(f"{major}.{minor}" for major, minor in caps)


def _build_key(arch):
    h = hashlib.sha256()
    for f in SOURCES + HEADERS:
        with open(os.path.join(CSRC, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read())
    h.update(repr((sys.version_info[:2], torch.__version__, torch.version.cuda, arch,
                   CXX_FLAGS, NVCC_FLAGS)).encode())
    tag = f"py{sys.version_info[0]}{sys.version_info[1]}-torch{torch.__version__}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tag) + "-" + h.hexdigest()[:12]


def _import_library(path):
    loader = importlib.machinery.ExtensionFileLoader(NAME, path)
    spec = importlib.util.spec_from_file_location(NAME, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _version_key(path):
    return tuple(int(p) for p in re.findall(r"\d+", os.path.basename(path)))


def _setup_cuda_home():
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or shutil.which("nvcc"):
        return
    if sys.platform == "win32":
        pattern = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                               "NVIDIA GPU Computing Toolkit", "CUDA", "v*")
    else:
        pattern = "/usr/local/cuda*"
    found = [d for d in glob.glob(pattern) if os.path.isfile(os.path.join(
        d, "bin", "nvcc.exe" if sys.platform == "win32" else "nvcc"))]
    if not found:
        return
    major = (torch.version.cuda or "").split(".")[0]
    same_major = [d for d in found if _version_key(d)[:1] == (int(major),)] if major.isdigit() else []
    os.environ["CUDA_HOME"] = max(same_major or found, key=_version_key)


def _setup_msvc():
    """Import the x64 MSVC environment (vcvars64.bat) when cl.exe is not on PATH."""
    if shutil.which("cl"):
        return
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return
    try:
        out = subprocess.check_output(
            [vswhere, "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return
    roots = [line.strip() for line in out.splitlines() if line.strip()]
    if not roots:
        return
    vcvars = os.path.join(roots[0], "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not os.path.isfile(vcvars):
        return
    try:
        env = subprocess.check_output(f'cmd /s /c ""{vcvars}" >nul && set"',
                                      text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return
    for line in env.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ[key] = value
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")


def _setup_ninja():
    if shutil.which("ninja"):
        return
    try:
        import ninja
    except ImportError:
        return
    os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")


def _hint():
    if sys.platform == "win32":
        return ("TrainStream needs the CUDA toolkit (same major version as PyTorch's CUDA, "
                "https://developer.nvidia.com/cuda-downloads) and Visual Studio 2022+ Build Tools "
                "with the 'Desktop development with C++' workload.")
    return ("TrainStream needs the CUDA toolkit (same major version as PyTorch's CUDA) "
            "and g++ 11+ (e.g. apt install build-essential).")


def load():
    """Return the compiled extension, building it first if no cached build matches."""
    if torch.version.cuda is None:
        raise BuildError(
            f"TrainStream needs a CUDA build of PyTorch, but torch {torch.__version__} is CPU-only. "
            "(On Windows, torch from PyPI is the CPU build; installing a package that depends on torch, "
            "such as transformers or accelerate, can pull it in.) Install a CUDA build, e.g.\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall\n"
            "    uv pip install torch --torch-backend=auto --reinstall-package torch\n"
            "See https://pytorch.org/get-started/locally/")
    if not torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        raise BuildError("TrainStream needs an NVIDIA GPU, but torch.cuda.is_available() is False "
                         "(set TORCH_CUDA_ARCH_LIST to compile without one).")

    arch = _arch_list()
    build_dir = os.path.join(_cache_root(), _build_key(arch))
    library = os.path.join(build_dir, NAME + LIB_EXT)
    if os.path.isfile(library):
        return _import_library(library)

    os.makedirs(build_dir, exist_ok=True)
    _setup_ninja()
    _setup_cuda_home()
    if sys.platform == "win32":
        _setup_msvc()
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch

    from torch.utils import cpp_extension
    if os.environ.get("CUDA_HOME"):
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
    if cpp_extension.CUDA_HOME is None:
        raise BuildError("TrainStream could not find the CUDA toolkit (nvcc). " + _hint()
                         + " Set CUDA_HOME if it is installed somewhere unusual.")

    print(f"TrainStream: compiling its CUDA extension for torch {torch.__version__} "
          f"(first import only, takes a few minutes; cached in {build_dir})", file=sys.stderr, flush=True)
    try:
        module = cpp_extension.load(
            name=NAME,
            sources=[os.path.join(CSRC, f) for f in SOURCES],
            extra_include_paths=[CSRC],
            extra_cflags=CXX_FLAGS,
            extra_cuda_cflags=NVCC_FLAGS,
            build_directory=build_dir,
            verbose=bool(os.environ.get("TRAINSTREAM_VERBOSE_BUILD")),
        )
    except Exception as exc:
        raise BuildError(f"TrainStream failed to compile its CUDA extension. {_hint()}\n"
                         f"Set TRAINSTREAM_VERBOSE_BUILD=1 to see the full compiler output.\n{exc}") from exc
    print("TrainStream: build finished.", file=sys.stderr, flush=True)
    return module
