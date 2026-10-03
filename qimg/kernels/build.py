# qimg-build: CPU-ядро, один раз (~36 мин). Собирает stable-diffusion.cpp под T4 (sm_75).
# Результат: /kaggle/working/sdcpp/{bin,lib} — бинари и все нужные либы настоящими файлами.
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

__TRANSPORT_LIB__

SDCPP_REPO = "https://github.com/leejet/stable-diffusion.cpp"
SDCPP_COMMIT = "3f8527a46c54ecf4cb4ed6003da8e8982283c73c"
CUDA = "/opt/cuda"
SRC = "/tmp/sdcpp-src"
OUT = "/kaggle/working/sdcpp"


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def install_cuda():
    sh("cd /tmp && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba")
    sh(f"/tmp/bin/micromamba create -y -p {CUDA} -c conda-forge cuda-version=12.8 cuda-nvcc cuda-cudart-dev "
       f"libcublas-dev cuda-driver-dev cuda-cccl")


def fetch_src():
    sh(f"git clone --filter=blob:none {SDCPP_REPO} {SRC}")
    sh(f"cd {SRC} && git checkout {SDCPP_COMMIT}")
    # сабмодули только нужные
    sh(f"cd {SRC} && git submodule update --init --depth 1 ggml thirdparty/libwebp examples/server/frontend")


def build():
    if not shutil.which("cmake"):
        sh(f"{sys.executable} -m pip install -q cmake")
    flags = ("-DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DSD_WEBM=OFF -DCMAKE_CUDA_ARCHITECTURES=75 "
             f"-DCMAKE_CUDA_COMPILER={CUDA}/bin/nvcc -DCUDAToolkit_ROOT={CUDA} "
             # GGML_NATIVE=OFF обязательно: иначе illegal instruction на GPU-машине
             "-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON")
    env = dict(os.environ, PATH=f"{CUDA}/bin:" + os.environ["PATH"])
    sh(f"cd {SRC} && cmake -B build {flags}", env=env)
    sh(f"cd {SRC} && cmake --build build --config Release --target sd-server sd-cli -j 4", env=env)


def collect():
    """Вывод ядра теряет симлинки: кладём либы настоящими файлами под soname."""
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(f"{OUT}/bin")
    os.makedirs(f"{OUT}/lib")
    bins = []
    for name in ("sd-server", "sd-cli"):
        hits = [os.path.join(d, name) for d, _, fs in os.walk(f"{SRC}/build") if name in fs]
        hits = [h for h in hits if os.access(h, os.X_OK)]
        if not hits:
            raise FileNotFoundError(name)
        shutil.copy(hits[0], f"{OUT}/bin/{name}")
        bins.append(f"{OUT}/bin/{name}")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{CUDA}/lib:{SRC}/build/bin:" + os.environ.get("LD_LIBRARY_PATH", ""))
    copied = set()
    for b in bins:
        out = subprocess.run(["ldd", b], capture_output=True, text=True, env=env).stdout
        print(out, flush=True)
        for line in out.splitlines():
            m = re.match(r"\s*(\S+)\s+=>\s+(\S+)", line)
            if not m or m.group(2) == "not":
                continue  # libcuda.so.1 на CPU-машине не найдена — её даёт Kaggle
            soname, path = m.groups()
            real = os.path.realpath(path)
            if real.startswith(CUDA) or real.startswith(SRC):
                if soname not in copied:
                    shutil.copyfile(real, f"{OUT}/lib/{soname}")
                    copied.add(soname)
    for f in os.listdir(f"{OUT}/bin"):
        os.chmod(f"{OUT}/bin/{f}", 0o755)
    print("libs:", sorted(copied), flush=True)
    sh(f"du -sh {OUT}; ls -la {OUT}/bin {OUT}/lib")


def main():
    beacon("start")
    t = time.time()
    install_cuda()
    beacon("cuda_ok", t=int(time.time() - t))
    fetch_src()
    build()
    beacon("built", t=int(time.time() - t))
    collect()
    sh(f"LD_LIBRARY_PATH={OUT}/lib {OUT}/bin/sd-server --version || true")
    beacon("done", t=int(time.time() - t))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        beacon("error", what="build", msg=f"{type(e).__name__}: {e}"[:400])
        sys.exit(1)
