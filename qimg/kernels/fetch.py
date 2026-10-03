# qimg-fetch-*: CPU-качалка весов в /kaggle/working. Вывод остаётся на аккаунте и монтируется в боевое ядро.
# Набор файлов подставляется клиентом вместо __FILES_JSON__: [[repo, basename], ...] или [["url", url]].
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

__TRANSPORT_LIB__

FILES = json.loads(r'''__FILES_JSON__''')
OUT = "/kaggle/working"


def main():
    beacon("start")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_transfer"], check=True)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download, list_repo_files

    t = time.time()
    listing = {}
    for repo, name in FILES:
        if repo == "url":
            dst = os.path.join(OUT, os.path.basename(name))
            print(f"GET {name}", flush=True)
            urllib.request.urlretrieve(name, dst)
            continue
        if repo not in listing:
            listing[repo] = list_repo_files(repo)
        # путь внутри репозитория ищем по точному имени файла
        cand = [f for f in listing[repo] if os.path.basename(f) == name]
        if not cand:
            raise FileNotFoundError(f"{repo}: нет {name}; есть: {listing[repo]}")
        print(f"HF {repo}/{cand[0]}", flush=True)
        hf_hub_download(repo, cand[0], local_dir=OUT)
        beacon("got", f=name, t=int(time.time() - t))
    shutil.rmtree(os.path.join(OUT, ".cache"), ignore_errors=True)
    subprocess.run(f"find {OUT} -type f -exec ls -la {{}} \\; ; du -sh {OUT}", shell=True)
    beacon("done", t=int(time.time() - t))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        beacon("error", what="fetch", msg=f"{type(e).__name__}: {e}"[:400])
        sys.exit(1)
