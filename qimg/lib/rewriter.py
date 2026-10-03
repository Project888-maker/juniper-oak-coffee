# ---- rewriter: Qwen-Image-2.1-PE-T2I через llama-server на GPU1, по времени ----
import glob
import json
import os
import re
import subprocess
import tarfile
import time
import urllib.request

LLAMA_RELEASES = "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest"
LLAMA_ASSET_SUFFIX = "cuda-12.8-amd64.tar.gz"


def fetch_llama(dest="/tmp/llama"):
    """Готовая CUDA-сборка llama.cpp. Возвращает путь к llama-server."""
    found = glob.glob(f"{dest}/**/llama-server", recursive=True)
    if found:
        return found[0]
    rel = json.loads(urllib.request.urlopen(LLAMA_RELEASES, timeout=60).read())
    asset = next(a for a in rel["assets"] if a["name"].endswith(LLAMA_ASSET_SUFFIX))
    tgz = "/tmp/llama.tar.gz"
    urllib.request.urlretrieve(asset["browser_download_url"], tgz)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tgz) as t:
        t.extractall(dest)
    os.remove(tgz)
    path = glob.glob(f"{dest}/**/llama-server", recursive=True)[0]
    os.chmod(path, 0o755)
    return path


def extract_rewritten(text):
    """Из ответа модели достать rewritten_prompt (JSON может быть в ```-блоке или после <think>)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for m in re.finditer(r"\{", text):
        depth = 0
        for j in range(m.start(), len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[m.start():j + 1])
                    except ValueError:
                        break
                    if isinstance(obj, dict) and obj.get("rewritten_prompt"):
                        return obj["rewritten_prompt"].strip()
                    break
    m = re.search(r'"rewritten_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S)
    if m:
        return json.loads('"' + m.group(1) + '"').strip()
    return None


class Rewriter:
    def __init__(self, model, system_prompt_path, gpu=1, port=8090):
        self.model = model
        self.system_prompt = open(system_prompt_path, encoding="utf8").read()
        self.gpu = gpu
        self.port = port
        self.bin = None
        self.proc = None

    def prepare(self):
        self.bin = fetch_llama()

    def load(self, timeout=300):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        d = os.path.dirname(self.bin)
        env["LD_LIBRARY_PATH"] = d + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        log = open("/tmp/llama.log", "ab")
        self.proc = subprocess.Popen(
            [self.bin, "-m", self.model, "-c", "12288", "-ngl", "99", "-fa", "on", "--jinja",
             "--host", "127.0.0.1", "--port", str(self.port)],
            stdout=log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server упал на старте")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5).read()
                return
            except Exception:  # noqa: BLE001
                time.sleep(2)
        raise RuntimeError("llama-server не поднялся")

    def unload(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def rewrite(self, prompt, aspect=None, timeout=300):
        user = prompt + (f"\n\nAspect ratio: {aspect}" if aspect else "")
        body = {"messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}],
                "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 8192}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return extract_rewritten(out["choices"][0]["message"]["content"])
# ---- /rewriter ----
