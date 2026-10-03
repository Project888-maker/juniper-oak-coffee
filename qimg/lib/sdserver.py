# ---- sdserver: запуск sd-server на одну карту, задания, прогресс по логу ----
import base64
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request


def http_json(url, body=None, timeout=60, headers=None):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=h, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {url}: {e.read()[:500].decode('utf8', 'replace')}") from None


# Полоска шагов sd.cpp: "|=====>   | 3/8 - 5.12s/it" (единица прыгает между it/s и s/it)
_STEP_RE = re.compile(rb"(\d+)/(\d+)\s*-\s*[\d.]+\s*(?:it/s|s/it)")


class SDServer:
    def __init__(self, idx, port, models, sd_root="/tmp/sdcpp", lora_dir="/tmp/loras", ups_dir="/tmp/ups",
                 extra_args=None):
        self.idx = idx
        self.port = port
        self.models = models  # dict: dit, te, mmproj, vae
        self.sd_root = sd_root
        self.lora_dir = lora_dir
        self.ups_dir = ups_dir
        self.extra_args = extra_args or []
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = f"/tmp/sd{idx}.log"
        self.proc = None
        self.lock = threading.Lock()  # одна задача на карту за раз

    # -- жизненный цикл --
    def cmd(self):
        m = self.models
        return [f"{self.sd_root}/bin/sd-server",
                "--listen-ip", "127.0.0.1", "--listen-port", str(self.port), "--backend", f"cuda{self.idx}",
                "--diffusion-model", m["dit"], "--llm", m["te"], "--llm_vision", m["mmproj"], "--vae", m["vae"],
                "--fa", "--eager-load", "--threads", "2",
                "--lora-model-dir", self.lora_dir, "--hires-upscalers-dir", self.ups_dir, "--rng", "cpu",
                *self.extra_args]

    def start(self):
        env = dict(os.environ)
        # только дописывать: libcuda.so.1 Kaggle отдаёт своим путём
        env["LD_LIBRARY_PATH"] = f"{self.sd_root}/lib" + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        self.log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(self.cmd(), stdout=self.log, stderr=subprocess.STDOUT, env=env)

    def wait_ready(self, timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.alive():
                raise RuntimeError(f"sd-server cuda{self.idx} упал на старте:\n{self.log_tail()}")
            try:
                return http_json(self.base + "/sdcpp/v1/capabilities", timeout=10)
            except Exception:  # noqa: BLE001
                time.sleep(3)
        raise RuntimeError(f"sd-server cuda{self.idx} не поднялся за {timeout} с")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        # только свой процесс: pkill -f sd-server убил бы и соседа
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        self.proc = None

    def restart(self):
        self.stop()
        self.start()
        self.wait_ready()

    def log_tail(self, n=3000):
        try:
            with open(self.log_path, "rb") as f:
                f.seek(max(0, os.path.getsize(self.log_path) - n))
                return f.read().decode("utf8", "replace")
        except OSError:
            return ""

    def log_size(self):
        try:
            return os.path.getsize(self.log_path)
        except OSError:
            return 0

    # -- прогресс --
    def progress_since(self, offset, phases=1):
        """Доля 0..1 по полоскам шагов в логе после offset. Сброс счётчика шагов = новая фаза."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                chunk = f.read()
        except OSError:
            return 0.0
        phase, prev_cur, prev_tot, frac = 0, 0, 0, 0.0
        for m in _STEP_RE.finditer(chunk):
            cur, tot = int(m.group(1)), int(m.group(2))
            if tot <= 0 or cur > tot:
                continue
            if prev_tot and (tot != prev_tot or cur < prev_cur):
                phase += 1
            prev_cur, prev_tot = cur, tot
            frac = cur / tot
        if prev_tot == 0:
            return 0.0
        return max(0.0, min(0.99, (min(phase, phases - 1) + frac) / phases))

    # -- задания --
    def generate(self, body, cancel_event=None, on_progress=None, phases=1, poll=1.5, timeout=1800):
        """Отдаёт байты первой картинки. Бросает RuntimeError при failed/cancelled/таймауте."""
        offset = self.log_size()
        job = http_json(self.base + "/sdcpp/v1/img_gen", body, timeout=120)
        jid = job["id"]
        t0 = time.time()
        while True:
            time.sleep(poll)
            if cancel_event is not None and cancel_event.is_set():
                try:
                    http_json(f"{self.base}/sdcpp/v1/jobs/{jid}/cancel", {}, timeout=10)
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError("cancelled")
            if not self.alive():
                raise RuntimeError("sd-server умер во время задания:\n" + self.log_tail(600))
            st = http_json(f"{self.base}/sdcpp/v1/jobs/{jid}", timeout=60)
            s = st.get("status")
            if s == "completed":
                return base64.b64decode(st["result"]["images"][0]["b64_json"])
            if s in ("failed", "cancelled"):
                err = st.get("error") or {}
                raise RuntimeError(f"{s}: {err.get('code')} {err.get('message')}")
            if on_progress:
                on_progress(self.progress_since(offset, phases))
            if time.time() - t0 > timeout:
                raise RuntimeError("timeout")

    def upscale(self, png_bytes, output_format="png"):
        out = http_json(self.base + "/sdcpp/v1/upscale",
                        {"image": base64.b64encode(png_bytes).decode(), "output_format": output_format}, timeout=900)
        if "images" not in out:
            raise RuntimeError(f"upscale: {out}")
        return base64.b64decode(out["images"][0]["b64_json"])
# ---- /sdserver ----
