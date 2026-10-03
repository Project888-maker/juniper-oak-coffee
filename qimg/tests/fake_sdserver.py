#!/usr/bin/env python3
"""Поддельный sd-server для локальных тестов: те же пути /sdcpp/v1/*, картинка нужного размера,
полоски шагов в stdout (как в логе sd.cpp). Тела запросов пишет в $FAKE_SD_BODIES."""
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

args = sys.argv[1:]
port = int(args[args.index("--listen-port") + 1])
backend = args[args.index("--backend") + 1]
assert "--rng" in args and args[args.index("--rng") + 1] == "cpu"
jobs, lock = {}, threading.Lock()
STEP_S = float(os.environ.get("FAKE_SD_STEP_S", "0.02"))
time.sleep(float(os.environ.get("FAKE_SD_BOOT_S", "0.3")))


def run(jid, body):
    j = jobs[jid]
    j["status"] = "generating"
    steps = body["sample_params"]["sample_steps"]
    phases = [steps] + ([body["hires"]["steps"]] if body.get("hires", {}).get("enabled") else [])
    for tot in phases:
        for i in range(1, tot + 1):
            if j.get("cancel"):
                j["status"] = "cancelled"
                return
            sys.stdout.write(f"\r  |{'=' * i}>| {i}/{tot} - {1.5:.2f}s/it")
            sys.stdout.flush()
            time.sleep(STEP_S)
    print(flush=True)
    if "FAIL" in body["prompt"]:
        j["status"], j["error"] = "failed", {"code": "generation_failed", "message": "fake failure"}
        return
    if "CRASH" in body["prompt"]:
        os._exit(3)
    w, h = body["width"], body["height"]
    if body.get("hires", {}).get("enabled"):
        w, h = int(w * body["hires"]["scale"]) // 8 * 8, int(h * body["hires"]["scale"]) // 8 * 8
    mode = "RGBA" if "transparent" in body["prompt"] else "RGB"
    im = Image.new(mode, (w, h), (200, 80, 40, 128) if mode == "RGBA" else (200, 80, 40))
    for x in range(0, w, 7):  # не чёрный и не плоский кадр
        im.putpixel((x, x % h), (0, 0, 0, 255) if mode == "RGBA" else (0, 0, 0))
    buf = io.BytesIO()
    fmt = body.get("output_format", "png")
    im.save(buf, format={"webp": "WEBP", "png": "PNG"}[fmt])
    j["status"] = "completed"
    j["result"] = {"output_format": fmt, "images": [{"index": 0, "b64_json": base64.b64encode(buf.getvalue()).decode()}]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        d = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(d)))
        self.end_headers()
        self.wfile.write(d)

    def do_GET(self):
        if self.path == "/sdcpp/v1/capabilities":
            return self._send(200, {"loras": [{"name": "p", "path": "p_qwen_image_2.1_8step_v0.1.safetensors"}]})
        if self.path.startswith("/sdcpp/v1/jobs/"):
            j = jobs.get(self.path.rsplit("/", 1)[1])
            if not j:
                return self._send(404, {"error": "no job"})
            return self._send(200, {k: v for k, v in j.items() if k != "cancel"})
        self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/sdcpp/v1/img_gen":
            if os.environ.get("FAKE_SD_BODIES"):
                with lock, open(os.environ["FAKE_SD_BODIES"], "a") as f:
                    slim = {k: (f"<b64:{len(v)}>" if k in ("init_image",) else
                                [f"<b64:{len(x)}>" for x in v] if k == "ref_images" else v) for k, v in body.items()}
                    f.write(json.dumps({"backend": backend, **slim}) + "\n")
            jid = f"{backend}-{len(jobs)}"
            jobs[jid] = {"id": jid, "status": "queued", "result": None, "error": None}
            threading.Thread(target=run, args=(jid, body), daemon=True).start()
            return self._send(202, {"id": jid, "status": "queued"})
        if self.path.endswith("/cancel"):
            j = jobs.get(self.path.split("/")[-2])
            if j:
                j["cancel"] = True
            return self._send(200, {"ok": True})
        if self.path == "/sdcpp/v1/upscale":
            im = Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB")
            im = im.resize((im.width * 4, im.height * 4))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return self._send(200, {"images": [{"index": 0, "b64_json": base64.b64encode(buf.getvalue()).decode()}]})
        self._send(404, {})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
