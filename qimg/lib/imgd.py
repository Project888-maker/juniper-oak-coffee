# ---- imgd: очередь с приоритетами + HTTP API на 127.0.0.1:8080 ----
import base64
import hmac
import heapq
import io
import json
import math
import os
import random
import secrets
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

MIME = {"png": "image/png", "webp": "image/webp", "jpeg": "image/jpeg"}


def _round32(x):
    return max(32, int(x) // 32 * 32)


def fit_area(w, h, max_area):
    """Сохранить пропорции, уложиться в площадь, кратно 32 вниз."""
    k = min(1.0, math.sqrt(max_area / (w * h)))
    return _round32(w * k), _round32(h * k)


def img_bytes(im, fmt="png"):
    buf = io.BytesIO()
    im.save(buf, format=fmt.upper())
    return buf.getvalue()


def has_alpha(im):
    return im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)


class Imgd:
    def __init__(self, presets, workers, api_key, out_dir="/tmp/out", rewriter=None, beacon=None, session_start=None,
                 max_session=11.5 * 3600):
        self.P = presets
        self.workers = workers  # список SDServer, индекс = карта
        self.api_key = api_key
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.rewriter = rewriter
        self.rewriter_state = "off" if rewriter is None else "preparing"
        self.beacon = beacon or (lambda *a, **k: None)
        self.t0 = session_start or time.time()
        self.max_session = max_session
        self.groups, self.jobs, self.files = {}, {}, {}
        self.order = []  # id групп по времени
        self.heap, self.seq = [], 0
        self.cv = threading.Condition()
        self.rewrite_queue = []
        self.gpu_locks = [threading.Lock() for _ in workers]
        self.busy = [None for _ in workers]
        self.paused = [False for _ in workers]  # карту занял переписчик
        self.last_activity = time.time()

    # ---------- приём заказов ----------
    def new_id(self):
        return secrets.token_hex(5)

    def submit(self, req):
        P = self.P
        preset = req.get("preset", "draft")
        if preset not in P["presets"]:
            raise ValueError(f"нет пресета {preset}")
        pp = P["presets"][preset]
        kind = pp["kind"]
        prompt = (req.get("prompt") or "").strip()
        alpha = bool(req.get("alpha"))
        src, refs = req.get("src"), [r for r in (req.get("refs") or []) if r]
        if kind == "txt2img" and not prompt:
            raise ValueError("нужен prompt")
        if kind in ("img2img", "up") and src not in self.files:
            raise ValueError("нужен src (id кадра или загрузки)")
        if kind == "edit":
            if not (1 <= len(refs) <= 3) or any(r not in self.files for r in refs):
                raise ValueError("нужно 1–3 refs (id кадров или загрузок)")
            if not prompt and not alpha:
                raise ValueError("нужна инструкция правки")
            if not prompt:
                prompt = P["remove_bg_prompt"]
        if kind in ("img2img",) and not prompt:
            prompt = self.files[src].get("prompt", "")
        aspect = req.get("aspect") or "1:1"
        if kind == "txt2img" and aspect not in P["sizes"]:
            raise ValueError(f"нет формата {aspect}")
        n = max(1, min(4, int(req.get("n") or pp.get("n", 1))))
        if kind == "up":
            n = 1
        seed = req.get("seed")
        seed = random.randint(0, 2**31 - 1) if seed in (None, "", -1) else int(seed)
        gid = self.new_id()
        want_rw = (pp.get("rewrite") and not alpha and req.get("rewrite", True)
                   and self.rewriter is not None and self.rewriter_state not in ("off", "failed"))
        g = {"id": gid, "preset": preset, "kind": kind, "prompt": prompt, "prompt_used": prompt, "neg": req.get("neg") or "",
             "aspect": aspect, "n": n, "seed": seed, "alpha": alpha, "src": src, "refs": refs, "created": time.time(),
             "rewrite": "pending" if want_rw else "skip", "jobs": []}
        with self.cv:
            self.groups[gid] = g
            self.order.append(gid)
            for i in range(n):
                jid = self.new_id()
                self.jobs[jid] = {"id": jid, "group": gid, "idx": i, "seed": seed + i, "status": "waiting" if want_rw else "queued",
                                  "progress": 0.0, "file": None, "error": None, "worker": None, "started": None,
                                  "finished": None, "cancel": threading.Event()}
                g["jobs"].append(jid)
            if want_rw:
                self.rewrite_queue.append(gid)
            else:
                self._enqueue(g)
            self.last_activity = time.time()
            self.cv.notify_all()
        return self.group_view(gid)

    def _enqueue(self, g):  # под self.cv
        prio = self.P["presets"][g["preset"]]["prio"]
        for jid in g["jobs"]:
            j = self.jobs[jid]
            if j["status"] in ("waiting", "queued"):
                j["status"] = "queued"
                self.seq += 1
                heapq.heappush(self.heap, (prio, self.seq, jid))

    def cancel(self, xid):
        with self.cv:
            ids = self.groups[xid]["jobs"] if xid in self.groups else [xid] if xid in self.jobs else []
            for jid in ids:
                j = self.jobs[jid]
                if j["status"] in ("waiting", "queued"):
                    j["status"] = "cancelled"
                elif j["status"] == "running":
                    j["cancel"].set()
            return len(ids)

    # ---------- загрузки ----------
    def upload(self, data, meta=None):
        im = Image.open(io.BytesIO(data))
        im.load()
        im = im.convert("RGBA") if has_alpha(im) else im.convert("RGB")
        m = self.P["upload_max_side"]
        if max(im.size) > m:
            k = m / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        fid = self.new_id()
        path = f"{self.out}/{fid}.png"
        im.save(path)
        self.files[fid] = {"path": path, "mime": "image/png", "w": im.width, "h": im.height, "alpha": im.mode == "RGBA",
                           **(meta or {})}
        self.last_activity = time.time()
        return {"id": fid, "w": im.width, "h": im.height, "alpha": im.mode == "RGBA"}

    def _open(self, fid):
        return Image.open(self.files[fid]["path"])

    # ---------- тело запроса к sd-server ----------
    def build_body(self, g, j):
        P, pp = self.P, self.P["presets"][g["preset"]]
        kind, fmt = pp["kind"], pp["format"]
        prompt = g["prompt_used"]
        if g["alpha"]:
            fmt = "png"
            prompt = P["alpha_prefix"] + prompt.rstrip(" .") + P["alpha_suffix"]
        body = {"prompt": prompt, "negative_prompt": g["neg"], "seed": j["seed"], "batch_count": 1,
                "sample_params": {"sample_method": "euler", "sample_steps": pp["steps"], "custom_sigmas": [],
                                  "guidance": {"txt_cfg": 1.0}},
                "lora": [], "output_format": fmt, "output_compression": 90 if fmt == "webp" else 100}
        if pp.get("cache_mode"):
            body["cache_mode"] = pp["cache_mode"]
        if pp.get("lora8"):
            body["lora"] = [{"path": P["lora_8step"], "multiplier": 1.0}]
            body["sample_params"]["custom_sigmas"] = P["sigmas_8step"]
        tiling = False
        if kind == "txt2img":
            w, h = P["sizes"][g["aspect"]]
            if pp.get("hires"):
                hr = dict(pp["hires"])
                hr["scale"] = round(min(hr["scale"], math.sqrt(P["max_area"] / (w * h))), 4)
                body["hires"] = hr
                tiling = True
        elif kind == "img2img":
            im = self._open(g["src"]).convert("RGB")
            k = pp.get("upscale", 1.0)
            w, h = fit_area(im.width * k, im.height * k, P["max_area"])
            if (w, h) != im.size:
                im = im.resize((w, h), Image.LANCZOS)
            body["init_image"] = base64.b64encode(img_bytes(im)).decode()
            body["strength"] = pp["strength"]
            tiling = True
        elif kind == "edit":
            ims = [self._open(r) for r in g["refs"]]
            w, h = fit_area(ims[0].width, ims[0].height, P["edit_max_area"])
            body["ref_images"] = [base64.b64encode(img_bytes(im.convert("RGB"))).decode() for im in ims]
        else:
            raise ValueError(kind)
        body["width"], body["height"] = w, h
        if kind != "edit" and (tiling or w * h > 1024 * 1024):
            body["vae_tiling_params"] = P["vae_tiling"]
        return body, fmt, (w, h), (2 if pp.get("hires") else 1)

    # ---------- исполнение ----------
    def _pop(self, wi):
        with self.cv:
            while True:
                while self.heap and not self.paused[wi]:
                    _, _, jid = heapq.heappop(self.heap)
                    j = self.jobs[jid]
                    if j["status"] == "queued":
                        j["status"], j["worker"], j["started"] = "running", wi, time.time()
                        self.busy[wi] = jid
                        return j
                self.cv.wait(5)

    def worker_loop(self, wi):
        sd = self.workers[wi]
        while True:
            j = self._pop(wi)
            g = self.groups[j["group"]]
            with self.gpu_locks[wi]:
                try:
                    if not sd.alive():
                        self.beacon("sd_restart", gpu=wi)
                        sd.restart()
                    self._run(sd, g, j)
                    j["status"], j["progress"] = "done", 1.0
                except Exception as e:  # noqa: BLE001  воркер не должен умирать
                    j["status"] = "cancelled" if j["cancel"].is_set() else "failed"
                    j["error"] = str(e)[:1000]
                    if j["status"] == "failed":
                        traceback.print_exc()
                    if not sd.alive():
                        self.beacon("error", what="sd_dead", gpu=wi)
                finally:
                    j["finished"] = time.time()
                    self.busy[wi] = None
                    self.last_activity = time.time()

    def _run(self, sd, g, j):
        pp = self.P["presets"][g["preset"]]

        def prog(x):
            j["progress"] = round(x, 3)

        meta = {"prompt": g["prompt"], "prompt_used": g["prompt_used"], "preset": g["preset"], "seed": j["seed"],
                "aspect": g["aspect"], "alpha": g["alpha"], "src": g["src"], "refs": g["refs"], "group": g["id"]}
        if pp["kind"] == "up":
            src = self._open(g["src"])
            alpha = src.getchannel("A") if has_alpha(src) else None
            out = Image.open(io.BytesIO(sd.upscale(img_bytes(src.convert("RGB")))))
            out.load()
            if alpha is not None:  # ESRGAN теряет альфу: увеличиваем её Lanczos отдельно
                out = out.convert("RGB")
                out.putalpha(alpha.resize(out.size, Image.LANCZOS))
            m = self.P["up_max_side"]
            if max(out.size) > m:
                k = m / max(out.size)
                out = out.resize((round(out.width * k), round(out.height * k)), Image.LANCZOS)
            data, fmt, (w, h) = img_bytes(out), "png", out.size
        else:
            body, fmt, (w, h), phases = self.build_body(g, j)
            data = sd.generate(body, cancel_event=j["cancel"], on_progress=prog, phases=phases)
            im = Image.open(io.BytesIO(data))
            w, h = im.size
        path = f"{self.out}/{j['id']}.{fmt}"
        with open(path, "wb") as f:
            f.write(data)
        meta.update(w=w, h=h, format=fmt)
        self.files[j["id"]] = {"path": path, "mime": MIME[fmt], "w": w, "h": h, "alpha": g["alpha"] or pp["kind"] == "up",
                               **meta}
        j["file"] = j["id"]
        j["meta"] = meta

    # ---------- переписчик ----------
    def rewriter_loop(self, gpu=1):
        rw = self.rewriter
        try:
            rw.prepare()
            self.rewriter_state = "ready"
        except Exception as e:  # noqa: BLE001
            self.rewriter_state = "failed"
            self.beacon("error", what="rewriter_fetch", msg=str(e)[:200])
            self._flush_rewrites()
            return
        while True:
            with self.cv:
                while not self.rewrite_queue:
                    self.cv.wait(5)
            # дождаться, пока воркер карты закончит кадр, и занять карту
            self.paused[gpu] = True
            with self.gpu_locks[gpu]:
                self.rewriter_state = "loading"
                sd = self.workers[gpu]
                sd.stop()
                loaded = False
                try:
                    rw.load()
                    loaded = True
                    self.rewriter_state = "rewriting"
                    while True:
                        with self.cv:
                            if not self.rewrite_queue:
                                break
                            gid = self.rewrite_queue.pop(0)
                        g = self.groups[gid]
                        if all(self.jobs[x]["status"] == "cancelled" for x in g["jobs"]):
                            continue
                        try:
                            text = rw.rewrite(g["prompt"], g["aspect"] if g["aspect"] != "1:1" else None)
                            if text:
                                g["prompt_used"], g["rewrite"] = text, "done"
                            else:
                                g["rewrite"] = "failed"
                        except Exception as e:  # noqa: BLE001  упал переписчик: рисуем по исходному
                            g["rewrite"] = "failed"
                            print(f"[rewriter] {e}", flush=True)
                        with self.cv:
                            self._enqueue(g)
                            self.cv.notify_all()
                except Exception as e:  # noqa: BLE001
                    self.beacon("error", what="rewriter_load", msg=str(e)[:200])
                    self._flush_rewrites()
                finally:
                    rw.unload()
                    self.rewriter_state = "reloading_sd"
                    try:
                        sd.restart()
                    except Exception as e:  # noqa: BLE001
                        self.beacon("error", what="sd_restart", gpu=gpu, msg=str(e)[:200])
                    self.rewriter_state = "ready" if loaded else "failed"
                    self.paused[gpu] = False
                    with self.cv:
                        self.cv.notify_all()
            if self.rewriter_state == "failed":
                self._flush_rewrites()
                return

    def _flush_rewrites(self):
        with self.cv:
            for gid in self.rewrite_queue:
                g = self.groups[gid]
                g["rewrite"] = "failed"
                self._enqueue(g)
            self.rewrite_queue.clear()
            self.cv.notify_all()

    # ---------- состояние ----------
    def idle_since(self):
        with self.cv:
            if self.rewrite_queue or any(b is not None for b in self.busy):
                return None
            if any(j["status"] in ("waiting", "queued", "running") for j in self.jobs.values()):
                return None
        return self.last_activity

    def job_view(self, jid):
        j = self.jobs[jid]
        v = {k: j[k] for k in ("id", "idx", "seed", "status", "progress", "file", "error", "worker", "started", "finished")}
        if j.get("meta"):
            v["meta"] = j["meta"]
        return v

    def group_view(self, gid):
        g = self.groups[gid]
        v = {k: g[k] for k in ("id", "preset", "kind", "prompt", "prompt_used", "aspect", "n", "seed", "alpha", "src",
                               "refs", "created", "rewrite")}
        v["jobs"] = [self.job_view(x) for x in g["jobs"]]
        st = [j["status"] for j in v["jobs"]]
        v["status"] = ("running" if "running" in st else "queued" if ("queued" in st or "waiting" in st)
                       else "done" if all(s in ("done", "cancelled", "failed") for s in st) else "?")
        v["est_s"] = self.P["presets"][g["preset"]].get("est_s")
        return v

    def status(self):
        now = time.time()
        idle = self.idle_since()
        return {"ok": True, "uptime": int(now - self.t0), "session_left": int(self.max_session - (now - self.t0)),
                "idle_s": None if idle is None else int(now - idle),
                "queued": sum(1 for j in self.jobs.values() if j["status"] in ("queued", "waiting")),
                "workers": [{"gpu": i, "alive": w.alive(), "job": self.busy[i]} for i, w in enumerate(self.workers)],
                "rewriter": self.rewriter_state, "presets": list(self.P["presets"]), "sizes": list(self.P["sizes"])}

    # ---------- HTTP ----------
    def serve(self, host="127.0.0.1", port=8080):
        imgd = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def _send(self, code, obj=None, raw=None, ctype="application/json"):
                data = raw if raw is not None else json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _auth(self):
                got = self.headers.get("Authorization", "")
                if hmac.compare_digest(got.encode(), f"Bearer {imgd.api_key}".encode()):
                    return True
                self._send(401, {"error": "unauthorized"})
                return False

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n > 80 * 1024 * 1024:
                    raise ValueError("слишком большое тело")
                return self.rfile.read(n) if n else b""

            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/v1/health":
                    return self._send(200, {"ok": True})
                if not self._auth():
                    return
                try:
                    if p == "/v1/status":
                        return self._send(200, imgd.status())
                    if p == "/v1/jobs":
                        return self._send(200, [imgd.group_view(g) for g in imgd.order[-50:]])
                    if p.startswith("/v1/jobs/"):
                        gid = p.rsplit("/", 1)[1]
                        if gid in imgd.groups:
                            return self._send(200, imgd.group_view(gid))
                        return self._send(404, {"error": "нет такой группы"})
                    if p.startswith("/v1/files/"):
                        fid = p.rsplit("/", 1)[1]
                        f = imgd.files.get(fid)
                        if not f:
                            return self._send(404, {"error": "нет такого файла"})
                        with open(f["path"], "rb") as fh:
                            return self._send(200, raw=fh.read(), ctype=f["mime"])
                    if p.startswith("/v1/meta/"):
                        fid = p.rsplit("/", 1)[1]
                        f = imgd.files.get(fid)
                        return self._send(200, f) if f else self._send(404, {"error": "нет такого файла"})
                    self._send(404, {"error": "not found"})
                except Exception as e:  # noqa: BLE001
                    self._send(500, {"error": str(e)})

            def do_POST(self):
                p = self.path.split("?")[0]
                if not self._auth():
                    return
                try:
                    body = self._body()
                    if p == "/v1/jobs":
                        return self._send(200, imgd.submit(json.loads(body or b"{}")))
                    if p == "/v1/upload":
                        ctype = self.headers.get("Content-Type", "")
                        if ctype.startswith("application/json"):
                            req = json.loads(body)
                            b64 = req["b64"].split(",", 1)[-1]
                            meta = {k: v for k, v in req.items() if k != "b64" and k in ("prompt", "preset", "seed", "name")}
                            return self._send(200, imgd.upload(base64.b64decode(b64), meta))
                        return self._send(200, imgd.upload(body))
                    if p.startswith("/v1/cancel/"):
                        return self._send(200, {"cancelled": imgd.cancel(p.rsplit("/", 1)[1])})
                    self._send(404, {"error": "not found"})
                except (ValueError, KeyError) as e:
                    self._send(400, {"error": str(e)})
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    self._send(500, {"error": str(e)})

        srv = ThreadingHTTPServer((host, port), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def start_workers(self):
        for i in range(len(self.workers)):
            threading.Thread(target=self.worker_loop, args=(i,), daemon=True).start()
        if self.rewriter is not None:
            threading.Thread(target=self.rewriter_loop, daemon=True).start()
# ---- /imgd ----
