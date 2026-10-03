#!/usr/bin/env python3
"""qimg — клиент Qwen-Image-2.1 на бесплатных 2×T4 Kaggle.

  qimg.py init --user NICK [--token KGAT_...|--kaggle-json PATH]
  qimg.py setup                 # один раз: сборка sd.cpp + качалки весов (CPU-ядра)
  qimg.py stub                  # проверить канал на CPU-заглушке (ноль GPU-квоты)
  qimg.py up | down | status
  qimg.py gen "кот в космосе" [-p draft|std|hq] [-a 16:9] [-n 2] [--alpha] [-i ref.png ...]
  qimg.py refine|vary|up4x|rmbg <файл или id> ;  qimg.py edit <файл/id ...> "инструкция"
  qimg.py ui                    # локальная страница на 127.0.0.1:7861
"""
import argparse
import base64
import datetime as dt
import json
import mimetypes
import os
import re
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(ROOT, "config.json")
SECRETS = os.path.join(ROOT, "secrets")
STATE = os.path.join(ROOT, "state")
OUT = os.path.join(ROOT, "out")
BUILD = os.path.join(ROOT, "build")
NTFY = "https://ntfy.sh"

FETCH_SERVE = [
    ["Comfy-Org/Qwen-Image-2.1", "qwen_image_2.1_int8_convrot.safetensors"],
    ["Comfy-Org/Qwen-Image-2.1", "qwen_image_2.1_vae_bf16.safetensors"],
    ["PrunaAI/Pruna-Qwen-Image-2.1", "p_qwen_image_2.1_8step_v0.1.safetensors"],
    ["pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF", "qwen3vl_8b_heretic-Q4_K_M.gguf"],
    ["pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF", "mmproj-qwen3vl_8b_heretic-f16.gguf"],
    ["url", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"],
]
FETCH_PE = [
    ["pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF", "pe_t2i_heretic-Q4_K_M.gguf"],
    ["pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF", "system_prompt.txt"],
]
# kind -> (шаблон, gpu, источники, подстановка файлов)
KERNELS = {
    "build": ("build.py", False, [], None),
    "fetch-serve": ("fetch.py", False, [], FETCH_SERVE),
    "fetch-pe": ("fetch.py", False, [], FETCH_PE),
    "stub": ("stub.py", False, [], None),
    "serve": ("serve.py", True, ["build", "fetch-serve", "fetch-pe"], None),
}
LIBS = {"__TRANSPORT_LIB__": "transport.py", "__SDSERVER_LIB__": "sdserver.py",
        "__REWRITER_LIB__": "rewriter.py", "__IMGD_LIB__": "imgd.py"}


# ---------------- конфиг и секреты ----------------
def load_cfg():
    if not os.path.exists(CONFIG):
        sys.exit("нет config.json — сначала: qimg.py init --user <ник на Kaggle>")
    with open(CONFIG) as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def cmd_init(a):
    cfg = load_json(CONFIG, {})
    cfg["kaggle_user"] = a.user
    cfg.setdefault("api_key", secrets.token_urlsafe(32))
    cfg.setdefault("topic", "qimg-" + secrets.token_hex(8))
    save_json(CONFIG, cfg)
    os.chmod(CONFIG, 0o600)
    cred = None
    if a.token:
        cred = {"username": a.user, "token": a.token.strip()} if a.token.strip().startswith("KGAT_") else \
            {"username": a.user, "key": a.token.strip()}
    elif a.kaggle_json:
        cred = load_json(a.kaggle_json, None)
        if not cred:
            sys.exit(f"не читается {a.kaggle_json}")
    if cred:
        os.makedirs(SECRETS, exist_ok=True)
        os.chmod(SECRETS, 0o700)
        save_json(os.path.join(SECRETS, "kaggle.json"), cred)
        os.chmod(os.path.join(SECRETS, "kaggle.json"), 0o600)
    print(f"ок: ник {a.user}, топик {cfg['topic']}" + (", токен сохранён в secrets/" if cred else ""))


KAGGLE_API = "https://api.kaggle.com/v1/kernels.KernelsApiService/"


def kaggle_token():
    """Токен из окружения или secrets/. Если его нет, заголовок подставляет прокси облачной среды."""
    tok = os.environ.get("KAGGLE_API_TOKEN")
    if not tok:
        cred = load_json(os.path.join(SECRETS, "kaggle.json"), None) or {}
        tok = cred.get("token")
    return tok if tok and "proxy" not in tok else None


def kapi(method, body, timeout=120):
    """Прямой вызов Kaggle API (без утилиты kaggle: ей нужен токен локально)."""
    headers = {"Content-Type": "application/json", "User-Agent": "qimg"}
    tok = kaggle_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(KAGGLE_API + method, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read()).get("error", {})
        except ValueError:
            err = {}
        raise KaggleError(e.code, err.get("message", str(e))) from None


class KaggleError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(f"Kaggle API {code}: {msg}")
        self.code = code


# ---------------- сборка исходника ядра ----------------
def render(kind, cfg):
    tpl, gpu, sources, files = KERNELS[kind]
    src = open(os.path.join(ROOT, "kernels", tpl), encoding="utf8").read()
    for mark, lib in LIBS.items():
        if mark in src:
            src = src.replace(mark, open(os.path.join(ROOT, "lib", lib), encoding="utf8").read())
    presets = open(os.path.join(ROOT, "presets.json"), encoding="utf8").read()
    subs = {"__TOPIC__": cfg["topic"], "__KIND__": kind, "__API_KEY__": cfg["api_key"],
            "__PRESETS_JSON__": presets, "__FILES_JSON__": json.dumps(files or [])}
    for k, v in subs.items():
        src = src.replace(k, v)
    left = re.findall(r"__[A-Z_]+_(?:LIB|JSON)__", src)
    if left:
        raise RuntimeError(f"не подставлено: {left}")
    user = cfg["kaggle_user"]
    meta = {"id": f"{user}/qimg-{kind}", "title": f"qimg-{kind}", "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [f"{user}/qimg-{s}" for s in sources]}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    d = os.path.join(BUILD, kind)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "kernel.py"), "w", encoding="utf8") as f:
        f.write(src)
    save_json(os.path.join(d, "kernel-metadata.json"), meta)
    return d


def kernel_status(kind, cfg):
    try:
        r = kapi("GetKernelSessionStatus", {"userName": cfg["kaggle_user"], "kernelSlug": f"qimg-{kind}"})
    except KaggleError as e:
        return "none" if e.code in (403, 404) else f"? ({e})"
    st = (r.get("status") or "unknown").lower()
    return st + (f": {r['failureMessage']}" if r.get("failureMessage") else "")


def push(kind, cfg):
    d = render(kind, cfg)
    meta = load_json(os.path.join(d, "kernel-metadata.json"), {})
    body = {"slug": meta["id"], "newTitle": meta["title"], "text": open(os.path.join(d, "kernel.py"), encoding="utf8").read(),
            "language": "python", "kernelType": "script", "isPrivate": True, "enableGpu": meta["enable_gpu"],
            "enableTpu": False, "enableInternet": True, "datasetDataSources": [], "competitionDataSources": [],
            "kernelDataSources": meta["kernel_sources"], "modelDataSources": [], "categoryIds": []}
    if meta.get("machine_shape"):
        body["machineShape"] = meta["machine_shape"]
    t = time.time()
    r = kapi("SaveKernel", body)
    if r.get("error"):
        sys.exit(f"Kaggle не принял ядро qimg-{kind}: {r['error']}")
    print(f"qimg-{kind}: версия {r.get('versionNumber')} — {r.get('url')}")
    log_launch(kind, t)
    return t


def quota():
    """Настоящая квота GPU из Kaggle: (сожжено ч, всего ч, когда обнулится)."""
    r = kapi("GetAcceleratorQuotaStatistics", {})
    g = r.get("gpuQuota", {})
    sec = lambda v: float(str(v or "0s").rstrip("s") or 0)  # noqa: E731
    return round(sec(g.get("timeUsed")) / 3600, 2), round(sec(g.get("totalTimeAllowed")) / 3600, 1), r.get("quotaRefreshTime")


def log_launch(kind, t):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, "launches.jsonl"), "a") as f:
        f.write(json.dumps({"kind": kind, "t": t}) + "\n")


# ---------------- маяк (ntfy) ----------------
def parse_msg(m):
    stage, _, rest = m.get("message", "").partition("&")
    kv = dict(urllib.parse.parse_qsl(rest))
    return {"id": m.get("id"), "time": m.get("time", 0), "stage": stage, **kv}


def beacons(cfg, since=None):
    """Читает топик и зеркалит в state/beacon.jsonl (ntfy хранит недолго). Возвращает все известные события."""
    path = os.path.join(STATE, "beacon.jsonl")
    known = []
    if os.path.exists(path):
        with open(path) as f:
            known = [json.loads(x) for x in f if x.strip()]
    ids = {e["id"] for e in known}
    s = "all" if since is None else str(int(since))
    try:
        body = urllib.request.urlopen(f"{NTFY}/{cfg['topic']}/json?poll=1&since={s}", timeout=30).read().decode()
    except Exception as e:  # noqa: BLE001
        print(f"[ntfy] {e}", file=sys.stderr)
        body = ""
    new = []
    for line in body.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("event") == "message" and m.get("id") not in ids:
            e = parse_msg(m)
            new.append(e)
            ids.add(e["id"])
    if new:
        os.makedirs(STATE, exist_ok=True)
        with open(path, "a") as f:
            for e in new:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    allv = sorted(known + new, key=lambda e: e["time"])
    return allv


def fmt_event(e):
    extra = " ".join(f"{k}={v}" for k, v in e.items() if k not in ("id", "time", "stage", "k"))
    return f"{dt.datetime.fromtimestamp(e['time']).strftime('%H:%M:%S')} [{e.get('k', '?')}] {e['stage']} {extra}".rstrip()


def wait_stage(cfg, kinds, done_stages, since, timeout, quiet=False):
    """Ждёт событий только после since (иначе поймаем ready прошлого прогона)."""
    seen, t0, result = set(), time.time(), {}
    kinds = set(kinds)
    while time.time() - t0 < timeout:
        for e in beacons(cfg, since - 5):
            if e["time"] < int(since) or e.get("k") not in kinds or e["id"] in seen:
                continue
            seen.add(e["id"])
            if not quiet:
                print(fmt_event(e), flush=True)
            if e["stage"] in done_stages or e["stage"] == "shutdown" or (e["stage"] == "error" and e.get("what") in
                                                                            ("fatal", "build", "fetch")):
                result[e["k"]] = e
        if kinds <= set(result):
            return result
        time.sleep(5)
    return result


def ntfy_post(topic, msg):
    urllib.request.urlopen(urllib.request.Request(f"{NTFY}/{topic}", data=msg.encode(), method="POST"), timeout=20).read()


# ---------------- сессия и квота ----------------
def week_start(now=None):
    """Начало недели квоты Kaggle (суббота 00:00 UTC; поменять в config.json: quota_week_start_weekday)."""
    cfg = load_json(CONFIG, {})
    wd = cfg.get("quota_week_start_weekday", 5)  # 0=пн ... 5=сб
    now = now or dt.datetime.now(dt.timezone.utc)
    d = now - dt.timedelta(days=(now.weekday() - wd) % 7)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def sessions(events):
    """Сессии боевого ядра по маяку: [start, end|None, last_stage, url]."""
    out, cur = [], None
    for e in events:
        if e.get("k") != "serve":
            continue
        if e["stage"] == "start":
            if cur:
                out.append(cur)
            cur = {"start": e["time"], "end": None, "stage": "start", "url": None}
        elif cur:
            cur["stage"] = e["stage"] if e["stage"] != "error" else cur["stage"]
            if e["stage"] == "ready":
                cur["url"] = e.get("url")
            if e["stage"] == "error":
                cur["last_error"] = f"{e.get('what')}: {e.get('msg', '')}"
            if e["stage"] == "shutdown":
                cur["end"] = e["time"]
                cur["reason"] = e.get("reason")
                out.append(cur)
                cur = None
    if cur:
        out.append(cur)
    return out


def session_state(cfg, refresh=True):
    ev = beacons(cfg) if refresh else beacons_local()
    ss = sessions(ev)
    now = time.time()
    ws = week_start()
    used = 0
    for s in ss:
        end = s["end"] or (now if s is ss[-1] else s["start"])
        # ядро могло умереть без shutdown — не считаем больше потолка сессии
        end = min(end, s["start"] + 12 * 3600)
        if end > ws:
            used += end - max(s["start"], ws)
    cur = ss[-1] if ss and ss[-1]["end"] is None and now - ss[-1]["start"] < 12 * 3600 else None
    return {"running": bool(cur), "stage": cur["stage"] if cur else "off", "url": cur["url"] if cur else None,
            "since": cur["start"] if cur else None, "last_error": (cur or {}).get("last_error"),
            "quota_used_h": round(used / 3600, 2), "quota_left_h": round(30 - used / 3600, 2)}


def beacons_local():
    path = os.path.join(STATE, "beacon.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return sorted((json.loads(x) for x in f if x.strip()), key=lambda e: e["time"])


# ---------------- команды ядер ----------------
def cmd_setup(a):
    cfg = load_cfg()
    kinds = a.only or ["build", "fetch-serve", "fetch-pe"]
    t = time.time()
    for k in kinds:
        print(f"push qimg-{k}")
        push(k, cfg)
    print("ждём завершения (сборка ~36 мин, сохранение вывода качалки ~20 мин)…")
    res = wait_stage(cfg, kinds, {"done"}, t, timeout=3 * 3600)
    for k in kinds:
        e = res.get(k)
        print(f"{k}: {'готово' if e and e['stage'] == 'done' else 'ОШИБКА' if e else 'нет ответа'}")
    print("Дождитесь, пока Kaggle сохранит вывод ядер (qimg.py kstatus), потом qimg.py up")


def cmd_kstatus(a):
    cfg = load_cfg()
    for k in a.kinds or KERNELS:
        print(f"qimg-{k}: {kernel_status(k, cfg)}")


def cmd_logs(a):
    cfg = load_cfg()
    d = os.path.join(STATE, "output", a.kind)
    os.makedirs(d, exist_ok=True)
    r = kapi("ListKernelSessionOutput", {"userName": cfg["kaggle_user"], "kernelSlug": f"qimg-{a.kind}"})
    with open(os.path.join(d, "log.txt"), "w", encoding="utf8") as f:
        f.write(r.get("log") or "")
    for fobj in r.get("files") or []:
        name = fobj.get("fileName", "")
        if a.all or name.startswith("logs/") or name.endswith((".txt", ".log")):
            dst = os.path.join(d, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                urllib.request.urlretrieve(fobj["url"], dst)
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: {e}")
    print(f"вывод в {d} (файлов в выводе ядра: {len(r.get('files') or [])})")
    if a.tail:
        print((r.get("log") or "")[-a.tail:])


def cmd_stub(a):
    cfg = load_cfg()
    if kernel_status("stub", cfg) in ("running", "queued"):
        sys.exit("заглушка уже идёт")
    t = push("stub", cfg)
    res = wait_stage(cfg, ["stub"], {"ready"}, t, timeout=20 * 60)
    e = res.get("stub")
    if not e or e["stage"] != "ready":
        sys.exit("заглушка не дошла до ready")
    url = e["url"]
    time.sleep(10)
    st = api_call(url, cfg, "GET", "/v1/status")
    print("через туннель:", st)
    bad = api_call(url, {"api_key": "wrong"}, "GET", "/v1/status", ok_codes=(401,))
    print("чужой ключ отвергнут:", bad)
    ntfy_post(cfg["topic"] + "-ctl", "stop")
    print("stop отправлен, ждём shutdown (до 2 мин)…")
    res = wait_stage(cfg, ["stub"], {"shutdown"}, t, timeout=300, quiet=True)
    print("канал исправен" if res.get("stub", {}).get("stage") == "shutdown" else "shutdown не пришёл")


def do_up(cfg, force=False, log=print, timeout=30 * 60):
    st = kernel_status("serve", cfg)
    if st in ("running", "queued") and not force:
        # пуш новой версии не отменяет идущую: квота ушла бы вдвое быстрее
        raise RuntimeError(f"qimg-serve уже {st}. Сначала qimg.py down (или --force, если уверены, что сессия мертва)")
    t = push("serve", cfg)
    log("ядро залито, ждём ready (холодный старт 5–7 мин, с новыми входами до 13)…")
    seen = set()
    t0 = time.time()
    while time.time() - t0 < timeout:
        for e in beacons(cfg, t - 5):
            if e["time"] < int(t) or e.get("k") != "serve" or e["id"] in seen:
                continue
            seen.add(e["id"])
            log(fmt_event(e))
            if e["stage"] == "ready":
                save_json(os.path.join(STATE, "session.json"), {"url": e["url"], "since": t})
                return e["url"]
            if e["stage"] == "shutdown":
                raise RuntimeError("ядро погасло на старте; смотрите qimg.py logs serve")
        time.sleep(5)
    raise RuntimeError("не дождались ready")


def cmd_up(a):
    cfg = load_cfg()
    try:
        url = do_up(cfg, a.force)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"готово: {url}")


def cmd_down(a):
    cfg = load_cfg()
    t = time.time()
    ntfy_post(cfg["topic"] + "-ctl", "stop")
    print("stop отправлен; ядро читает сигнал раз в минуту")
    if not a.no_wait:
        res = wait_stage(cfg, ["serve"], {"shutdown"}, t, timeout=240, quiet=True)
        print("погашено" if res.get("serve") else "shutdown пока не пришёл — проверьте qimg.py status")


def cmd_status(a):
    cfg = load_cfg()
    s = session_state(cfg)
    print(f"ядро: {s['stage']}" + (f" с {dt.datetime.fromtimestamp(s['since']):%H:%M}" if s["since"] else ""))
    if s["url"]:
        print(f"адрес: {s['url']}")
    if s["last_error"]:
        print(f"последняя ошибка: {s['last_error']}")
    try:
        used, total, refresh = quota()
        print(f"квота GPU (Kaggle): сожжено {used} ч из {total}, обнулится {refresh}")
    except Exception as e:  # noqa: BLE001
        print(f"квота по маяку: сожжено {s['quota_used_h']} ч, осталось ~{s['quota_left_h']} ч из 30 ({e})")
    if s["url"] and s["stage"] == "ready":
        try:
            print(json.dumps(api_call(s["url"], cfg, "GET", "/v1/status"), ensure_ascii=False, indent=1))
        except Exception as e:  # noqa: BLE001
            print(f"API не отвечает: {e}")
    if a.events:
        for e in beacons_local()[-a.events:]:
            print(fmt_event(e))


# ---------------- API ядра ----------------
def api_call(url, cfg, method, path, body=None, raw=None, ctype=None, ok_codes=(200,), timeout=120, full=False):
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "User-Agent": "qimg"}
    data = None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    elif raw is not None:
        data, headers["Content-Type"] = raw, ctype or "application/octet-stream"
    req = urllib.request.Request(url.rstrip("/") + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload, code, rtype = r.read(), r.status, r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        payload, code, rtype = e.read(), e.code, e.headers.get("Content-Type", "")
    if full:
        return code, rtype, payload
    if code not in ok_codes:
        raise RuntimeError(f"{method} {path}: HTTP {code} {payload[:300].decode('utf8', 'replace')}")
    return json.loads(payload) if rtype.startswith("application/json") else payload


def current_url(cfg):
    s = load_json(os.path.join(STATE, "session.json"), {})
    st = session_state(cfg)
    url = st["url"] or (s.get("url") if st["running"] else None)
    if not url:
        sys.exit("ядро не поднято: qimg.py up")
    return url


def saved_index():
    return load_json(os.path.join(STATE, "saved.json"), {})


_save_lock = threading.Lock()


def save_job(url, cfg, group, job):
    """Скачать готовый кадр к себе: out/ДАТА/время_пресет_сид_id.ext + JSON рядом."""
    with _save_lock:
        idx = saved_index()
        if job["id"] in idx:
            return idx[job["id"]]
    data = api_call(url, cfg, "GET", f"/v1/files/{job['file']}")
    meta = dict(job.get("meta") or {})
    ext = meta.get("format", "png")
    now = dt.datetime.now()
    d = os.path.join(OUT, now.strftime("%Y-%m-%d"))
    os.makedirs(d, exist_ok=True)
    name = f"{now:%H%M%S}_{group['preset']}_{job['seed']}_{job['id']}.{ext}"
    path = os.path.join(d, name)
    with open(path, "wb") as f:
        f.write(data)
    meta.update(job_id=job["id"], saved=now.isoformat(timespec="seconds"), group=group["id"])
    save_json(path.rsplit(".", 1)[0] + ".json", meta)
    rel = os.path.relpath(path, OUT)
    with _save_lock:
        idx = saved_index()
        idx[job["id"]] = rel
        save_json(os.path.join(STATE, "saved.json"), idx)
    return rel


def upload_any(url, cfg, ref):
    """Локальный файл -> /v1/upload (с промптом из JSON рядом, если есть). Иначе считаем ref id на ядре."""
    if os.path.exists(ref):
        side = ref.rsplit(".", 1)[0] + ".json"
        meta = load_json(side, {})
        b64 = base64.b64encode(open(ref, "rb").read()).decode()
        body = {"b64": b64, "name": os.path.basename(ref), **{k: meta[k] for k in ("prompt", "preset", "seed") if k in meta}}
        return api_call(url, cfg, "POST", "/v1/upload", body=body)["id"]
    return ref


def run_group(url, cfg, req):
    g = api_call(url, cfg, "POST", "/v1/jobs", body=req)
    print(f"группа {g['id']}: {g['preset']} ×{g['n']}, сид {g['seed']}" +
          (", переписываю промпт…" if g["rewrite"] == "pending" else ""))
    done = set()
    last = ""
    while True:
        g = api_call(url, cfg, "GET", f"/v1/jobs/{g['id']}")
        line = "  ".join(f"[{j['idx']}] {j['status']} {int(j['progress'] * 100)}%" for j in g["jobs"])
        if line != last:
            print("\r" + line + " " * 10, end="", flush=True)
            last = line
        for j in g["jobs"]:
            if j["status"] == "done" and j["id"] not in done:
                done.add(j["id"])
                print(f"\n  → {os.path.join('out', save_job(url, cfg, g, j))}")
            if j["status"] == "failed" and j["id"] not in done:
                done.add(j["id"])
                print(f"\n  ✗ {j['error']}")
        if g["status"] == "done":
            print()
            if g.get("rewrite") == "done":
                print("переписанный промпт:", g["prompt_used"][:300] + ("…" if len(g["prompt_used"]) > 300 else ""))
            return g
        time.sleep(2)


def cmd_gen(a):
    cfg = load_cfg()
    url = current_url(cfg)
    refs = [upload_any(url, cfg, r) for r in (a.image or [])]
    req = {"preset": "edit" if refs else a.preset, "prompt": a.prompt, "aspect": a.aspect, "n": a.n, "seed": a.seed,
           "neg": a.neg, "alpha": a.alpha, "rewrite": not a.no_rewrite, "refs": refs}
    run_group(url, cfg, req)


def cmd_frame(a):
    cfg = load_cfg()
    url = current_url(cfg)
    if a.cmd == "edit":
        refs = [upload_any(url, cfg, r) for r in a.src]
        req = {"preset": "edit", "refs": refs, "prompt": a.prompt, "alpha": a.alpha, "seed": a.seed}
    elif a.cmd == "rmbg":
        req = {"preset": "edit", "refs": [upload_any(url, cfg, a.src[0])], "prompt": "", "alpha": True, "seed": a.seed}
    else:
        preset = {"refine": "refine", "vary": "vary", "up4x": "up"}[a.cmd]
        req = {"preset": preset, "src": upload_any(url, cfg, a.src[0]), "prompt": a.prompt, "seed": a.seed, "n": a.n}
    run_group(url, cfg, req)


# ---------------- локальная страница ----------------
class UIState:
    def __init__(self):
        self.launch_log = []
        self.launching = False
        self.lock = threading.Lock()


def cmd_ui(a):
    cfg = load_cfg()
    ui = UIState()
    html_path = os.path.join(ROOT, "ui", "index.html")

    def remote():
        s = load_json(os.path.join(STATE, "session.json"), {})
        return s.get("url")

    def gallery(limit=400):
        items = []
        for d, _, fs in os.walk(OUT):
            for f in fs:
                if f.rsplit(".", 1)[-1] in ("png", "webp", "jpg", "jpeg"):
                    p = os.path.join(d, f)
                    meta = load_json(p.rsplit(".", 1)[0] + ".json", {})
                    items.append({"path": os.path.relpath(p, OUT), "mtime": os.path.getmtime(p),
                                  "prompt": meta.get("prompt", ""), "preset": meta.get("preset"), "seed": meta.get("seed"),
                                  "w": meta.get("w"), "h": meta.get("h")})
        items.sort(key=lambda x: -x["mtime"])
        return items[:limit]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, obj=None, raw=None, ctype="application/json"):
            data = raw if raw is not None else json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _proxy(self, method):
            url = remote()
            if not url:
                return self._send(503, {"error": "ядро не поднято"})
            path = self.path[len("/api"):]
            body = self._body() if method == "POST" else None
            try:
                code, rtype, payload = api_call(url, cfg, method, path, raw=body, ctype=self.headers.get("Content-Type"),
                                                timeout=300, full=True)
            except Exception as e:  # noqa: BLE001
                return self._send(502, {"error": str(e)})
            if code != 200 or not rtype.startswith("application/json"):
                return self._send(code, raw=payload, ctype=rtype or "application/octet-stream")
            res = json.loads(payload)
            # готовые кадры сразу сохраняем к себе
            if method == "GET" and path.startswith("/v1/jobs/") and isinstance(res, dict) and "jobs" in res:
                for j in res["jobs"]:
                    if j["status"] == "done":
                        try:
                            j["local"] = save_job(url, cfg, res, j)
                        except Exception as e:  # noqa: BLE001
                            j["local_error"] = str(e)
            return self._send(200, res)

        def do_GET(self):
            p = urllib.parse.urlparse(self.path).path
            if p in ("/", "/index.html"):
                return self._send(200, raw=open(html_path, "rb").read(), ctype="text/html; charset=utf-8")
            if p.startswith("/api/"):
                return self._proxy("GET")
            if p == "/local/state":
                s = session_state(cfg)
                if s["url"]:
                    save_json(os.path.join(STATE, "session.json"), {"url": s["url"], "since": s["since"]})
                return self._send(200, {**s, "launching": ui.launching, "launch_log": ui.launch_log[-30:],
                                        "presets": json.load(open(os.path.join(ROOT, "presets.json")))})
            if p == "/local/gallery":
                return self._send(200, gallery())
            if p.startswith("/local/img/"):
                rel = urllib.parse.unquote(p[len("/local/img/"):])
                full = os.path.realpath(os.path.join(OUT, rel))
                if not full.startswith(os.path.realpath(OUT) + os.sep) or not os.path.exists(full):
                    return self._send(404, {"error": "нет файла"})
                return self._send(200, raw=open(full, "rb").read(),
                                  ctype=mimetypes.guess_type(full)[0] or "application/octet-stream")
            self._send(404, {"error": "not found"})

        def do_POST(self):
            p = urllib.parse.urlparse(self.path).path
            if p.startswith("/api/"):
                return self._proxy("POST")
            if p == "/local/up":
                with ui.lock:
                    if ui.launching:
                        return self._send(200, {"ok": True, "already": True})
                    ui.launching, ui.launch_log = True, []

                def run():
                    try:
                        do_up(cfg, log=ui.launch_log.append)
                    except Exception as e:  # noqa: BLE001
                        ui.launch_log.append(f"ошибка: {e}")
                    finally:
                        ui.launching = False

                threading.Thread(target=run, daemon=True).start()
                return self._send(200, {"ok": True})
            if p == "/local/down":
                ntfy_post(cfg["topic"] + "-ctl", "stop")
                return self._send(200, {"ok": True})
            if p == "/local/reupload":
                url = remote()
                req = json.loads(self._body() or b"{}")
                full = os.path.realpath(os.path.join(OUT, req.get("path", "")))
                if not url or not full.startswith(os.path.realpath(OUT) + os.sep):
                    return self._send(400, {"error": "нет ядра или файла"})
                try:
                    return self._send(200, {"id": upload_any(url, cfg, full)})
                except Exception as e:  # noqa: BLE001
                    return self._send(502, {"error": str(e)})
            self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"страница: http://127.0.0.1:{a.port}")
    srv.serve_forever()


# ---------------- разбор аргументов ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init", help="ник Kaggle, токен, ключ API и топик")
    p.add_argument("--user", required=True)
    p.add_argument("--token", help="KGAT_... или key из kaggle.json")
    p.add_argument("--kaggle-json")
    p.set_defaults(fn=cmd_init)
    p = sp.add_parser("setup", help="залить сборку и качалки (CPU)")
    p.add_argument("--only", nargs="*", choices=["build", "fetch-serve", "fetch-pe"])
    p.set_defaults(fn=cmd_setup)
    p = sp.add_parser("kstatus", help="статусы ядер на Kaggle")
    p.add_argument("kinds", nargs="*")
    p.set_defaults(fn=cmd_kstatus)
    p = sp.add_parser("logs", help="лог и вывод завершённого ядра")
    p.add_argument("kind", choices=list(KERNELS))
    p.add_argument("--all", action="store_true", help="скачать все файлы вывода")
    p.add_argument("--tail", type=int, default=3000)
    p.set_defaults(fn=cmd_logs)
    p = sp.add_parser("render", help="только собрать исходник ядра в build/")
    p.add_argument("kind", choices=list(KERNELS))
    p.set_defaults(fn=lambda a: print(render(a.kind, load_cfg())))
    sp.add_parser("stub", help="проверить канал на CPU-заглушке").set_defaults(fn=cmd_stub)
    p = sp.add_parser("up", help="поднять боевое ядро")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_up)
    p = sp.add_parser("down", help="погасить")
    p.add_argument("--no-wait", action="store_true")
    p.set_defaults(fn=cmd_down)
    p = sp.add_parser("status", help="состояние и квота")
    p.add_argument("--events", type=int, default=0)
    p.set_defaults(fn=cmd_status)
    presets = json.load(open(os.path.join(ROOT, "presets.json")))
    p = sp.add_parser("gen", help="картинка по тексту (или правка, если есть -i)")
    p.add_argument("prompt")
    p.add_argument("-p", "--preset", default="draft", choices=["draft", "std", "hq"])
    p.add_argument("-a", "--aspect", default="1:1", choices=list(presets["sizes"]))
    p.add_argument("-n", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--neg", default="")
    p.add_argument("--alpha", action="store_true", help="прозрачный фон")
    p.add_argument("--no-rewrite", action="store_true", help="не разворачивать промпт")
    p.add_argument("-i", "--image", action="append", help="картинка на вход (до 3) — тогда это правка")
    p.set_defaults(fn=cmd_gen)
    for name, hlp in (("refine", "дорисовать кадр"), ("vary", "похожие"), ("up4x", "×4"), ("rmbg", "убрать фон")):
        p = sp.add_parser(name, help=hlp)
        p.add_argument("src", nargs=1, help="файл из out/ или id на ядре")
        p.add_argument("--prompt", default="")
        p.add_argument("--seed", type=int)
        p.add_argument("-n", type=int)
        p.set_defaults(fn=cmd_frame)
    p = sp.add_parser("edit", help="правка словами: qimg.py edit a.png [b.png c.png] \"инструкция\"")
    p.add_argument("args", nargs="+")
    p.add_argument("--alpha", action="store_true")
    p.add_argument("--seed", type=int)
    p.set_defaults(fn=lambda a: (setattr(a, "src", a.args[:-1]), setattr(a, "prompt", a.args[-1]), cmd_frame(a)))
    p = sp.add_parser("ui", help="локальная страница")
    p.add_argument("--port", type=int, default=7861)
    p.set_defaults(fn=cmd_ui)
    a = ap.parse_args()
    if a.cmd == "edit" and len(a.args) < 2:
        sys.exit("нужно: edit <картинка...> \"инструкция\"")
    a.fn(a)


if __name__ == "__main__":
    main()
