#!/usr/bin/env python3
"""Локальная проверка без Kaggle: поддельные sd-server ×2 и llama-server, настоящие imgd, клиент и страница.
Запуск: python3 tests/test_local.py"""
import base64
import io
import json
import os
import py_compile
import shutil
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "lib")]

from PIL import Image  # noqa: E402

import imgd as imgd_mod  # noqa: E402
import qimg  # noqa: E402
import rewriter as rw_mod  # noqa: E402
import sdserver  # noqa: E402

TMP = tempfile.mkdtemp(prefix="qimg-test-")
BODIES = os.path.join(TMP, "bodies.jsonl")
os.environ["FAKE_SD_BODIES"] = BODIES
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def bodies():
    with open(BODIES) as f:
        return [json.loads(x) for x in f]


FAKE_LLAMA = r'''#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
assert "--jinja" in sys.argv and "-ngl" in sys.argv
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _s(self, o):
        d = json.dumps(o).encode(); self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_GET(self): self._s({"status": "ok"})
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = b["messages"][1]["content"]
        assert b["messages"][0]["content"].startswith("SYSTEM")
        txt = "<think>hm</think>```json\n" + json.dumps({"rewritten_prompt": "REWRITTEN: " + user, "wh_ratio": 1}) + "\n```"
        self._s({"choices": [{"message": {"content": txt}}]})
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def setup_fakes():
    root = os.path.join(TMP, "sdcpp")
    os.makedirs(f"{root}/bin")
    os.makedirs(f"{root}/lib")
    with open(f"{root}/bin/sd-server", "w") as f:
        f.write(f"#!/bin/sh\nexec {sys.executable} {HERE}/fake_sdserver.py \"$@\"\n")
    os.chmod(f"{root}/bin/sd-server", 0o755)
    os.makedirs(f"{TMP}/llama")
    with open(f"{TMP}/llama/llama-server", "w") as f:
        f.write(f"#!{sys.executable}\n" + FAKE_LLAMA)
    os.chmod(f"{TMP}/llama/llama-server", 0o755)
    with open(f"{TMP}/system_prompt.txt", "w") as f:
        f.write("SYSTEM prompt")
    return root


def wait_group(base, cfg, gid, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        g = qimg.api_call(base, cfg, "GET", f"/v1/jobs/{gid}")
        if g["status"] == "done":
            return g
        time.sleep(0.3)
    raise TimeoutError(gid)


def png_b64(size, mode="RGB"):
    im = Image.new(mode, size, (10, 120, 200, 90) if mode == "RGBA" else (10, 120, 200))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    print(f"tmp: {TMP}")
    presets = json.load(open(os.path.join(ROOT, "presets.json")))

    print("1. шаблоны ядер собираются и компилируются")
    cfg = {"kaggle_user": "tester", "api_key": "k-" + "x" * 20, "topic": "qimg-test"}
    qimg.BUILD = os.path.join(TMP, "build")
    for kind in qimg.KERNELS:
        d = qimg.render(kind, cfg)
        src = open(os.path.join(d, "kernel.py")).read()
        py_compile.compile(os.path.join(d, "kernel.py"), doraise=True)
        meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
        check(True, f"{kind}: py_compile, метки подставлены")
        check(cfg["topic"] in src and f'KIND = "{kind}"' in src, f"{kind}: топик и вид подставлены")
        if kind == "serve":
            check(meta["enable_gpu"] and meta["machine_shape"] == "NvidiaTeslaT4", "serve: GPU T4")
            check(meta["kernel_sources"] == ["tester/qimg-build", "tester/qimg-fetch-serve", "tester/qimg-fetch-pe"],
                  "serve: kernel_sources")
            check(cfg["api_key"] in src, "serve: ключ подставлен")
        if kind == "fetch-pe":
            check("pe_t2i_heretic-Q4_K_M.gguf" in src and "system_prompt.txt" in src, "fetch-pe: файлы")

    print("2. поднимаем поддельные sd-server по одному на карту")
    root = setup_fakes()
    models = {"dit": "/x/dit", "te": "/x/te", "mmproj": "/x/mm", "vae": "/x/vae"}
    sds = [sdserver.SDServer(i, 18101 + i, models, sd_root=root) for i in range(2)]
    for sd in sds:
        sd.log_path = os.path.join(TMP, f"sd{sd.idx}.log")
        sd.start()
        sd.wait_ready(30)
    check(all(sd.alive() for sd in sds), "оба sd-server живы")

    rw = rw_mod.Rewriter(os.path.join(TMP, "pe.gguf"), os.path.join(TMP, "system_prompt.txt"), gpu=1, port=18090)
    rw.prepare = lambda: setattr(rw, "bin", os.path.join(TMP, "llama", "llama-server"))
    beacons = []
    d = imgd_mod.Imgd(presets, sds, cfg["api_key"], out_dir=os.path.join(TMP, "out_remote"), rewriter=rw,
                      beacon=lambda *a, **k: beacons.append((a, k)))
    d.serve("127.0.0.1", 18080)
    d.start_workers()
    base = "http://127.0.0.1:18080"
    time.sleep(0.5)

    print("3. API: ключ, здоровье, статус")
    code, _, _ = qimg.api_call(base, {"api_key": "wrong"}, "GET", "/v1/status", full=True)
    check(code == 401, "чужой ключ -> 401")
    check(urllib.request.urlopen(base + "/v1/health").status == 200, "/v1/health без ключа")
    st = qimg.api_call(base, cfg, "GET", "/v1/status")
    check(st["ok"] and len(st["workers"]) == 2, "статус")

    print("4. черновик ×2 с переписчиком (GPU1 делится по времени)")
    qimg.OUT = os.path.join(TMP, "out")
    qimg.STATE = os.path.join(TMP, "state")
    g = qimg.run_group(base, cfg, {"preset": "draft", "prompt": "кот в космосе", "aspect": "16:9"})
    check(all(j["status"] == "done" for j in g["jobs"]), "оба кадра готовы")
    check(g["prompt_used"].startswith("REWRITTEN: кот в космосе\n\nAspect ratio: 16:9"), "промпт переписан, формат передан")
    seeds = sorted(j["seed"] for j in g["jobs"])
    check(seeds[1] == seeds[0] + 1, "два сида подряд")
    b = [x for x in bodies() if x["prompt"].startswith("REWRITTEN")]
    check(len(b) == 2 and all(x["width"] == 1376 and x["height"] == 768 for x in b), "размер 16:9 = 1376×768")
    check(all(x["lora"] == [{"path": "p_qwen_image_2.1_8step_v0.1.safetensors", "multiplier": 1.0}] for x in b), "LoRA Pruna-8")
    check(all(len(x["sample_params"]["custom_sigmas"]) == 9 and x["sample_params"]["sample_steps"] == 8 for x in b),
          "8 шагов, 9 сигм")
    check(all(x["sample_params"]["guidance"]["txt_cfg"] == 1.0 for x in b), "CFG 1")
    check(all(x["output_format"] == "webp" and "cache_mode" not in x for x in b), "webp, без кэша")
    check(all("vae_tiling_params" in x for x in b), "16:9 = 1.06 Мп > 1 Мп: плитки VAE")
    saved = qimg.saved_index()
    check(all(j["id"] in saved and os.path.exists(os.path.join(qimg.OUT, saved[j["id"]])) for j in g["jobs"]),
          "кадры сохранены у клиента")
    side = os.path.join(qimg.OUT, saved[g["jobs"][0]["id"]]).rsplit(".", 1)[0] + ".json"
    meta = json.load(open(side))
    check(meta["prompt"] == "кот в космосе" and meta["preset"] == "draft" and "seed" in meta and meta["w"] == 1376,
          "JSON рядом: промпт, пресет, сид, размер")
    check(sds[1].alive() and d.rewriter_state == "ready", "sd-server cuda1 поднят обратно после переписчика")

    print("5. стандарт с прозрачностью: без переписчика, обёртка промпта, png")
    g = qimg.run_group(base, cfg, {"preset": "std", "prompt": "a glass bottle", "alpha": True})
    x = bodies()[-1]
    check(g["rewrite"] == "skip", "прозрачность не переписывается")
    check(x["prompt"] == presets["alpha_prefix"] + "a glass bottle" + presets["alpha_suffix"], "обёртка RGBA")
    check(x["output_format"] == "png" and x["cache_mode"] == "spectrum" and x["sample_params"]["sample_steps"] == 28,
          "png, spectrum, 28 шагов")

    print("6. максимум: hires ×1.25 с потолком площади и плитки")
    g = qimg.run_group(base, cfg, {"preset": "hq", "prompt": "city", "aspect": "16:9", "rewrite": False})
    x = bodies()[-1]
    check(x["hires"]["enabled"] and x["hires"]["steps"] == 16 and x["hires"]["denoising_strength"] == 0.35, "hires 16/0.35")
    check(1376 * 768 * x["hires"]["scale"] ** 2 <= presets["max_area"] + 1, f"площадь ≤ 1280² (scale={x['hires']['scale']})")
    check(x["vae_tiling_params"]["tile_size_w"] == 256 and "cache_mode" not in x and x["output_format"] == "png",
          "плитки 256, без кэша, png")
    g1 = qimg.run_group(base, cfg, {"preset": "hq", "prompt": "cat", "rewrite": False})
    check(bodies()[-1]["hires"]["scale"] == 1.25, "1:1 -> scale 1.25 (1280²)")
    hq_file = g1["jobs"][0]["file"]

    print("7. дорисовка, похожие, правка, ×4")
    g = qimg.run_group(base, cfg, {"preset": "refine", "src": g["jobs"][0]["file"]})
    x = bodies()[-1]
    check(x["width"] * x["height"] <= presets["max_area"] and x["width"] % 32 == 0 and x["height"] % 32 == 0,
          f"refine: {x['width']}×{x['height']} кратно 32, ≤ 1280²")
    check(x["strength"] == 0.45 and x["init_image"].startswith("<b64") and "vae_tiling_params" in x, "refine: сила 0.45, плитки")
    check(x["prompt"] == "city", "refine: промпт берётся из кадра")
    src = qimg.api_call(base, cfg, "POST", "/v1/upload", body={"b64": png_b64((1000, 700)), "prompt": "dog"})
    g = qimg.run_group(base, cfg, {"preset": "vary", "src": src["id"]})
    x = bodies()[-1]
    check(len(g["jobs"]) == 2 and x["strength"] == 0.65 and x["lora"] and x["prompt"] == "dog", "vary: ×2, Pruna-8, 0.65")
    check((x["width"], x["height"]) == (992, 672), f"vary: размер кратно 32 вниз ({x['width']}×{x['height']})")
    big = qimg.api_call(base, cfg, "POST", "/v1/upload", body={"b64": png_b64((3000, 1500))})
    check(max(big["w"], big["h"]) == 2048, "загрузка ужимается до 2048")
    g = qimg.run_group(base, cfg, {"preset": "edit", "refs": [big["id"], src["id"]], "prompt": "put the dog in the city"})
    x = bodies()[-1]
    check(len(x["ref_images"]) == 2 and x["sample_params"]["sample_steps"] == 20 and "vae_tiling_params" not in x,
          "edit: 2 refs, 20 шагов, без плиток")
    check(x["width"] * x["height"] <= 1024 * 1024 and x["width"] % 32 == 0, f"edit: ≤ 1024² ({x['width']}×{x['height']})")
    g = qimg.run_group(base, cfg, {"preset": "edit", "refs": [hq_file], "prompt": "", "alpha": True})
    x = bodies()[-1]
    check(x["prompt"].startswith(presets["alpha_prefix"] + presets["remove_bg_prompt"]), "убрать фон = правка с альфой")
    alpha_file = g["jobs"][0]["file"]
    g = qimg.run_group(base, cfg, {"preset": "up", "src": alpha_file})
    data = qimg.api_call(base, cfg, "GET", f"/v1/files/{g['jobs'][0]['file']}")
    im = Image.open(io.BytesIO(data))
    check(max(im.size) == 2048 and im.mode == "RGBA", f"×4: ужато до 2048, альфа сохранена ({im.size}, {im.mode})")

    print("8. ошибки: задача падает, воркер живёт; sd-server умер — поднимается")
    g = qimg.run_group(base, cfg, {"preset": "std", "prompt": "FAIL please", "rewrite": False})
    check(g["jobs"][0]["status"] == "failed", "failed")
    g = qimg.run_group(base, cfg, {"preset": "std", "prompt": "CRASH now", "rewrite": False})
    check(g["jobs"][0]["status"] == "failed", "sd-server упал -> failed")
    g = qimg.run_group(base, cfg, {"preset": "draft", "prompt": "after crash", "rewrite": False})
    check(all(j["status"] == "done" for j in g["jobs"]), "после падения обе карты снова рисуют")
    check(any(a[0] == "sd_restart" for a, _ in beacons) or any(a[0] == "error" for a, _ in beacons), "маяк о падении")
    code, _, body = qimg.api_call(base, cfg, "POST", "/v1/jobs", body={"preset": "draft"}, full=True)
    check(code == 400, "пустой промпт -> 400")

    print("9. приоритеты и отмена")
    os.environ["FAKE_SD_STEP_S"] = "0.05"
    for sd in sds:  # медленнее, чтобы успеть посмотреть очередь
        sd.restart()
    a1 = qimg.api_call(base, cfg, "POST", "/v1/jobs", body={"preset": "std", "prompt": "a", "n": 2, "rewrite": False})
    a2 = qimg.api_call(base, cfg, "POST", "/v1/jobs", body={"preset": "draft", "prompt": "b", "n": 2, "rewrite": False})
    a3 = qimg.api_call(base, cfg, "POST", "/v1/jobs", body={"preset": "up", "src": src["id"]})
    qimg.api_call(base, cfg, "POST", f"/v1/cancel/{a2['id']}")
    g3 = wait_group(base, cfg, a3["id"])
    g2 = wait_group(base, cfg, a2["id"])
    g1 = wait_group(base, cfg, a1["id"])
    check(all(j["status"] == "cancelled" for j in g2["jobs"]), "отменённые черновики не рисовались")
    check(g3["jobs"][0]["started"] <= max(j["finished"] for j in g1["jobs"]), "×4 обогнал очередь")
    os.environ["FAKE_SD_STEP_S"] = "0.3"
    for sd in sds:
        sd.restart()
    a4 = qimg.api_call(base, cfg, "POST", "/v1/jobs", body={"preset": "std", "prompt": "long", "rewrite": False})
    time.sleep(2)
    run = qimg.api_call(base, cfg, "GET", f"/v1/jobs/{a4['id']}")
    check(0 < run["jobs"][0]["progress"] < 1, f"прогресс из лога: {run['jobs'][0]['progress']}")
    qimg.api_call(base, cfg, "POST", f"/v1/cancel/{a4['id']}")
    g4 = wait_group(base, cfg, a4["id"])
    check(g4["jobs"][0]["status"] == "cancelled", "отмена идущего кадра")
    check(d.idle_since() is not None, "простой считается, когда всё закончено")

    print("10. локальная страница проксирует и сохраняет")
    os.makedirs(qimg.STATE, exist_ok=True)
    qimg.save_json(os.path.join(qimg.STATE, "session.json"), {"url": base, "since": time.time()})
    qimg.session_state = lambda cfg_, refresh=True: {"running": True, "stage": "ready", "url": base, "since": time.time(),
                                                      "last_error": None, "quota_used_h": 1.5, "quota_left_h": 28.5}
    qimg.load_cfg = lambda: cfg

    class A:
        port = 17861

    threading.Thread(target=qimg.cmd_ui, args=(A,), daemon=True).start()
    time.sleep(0.5)
    ui = "http://127.0.0.1:17861"
    html = urllib.request.urlopen(ui + "/").read().decode()
    check("<title>qimg</title>" in html, "страница отдаётся")
    stt = json.loads(urllib.request.urlopen(ui + "/local/state").read())
    check(stt["stage"] == "ready" and "sizes" in stt["presets"], "/local/state")
    req = urllib.request.Request(ui + "/api/v1/jobs", data=json.dumps({"preset": "draft", "prompt": "ui test", "rewrite": False}).encode(),
                                 headers={"Content-Type": "application/json"})
    os.environ["FAKE_SD_STEP_S"] = "0.02"
    for sd in sds:
        sd.restart()
    gid = json.loads(urllib.request.urlopen(req).read())["id"]
    t = time.time()
    while time.time() - t < 30:
        gv = json.loads(urllib.request.urlopen(ui + f"/api/v1/jobs/{gid}").read())
        if gv["status"] == "done":
            break
        time.sleep(0.3)
    check(all(j.get("local") for j in gv["jobs"]), "страница сама сохраняет готовые кадры")
    img = urllib.request.urlopen(ui + "/local/img/" + gv["jobs"][0]["local"])
    check(img.headers["Content-Type"] == "image/webp", "кадр с диска отдаётся")
    gal = json.loads(urllib.request.urlopen(ui + "/local/gallery").read())
    check(len(gal) >= 2 and gal[0]["prompt"], "галерея на диске")
    r = urllib.request.Request(ui + "/local/reupload", data=json.dumps({"path": gal[0]["path"]}).encode(),
                               headers={"Content-Type": "application/json"})
    rid = json.loads(urllib.request.urlopen(r).read())["id"]
    check(d.files[rid].get("prompt") == gal[0]["prompt"], "кадр прошлой сессии дозагружается с промптом")
    raw = urllib.request.urlopen(ui + f"/api/v1/files/{gv['jobs'][0]['file']}")
    check(raw.headers["Content-Type"] == "image/webp", "прокси файлов с типом")
    try:
        urllib.request.urlopen(ui + "/local/img/../../etc/passwd")
        check(False, "выход из out/ запрещён")
    except urllib.error.HTTPError as e:
        check(e.code == 404, "выход из out/ запрещён")

    print("11. квота по маяку")
    ev = [{"id": "1", "time": 1000, "stage": "start", "k": "serve"},
          {"id": "2", "time": 1300, "stage": "ready", "k": "serve", "url": "https://a.trycloudflare.com"},
          {"id": "3", "time": 4600, "stage": "shutdown", "k": "serve"},
          {"id": "4", "time": 5000, "stage": "start", "k": "build"},
          {"id": "5", "time": 9000, "stage": "start", "k": "serve"}]
    ss = qimg.sessions(ev)
    check(len(ss) == 2 and ss[0]["end"] == 4600 and ss[1]["end"] is None and ss[0]["url"].startswith("https://"),
          "сессии serve, сборка не считается")
    check(rw_mod.extract_rewritten('blah {"rewritten_prompt": "x \\"y\\"", "wh_ratio": 1.5} tail') == 'x "y"',
          "разбор ответа переписчика")

    for sd in sds:
        sd.stop()
    print()
    print("ВСЕ ПРОВЕРКИ ПРОШЛИ" if not FAILS else f"ПРОВАЛЕНО: {len(FAILS)}\n - " + "\n - ".join(FAILS))
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
