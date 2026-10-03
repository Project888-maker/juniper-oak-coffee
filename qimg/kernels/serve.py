# qimg-serve: боевое GPU-ядро (2× T4). Собирается клиентом из шаблона, метки __X__ подставляются на пуше.
import glob
import io
import os
import shutil
import statistics
import sys
import threading
import time
import traceback

__TRANSPORT_LIB__

__SDSERVER_LIB__

__REWRITER_LIB__

__IMGD_LIB__

API_KEY = "__API_KEY__"
PRESETS = json.loads(r'''__PRESETS_JSON__''')
IDLE_S = 15 * 60
MAX_S = 11.5 * 3600


def find_one(name):
    """Входы монтируются вглубь /kaggle/input/, путь заранее неизвестен: ищем по точному имени."""
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def prepare_sdcpp():
    srv = find_one("sdcpp/bin/sd-server")
    root = os.path.dirname(os.path.dirname(srv))
    shutil.rmtree("/tmp/sdcpp", ignore_errors=True)
    shutil.copytree(root, "/tmp/sdcpp", symlinks=False)  # вход только для чтения, без бита исполнения
    for f in glob.glob("/tmp/sdcpp/bin/*"):
        os.chmod(f, 0o755)


def link(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not os.path.exists(dst):
        os.symlink(src, dst)


def warmup(sd):
    body = {"prompt": "a red apple on a wooden table, photo", "negative_prompt": "", "seed": 1, "batch_count": 1,
            "width": 512, "height": 512,
            "sample_params": {"sample_method": "euler", "sample_steps": 8, "custom_sigmas": PRESETS["sigmas_8step"],
                              "guidance": {"txt_cfg": 1.0}},
            "lora": [{"path": PRESETS["lora_8step"], "multiplier": 1.0}], "output_format": "png"}
    data = sd.generate(body, timeout=900)
    im = Image.open(io.BytesIO(data)).convert("L")
    spread = statistics.pstdev(im.getdata())
    return spread


def main():
    beacon("start")
    prepare_sdcpp()
    models = {
        # точные имена: маска *heretic-Q4_K_M.gguf поймала бы и файл переписчика
        "dit": find_one("qwen_image_2.1_int8_convrot.safetensors"),
        "vae": find_one("qwen_image_2.1_vae_bf16.safetensors"),
        "te": find_one("qwen3vl_8b_heretic-Q4_K_M.gguf"),
        "mmproj": find_one("mmproj-qwen3vl_8b_heretic-f16.gguf"),
    }
    link(find_one(PRESETS["lora_8step"]), "/tmp/loras")
    link(find_one("RealESRGAN_x4plus.pth"), "/tmp/ups")
    print("models:", models, flush=True)

    sds = [SDServer(0, 8101, models), SDServer(1, 8102, models)]
    for sd in sds:  # по очереди, иначе тесно по ОЗУ
        t = time.time()
        sd.start()
        sd.wait_ready()
        print(f"sd-server cuda{sd.idx} готов за {time.time() - t:.0f} с", flush=True)
    beacon("model_loaded", t=int(time.time() - T_START))

    spreads = [None, None]

    def _w(i):
        try:
            spreads[i] = warmup(sds[i])
        except Exception as e:  # noqa: BLE001
            spreads[i] = -1
            print(f"warmup cuda{i}: {e}", flush=True)

    ths = [threading.Thread(target=_w, args=(i,)) for i in range(2)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    for i, s in enumerate(spreads):
        if s is None or s < 3:
            beacon("error", what="warmup_black" if s is not None and s >= 0 else "warmup_failed", gpu=i,
                   spread=None if s is None else round(s, 2))
    beacon("warm", t=int(time.time() - T_START), spread=",".join(f"{s:.1f}" for s in spreads if s is not None))

    rewriter = None
    try:
        rewriter = Rewriter(find_one("pe_t2i_heretic-Q4_K_M.gguf"), find_one("system_prompt.txt"), gpu=1)
    except Exception as e:  # noqa: BLE001
        beacon("error", what="rewriter_missing", msg=str(e)[:200])

    imgd = Imgd(PRESETS, sds, API_KEY, rewriter=rewriter, beacon=beacon, session_start=T_START, max_session=MAX_S)
    imgd.serve("127.0.0.1", 8080)
    imgd.start_workers()

    tunnel = Tunnel(8080)
    beacon("ready", url=tunnel.start(), t=int(time.time() - T_START))

    def cleanup():
        for sd in sds:
            sd.stop()
        if rewriter:
            rewriter.unload()
        os.makedirs("/kaggle/working/logs", exist_ok=True)  # для kaggle kernels output
        for f in glob.glob("/tmp/sd*.log") + glob.glob("/tmp/llama.log") + glob.glob("/tmp/cloudflared-*.log"):
            shutil.copy(f, "/kaggle/working/logs/")

    watchdog(tunnel, imgd.idle_since, idle_s=IDLE_S, max_s=MAX_S, on_exit=cleanup)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        beacon("error", what="fatal", msg=f"{type(e).__name__}: {e}"[:400])
        beacon("shutdown", reason="fatal", up=int(time.time() - T_START))
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
