# ---- transport: маяк ntfy, сигнал stop, туннель cloudflared, сторож ----
# Вставляется клиентом в исходник ядра на место метки транспорта. Только stdlib.
import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request

NTFY = "https://ntfy.sh"
TOPIC = "__TOPIC__"
KIND = "__KIND__"
T_START = time.time()


def beacon(stage, **kw):
    """Отметка стадии в ntfy: 'stage&k=<kind>&key=val'. Ошибки сети не роняют ядро."""
    kw = {"k": KIND, **{k: v for k, v in kw.items() if v is not None}}
    msg = stage + "&" + urllib.parse.urlencode(kw)
    print(f"[beacon] {msg}", flush=True)
    for _ in range(3):
        try:
            req = urllib.request.Request(f"{NTFY}/{TOPIC}", data=msg.encode(), method="POST")
            urllib.request.urlopen(req, timeout=15).read()
            return
        except Exception as e:  # noqa: BLE001
            print(f"[beacon] fail: {e}", flush=True)
            time.sleep(2)


def stop_requested():
    """Есть ли 'stop' в <топик>-ctl новее старта этого ядра."""
    try:
        url = f"{NTFY}/{TOPIC}-ctl/json?poll=1&since={int(T_START)}"
        body = urllib.request.urlopen(url, timeout=20).read().decode()
    except Exception as e:  # noqa: BLE001
        print(f"[ctl] poll fail: {e}", flush=True)
        return False
    for line in body.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("event") == "message" and m.get("time", 0) >= T_START and m.get("message", "").strip() == "stop":
            return True
    return False


CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


class Tunnel:
    def __init__(self, port, health_path="/v1/health"):
        self.port = port
        self.health_path = health_path
        self.bin = "/tmp/cloudflared"
        self.proc = None
        self.url = None
        self.fails = 0
        self.started_at = 0

    def _ensure_bin(self):
        if not os.path.exists(self.bin):
            urllib.request.urlretrieve(CF_URL, self.bin)
            os.chmod(self.bin, 0o755)

    def start(self, timeout=120):
        self._ensure_bin()
        self.stop()
        log_path = f"/tmp/cloudflared-{int(time.time())}.log"
        log = open(log_path, "wb")
        self.proc = subprocess.Popen(
            [self.bin, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
            stdout=log, stderr=subprocess.STDOUT)
        self.url = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(1)
            if self.proc.poll() is not None:
                break
            m = re.search(rb"https://[a-z0-9-]+\.trycloudflare\.com", open(log_path, "rb").read())
            if m:
                self.url = m.group(0).decode()
                self.fails = 0
                self.started_at = time.time()
                return self.url
        raise RuntimeError("cloudflared не выдал адрес: " + open(log_path, "rb").read()[-2000:].decode("utf8", "replace"))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def healthy(self):
        """Процесс жив и адрес отвечает снаружи. Два провала подряд = мёртв."""
        if not self.proc or self.proc.poll() is not None:
            return False
        if time.time() - self.started_at < 90:  # DNS свежего адреса ещё расходится
            return True
        try:
            urllib.request.urlopen(self.url + self.health_path, timeout=20).read()
            self.fails = 0
        except Exception as e:  # noqa: BLE001
            self.fails += 1
            print(f"[tunnel] health fail {self.fails}: {e}", flush=True)
        return self.fails < 2


def watchdog(tunnel, is_idle_since, idle_s=900, max_s=11.5 * 3600, period=60, on_exit=None):
    """Блокирует до выхода. is_idle_since() -> время начала простоя или None, если идёт работа."""
    reason = None
    while reason is None:
        time.sleep(period)
        if stop_requested():
            reason = "stop"
        elif time.time() - T_START > max_s:
            reason = "max_session"
        else:
            since = is_idle_since()
            if since is not None and time.time() - since > idle_s:
                reason = "idle"
        if reason is None and tunnel is not None and not tunnel.healthy():
            beacon("tunnel_restart")
            try:
                beacon("ready", url=tunnel.start())
            except Exception as e:  # noqa: BLE001
                beacon("error", what="tunnel", msg=str(e)[:300])
    beacon("shutdown", reason=reason, up=int(time.time() - T_START))
    if on_exit:
        try:
            on_exit()
        except Exception as e:  # noqa: BLE001
            print(f"[watchdog] on_exit: {e}", flush=True)
    if tunnel is not None:
        tunnel.stop()
    return reason
# ---- /transport ----
