# qimg-stub: CPU-заглушка с тем же транспортом (туннель + маяк + stop). Ноль GPU-квоты.
# Отвечает на /v1/health и /v1/status с тем же Bearer-ключом; гаснет по stop или через 10 мин простоя.
import hmac
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__TRANSPORT_LIB__

API_KEY = "__API_KEY__"
LAST = [time.time()]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/health"):
            return self._send(200, {"ok": True})
        if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {API_KEY}".encode()):
            return self._send(401, {"error": "unauthorized"})
        LAST[0] = time.time()
        self._send(200, {"ok": True, "stub": True, "uptime": int(time.time() - T_START)})


def main():
    beacon("start")
    srv = ThreadingHTTPServer(("127.0.0.1", 8080), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tunnel = Tunnel(8080)
    beacon("ready", url=tunnel.start())
    watchdog(tunnel, lambda: LAST[0], idle_s=600, max_s=3600, period=30)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        beacon("error", what="fatal", msg=str(e)[:300])
        sys.exit(1)
