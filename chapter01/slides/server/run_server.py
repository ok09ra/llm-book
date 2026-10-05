"""Slidev の {monaco-run} から Python を実行するためのローカルサーバ。

Jupyter カーネルを1つ常駐させ、POST /run で受け取ったコードを実行して出力を返す。
カーネルは使い回すので、ノートブックと同じく変数がスライド間で引き継がれる。

    uv run python run_server.py               # 起動時に warmup.py を実行
    uv run python run_server.py --no-warmup
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jupyter_client.manager import start_new_kernel

km, kc = None, None
lock = threading.Lock()


def run(code: str, timeout: float = 300) -> dict:
    outputs, error = [], None

    def hook(msg):
        nonlocal error
        kind, content = msg["msg_type"], msg["content"]
        if kind == "stream":
            outputs.append(content["text"])
        elif kind in ("execute_result", "display_data"):
            outputs.append(content["data"].get("text/plain", "") + "\n")
        elif kind == "error":
            # スライドに収まるよう、トレースバックは省いて例外の種類とメッセージだけ返す
            error = f'{content["ename"]}: {content["evalue"]}'

    with lock:
        kc.execute_interactive(code, output_hook=hook, timeout=timeout)
    return {"output": "".join(outputs), "error": error}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict | None = None):
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        if body is not None:
            self.wfile.write(json.dumps(body, ensure_ascii=False).encode())

    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        self._send(200, {"status": "ok"})

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        code = json.loads(self.rfile.read(length))["code"]
        self._send(200, run(code))

    def log_message(self, fmt, *args):
        pass


def main():
    global km, kc
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    km, kc = start_new_kernel(kernel_name="python3")
    # 発表中に警告やダウンロードの進捗バーが出ないようにする
    run(
        "import os, warnings; warnings.filterwarnings('ignore');"
        "os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1';"
        "from transformers.utils import logging; logging.set_verbosity_error()"
    )
    if not args.no_warmup:
        print("warming up (loading models)...", flush=True)
        result = run((Path(__file__).parent / "warmup.py").read_text(), timeout=1800)
        print(result["error"] or result["output"].strip(), flush=True)
    else:
        run("from transformers import pipeline")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ready: http://127.0.0.1:{args.port}/run", flush=True)
    try:
        server.serve_forever()
    finally:
        km.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
