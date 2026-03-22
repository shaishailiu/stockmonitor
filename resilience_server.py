#!/usr/bin/env python3
"""
韧性筛选报告 — 本地服务器
提供:
  1. resilience_report/ 目录的静态文件服务
  2. POST /api/stock_type  更新 config.json 中股票的 stock_type 字段
  3. GET  /api/stock_types  获取所有股票的 stock_type 映射

用法:
  python3 resilience_server.py          # 默认端口 8089
  python3 resilience_server.py --port 9000
"""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

PORT = 8089
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")
REPORT_DIR = os.path.join(PROJECT_DIR, "resilience_report")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")


class ReportHandler(SimpleHTTPRequestHandler):
    """自定义 handler：静态文件 + API"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=REPORT_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/stock_types":
            self._handle_get_stock_types()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/stock_type":
            self._handle_update_stock_type()
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        """处理 CORS 预检"""
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # ── API: 获取所有 stock_type ──
    def _handle_get_stock_types(self):
        try:
            cfg = load_config()
            mapping = {}
            duplicates = {}
            for s in cfg.get("stocks", []):
                mapping[s["symbol"]] = s.get("stock_type", "unclassified")
                if s.get("duplicate_of"):
                    duplicates[s["symbol"]] = s["duplicate_of"]
            self._json_response({"ok": True, "stock_types": mapping, "duplicates": duplicates})
        except Exception as e:
            self._json_response({"ok": False, "error": str(e)}, status=500)

    # ── API: 更新单只股票的 stock_type ──
    def _handle_update_stock_type(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            symbol = data.get("symbol")
            stock_type = data.get("stock_type")

            if not symbol or stock_type not in ("cyclical", "non_cyclical", "unclassified"):
                self._json_response(
                    {"ok": False, "error": f"无效参数: symbol={symbol}, stock_type={stock_type}"},
                    status=400,
                )
                return

            cfg = load_config()
            found = False
            for s in cfg.get("stocks", []):
                if s["symbol"] == symbol:
                    s["stock_type"] = stock_type
                    found = True
                    break

            if not found:
                self._json_response({"ok": False, "error": f"未找到 symbol={symbol}"}, status=404)
                return

            save_config(cfg)
            self._json_response({"ok": True, "symbol": symbol, "stock_type": stock_type})
            print(f"  ✅ 已更新 {symbol} → {stock_type}")

        except json.JSONDecodeError:
            self._json_response({"ok": False, "error": "JSON 解析失败"}, status=400)
        except Exception as e:
            self._json_response({"ok": False, "error": str(e)}, status=500)

    # ── 工具方法 ──
    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        # 只打印 API 请求，静态文件请求不打印
        if "/api/" in (args[0] if args else ""):
            super().log_message(format, *args)


def main():
    port = PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    server = HTTPServer(("0.0.0.0", port), ReportHandler)
    print(f"🚀 韧性筛选报告服务器启动")
    print(f"   📊 页面: http://localhost:{port}")
    print(f"   📁 静态: {REPORT_DIR}")
    print(f"   📝 配置: {CONFIG_PATH}")
    print(f"   API:")
    print(f"     GET  /api/stock_types")
    print(f"     POST /api/stock_type  {{symbol, stock_type}}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
        server.server_close()


if __name__ == "__main__":
    main()
