"""stock_history_v2.py

使用 stock-data CLI（腾讯证券接口）拉取历史 K 线数据，
同时记录每日 PE 值，写入 newdata/ 目录。

数据字段：date, symbol, open, high, low, close, volume, amount, pe

逻辑：
- 无本地数据 → 拉取最多 150 条历史（qfq 前复权，约7个月）
- 有本地数据且非最新 → 拉取最近 30 条，取差值合并
- 已是最新 → 跳过

注意：接口有频率限制，每次请求间隔 3 秒。
"""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta

CONFIG_FILE = "config.json"
NEW_DATA_DIR = "newdata"

# stock-data 二进制路径
STOCK_DATA_BIN = "/projects/.openclaw/skills/stock-data/bin/stock-data-linux-x86_64"

# 接口参数
KLINE_FULL_COUNT = 150   # 首次拉取条数（接口上限约150）
KLINE_INCR_COUNT = 30    # 增量拉取条数（最近30个交易日足够覆盖差值）
REQUEST_INTERVAL = 3     # 请求间隔秒数（避免频率限制）


# ──────────────────────────────────────
#  代码格式转换：config.json → stock-data 格式
# ──────────────────────────────────────

def to_stockdata_code(item: dict) -> str:
    """将 config.json 中的 symbol 转换为 stock-data CLI 所需的代码格式。"""
    market = item["market"]
    symbol = item["symbol"]

    if market == "us":
        # "105.AAPL" / "106.TME" → "usAAPL"
        ticker = symbol.split(".")[-1]
        return f"us{ticker}"
    elif market == "hk":
        # "03690" → "hk03690"
        return f"hk{symbol}"
    elif market == "a":
        # 沪市：6xxxx / 688xxx / 9xxxx → sh
        # 深市：0xxxx / 3xxxx → sz
        if symbol.startswith("6") or symbol.startswith("9"):
            return f"sh{symbol}"
        else:
            return f"sz{symbol}"
    else:
        return symbol


def get_output_filename(item: dict) -> str:
    """生成 newdata/ 下的输出文件名。"""
    symbol = item["symbol"]
    if item["market"] == "us":
        name = f"{symbol.split('.')[-1]}.json"
    else:
        name = f"{symbol}.json"
    return os.path.join(NEW_DATA_DIR, name)


# ──────────────────────────────────────
#  调用 stock-data CLI
# ──────────────────────────────────────

def run_cmd(args: list) -> dict | None:
    """执行 stock-data 命令，返回解析后的 JSON，失败返回 None。"""
    cmd = [STOCK_DATA_BIN] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # stdout 可能包含 [HTTP Request] 日志行，过滤掉
        lines = [l for l in result.stdout.splitlines() if not l.startswith("[")]
        clean = "\n".join(lines).strip()
        if not clean or clean == "null":
            return None
        return json.loads(clean)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def fetch_kline(code: str, count: int) -> list[dict]:
    """拉取最近 count 个交易日的 K 线数据（前复权）。"""
    data = run_cmd(["kline", code, "day", str(count), "qfq"])
    if not data:
        return []

    try:
        if data.get("code", -1) != 0:
            return []
        nodes = data.get("data", {}).get("nodes", [])
    except AttributeError:
        return []

    records = []
    for node in nodes:
        try:
            def to_float(v):
                try:
                    return float(v) if v not in (None, "") else None
                except (ValueError, TypeError):
                    return None

            records.append({
                "date": node.get("date", ""),
                "open": to_float(node.get("open")),
                "high": to_float(node.get("high")),
                "low": to_float(node.get("low")),
                "close": to_float(node.get("last")),   # kline 里收盘价字段名是 last
                "volume": to_float(node.get("volume")),
                "amount": to_float(node.get("amount")),
                "pe": None,                             # kline 无 PE，稍后用 quote 补
            })
        except Exception:
            continue

    records = [r for r in records if r["date"]]
    records.sort(key=lambda x: x["date"])
    return records


def fetch_quote_pe(code: str) -> float | None:
    """用 quote 接口获取当前 PE 值。"""
    data = run_cmd(["quote", code])
    if not data:
        return None
    try:
        stock = list(data.values())[0]
        pe = stock.get("pe_ratio")
        if pe is not None and pe != 0:
            return round(float(pe), 2)
        return None
    except Exception:
        return None


# ──────────────────────────────────────
#  本地数据读写
# ──────────────────────────────────────

def load_local(filename: str) -> list[dict]:
    """读取本地 JSON 数据，返回记录列表。"""
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_local(records: list[dict], filename: str):
    """写入本地 JSON 文件。"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def merge_records(local: list[dict], new: list[dict]) -> list[dict]:
    """合并本地和新数据，按日期去重（新数据优先），排序返回。"""
    merged = {r["date"]: r for r in local}
    for r in new:
        merged[r["date"]] = r
    result = sorted(merged.values(), key=lambda x: x["date"])
    return result


# ──────────────────────────────────────
#  增量逻辑
# ──────────────────────────────────────

def get_last_trading_date() -> str:
    """获取最近一个交易日（排除周末）。"""
    d = datetime.now()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def fetch_and_save(item: dict) -> dict:
    """拉取并保存单只股票数据，返回 {name, status, detail}。"""
    name = item["name"]
    code = to_stockdata_code(item)
    filename = get_output_filename(item)

    local = load_local(filename)

    # ── 判断是否需要更新 ──
    last_td = get_last_trading_date()
    if local:
        local_dates = [r["date"] for r in local if r.get("date")]
        last_local_date = max(local_dates) if local_dates else None
        if last_local_date and last_local_date >= last_td:
            return {"name": name, "status": "skipped", "detail": f"已是最新（{last_local_date}）"}

    # ── 拉取 K 线 ──
    # 有本地数据 → 增量（30条覆盖差值）；无本地数据 → 全量（150条）
    count = KLINE_INCR_COUNT if local else KLINE_FULL_COUNT
    time.sleep(REQUEST_INTERVAL)
    new_records = fetch_kline(code, count=count)

    if not new_records:
        return {"name": name, "status": "failed", "detail": f"K线数据为空（{code}）"}

    # 过滤掉本地已有的日期
    if local:
        local_dates_set = {r["date"] for r in local if r.get("date")}
        truly_new = [r for r in new_records if r["date"] not in local_dates_set]
    else:
        truly_new = new_records

    if not truly_new and local:
        last_d = max(r["date"] for r in local if r.get("date"))
        return {"name": name, "status": "skipped", "detail": f"已是最新（{last_d}）"}

    # ── 为最新一条记录补充当日 PE（来自 quote 接口）──
    time.sleep(REQUEST_INTERVAL)
    current_pe = fetch_quote_pe(code)
    if truly_new and current_pe is not None:
        truly_new[-1]["pe"] = current_pe

    # ── 合并保存 ──
    all_new = new_records if not local else truly_new
    merged = merge_records(local, all_new)
    save_local(merged, filename)

    new_count = len(truly_new)
    total = len(merged)
    date_range = f"{merged[0]['date']} ~ {merged[-1]['date']}"
    return {"name": name, "status": "updated", "detail": f"+{new_count}条，共{total}条（{date_range}）"}


# ──────────────────────────────────────
#  主入口
# ──────────────────────────────────────

def load_config() -> list[dict]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["stocks"]


def run_update() -> str:
    os.makedirs(NEW_DATA_DIR, exist_ok=True)
    config = load_config()

    updated = []
    skipped = []
    failed = []

    for item in config:
        market = item.get("market", "")
        # crypto 暂不支持（stock-data 不覆盖）
        if market == "crypto":
            failed.append({"name": item["name"], "status": "failed", "detail": "crypto市场暂不支持"})
            continue

        result = fetch_and_save(item)

        if result["status"] == "updated":
            updated.append(result)
        elif result["status"] == "skipped":
            skipped.append(result)
        else:
            failed.append(result)

    lines = []
    lines.append(f"[stock_history_v2] 数据更新完成：{len(updated)}个已更新，{len(skipped)}个无需更新，{len(failed)}个失败")

    if updated:
        lines.append(f"\n✅ 已更新（{len(updated)}）：")
        for r in updated:
            lines.append(f"  {r['name']}：{r['detail']}")

    if failed:
        lines.append(f"\n❌ 更新失败（{len(failed)}）：")
        for r in failed:
            lines.append(f"  {r['name']}：{r['detail']}")

    report = "\n".join(lines)
    print(report)
    return report


if __name__ == "__main__":
    run_update()
