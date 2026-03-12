"""stock_history_v2.py  —  统一历史数据构建脚本

数据来源三层优先级：
  1. 存量迁移  : data/{symbol}.json (已有 akshare 历史 K 线)
               + pedata/{symbol}.json (已有历史 PE)
               合并写入 newdata/{symbol}.json
  2. akshare兜底: 若 data/ 无数据，用 akshare 拉全量历史 K 线
               + 东方财富 EPS 计算 PE(TTM) 写入 newdata/
  3. 增量更新  : 每次运行时，用 stock-data CLI 补充最新 N 条 K 线
               + quote 接口写入最新交易日的 PE

输出格式 (newdata/{symbol}.json):
  [{date, open, high, low, close, volume, amount, pe}, ...]

运行方式:
  python3 stock_history_v2.py           # 全量：迁移+兜底+增量
  python3 stock_history_v2.py --incr    # 仅增量更新（跳过迁移/兜底）
"""

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
import requests

CONFIG_FILE = "config.json"
DATA_DIR    = "data"        # 旧 K 线目录
PE_DATA_DIR = "pedata"      # 旧 PE 目录
NEW_DATA_DIR = "newdata"    # 新统一目录

STOCK_DATA_BIN = "/projects/.openclaw/skills/stock-data/bin/stock-data-linux-x86_64"
KLINE_INCR_COUNT = 30       # 增量拉取条数
REQUEST_INTERVAL = 3        # stock-data 请求间隔秒数
BATCH_SLEEP = 1.0           # akshare / EPS 请求间隔秒数


# ══════════════════════════════════════════
#  工具：文件路径
# ══════════════════════════════════════════

def _base_name(item: dict) -> str:
    symbol = item["symbol"]
    if item["market"] == "us":
        return f"{symbol.split('.')[-1]}.json"
    return f"{symbol}.json"

def old_kline_path(item): return os.path.join(DATA_DIR, _base_name(item))
def old_pe_path(item):    return os.path.join(PE_DATA_DIR, _base_name(item))
def new_path(item):       return os.path.join(NEW_DATA_DIR, _base_name(item))


# ══════════════════════════════════════════
#  工具：JSON 读写 & 合并
# ══════════════════════════════════════════

def load_json(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_json(records: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def merge_by_date(base: list[dict], overlay: list[dict]) -> list[dict]:
    """合并两份记录，overlay 覆盖同日期 base，按日期升序返回。"""
    merged = {r["date"]: r for r in base}
    merged.update({r["date"]: r for r in overlay})
    return sorted(merged.values(), key=lambda x: x["date"])


# ══════════════════════════════════════════
#  层 1：存量迁移（data/ + pedata/ → newdata/）
# ══════════════════════════════════════════

def migrate_from_old_dirs(item: dict) -> list[dict]:
    """
    读取 data/{sym}.json（K 线）和 pedata/{sym}.json（PE），
    按日期对齐合并，返回含 pe 字段的完整记录列表。
    """
    kline_records = load_json(old_kline_path(item))
    if not kline_records:
        return []

    # PE 字典：{date: pe_value}
    pe_raw = load_json(old_pe_path(item))
    pe_map = {r["date"]: r.get("pe") for r in pe_raw if "date" in r}

    result = []
    for r in kline_records:
        date = r.get("date") or r.get("Date") or r.get("日期", "")
        if not date:
            continue
        # 统一日期格式 YYYY-MM-DD
        date = str(date)[:10]
        result.append({
            "date":   date,
            "open":   _to_float(r.get("open") or r.get("Open")),
            "high":   _to_float(r.get("high") or r.get("High")),
            "low":    _to_float(r.get("low")  or r.get("Low")),
            "close":  _to_float(r.get("close") or r.get("Close")),
            "volume": _to_float(r.get("volume") or r.get("Volume")),
            "amount": _to_float(r.get("amount") or r.get("Amount")),
            "pe":     pe_map.get(date),
        })

    result.sort(key=lambda x: x["date"])
    return result


# ══════════════════════════════════════════
#  层 2：akshare 全量历史 + EPS PE 计算
# ══════════════════════════════════════════

def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None   # NaN check
    except (TypeError, ValueError):
        return None


def fetch_akshare_kline(item: dict) -> list[dict]:
    """用 akshare 拉取全量日线历史，返回标准记录列表。"""
    market = item["market"]
    symbol = item["symbol"]
    try:
        if market == "a":
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date="19900101", end_date="21001231",
                adjust="qfq"
            )
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"
            })
        elif market == "hk":
            df = ak.stock_hk_hist(
                symbol=symbol, period="daily",
                start_date="19900101", end_date="21001231",
                adjust="qfq"
            )
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"
            })
        elif market == "us":
            ticker = symbol.split(".")[-1]
            df = ak.stock_us_hist(
                symbol=symbol, period="daily",
                start_date="19900101", end_date="21001231",
                adjust="qfq"
            )
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"
            })
        else:
            return []

        if df is None or df.empty:
            return []

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        records = []
        for _, row in df.iterrows():
            records.append({
                "date":   row["date"],
                "open":   _to_float(row.get("open")),
                "high":   _to_float(row.get("high")),
                "low":    _to_float(row.get("low")),
                "close":  _to_float(row.get("close")),
                "volume": _to_float(row.get("volume")),
                "amount": _to_float(row.get("amount")),
                "pe":     None,
            })
        return records

    except Exception:
        traceback.print_exc()
        return []


# ── EPS & PE 计算（直接从 pe_store.py 引入核心逻辑）──

def _get_eastmoney_secucode(market: str, symbol: str):
    if market == "hk":
        return f"{symbol}.HK"
    elif market == "us":
        ticker = symbol.split(".")[-1]
        return [f"{ticker}.O", f"{ticker}.N"]
    else:
        suffix = "SH" if symbol.startswith(("6", "9")) else "SZ"
        return f"{symbol}.{suffix}"


def _fetch_quarterly_eps(market: str, symbol: str) -> list[dict] | None:
    secucode_raw = _get_eastmoney_secucode(market, symbol)
    secucodes = secucode_raw if isinstance(secucode_raw, list) else [secucode_raw]

    if market == "hk":
        report_name, date_field = "RPT_HKF10_FN_MAININDICATOR", "REPORT_DATE"
    elif market == "us":
        report_name, date_field = "RPT_USF10_FN_GMAININDICATOR", "REPORT_DATE"
    else:
        report_name, date_field = "RPT_LICO_FN_CPD", "REPORTDATE"

    for secucode in secucodes:
        try:
            url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
            params = {
                "reportName": report_name,
                "columns": f"SECUCODE,{date_field},BASIC_EPS",
                "filter": f'(SECUCODE="{secucode}")',
                "pageNumber": 1, "pageSize": 50,
                "sortTypes": "-1", "sortColumns": date_field,
            }
            resp = requests.get(url, params=params,
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            data = resp.json()
            if not data.get("result") or not data["result"].get("data"):
                continue
            result = []
            for r in data["result"]["data"]:
                rd = r.get(date_field, "")[:10]
                eps = r.get("BASIC_EPS")
                if rd and eps is not None:
                    result.append({"report_date": rd, "eps": float(eps)})
            if result:
                return result
        except Exception:
            continue
    return None


def _calc_ttm_eps(quarterly_eps: list[dict], target_date: str) -> float | None:
    """计算目标日期对应的 TTM EPS（完整逻辑来自 pe_store.py）。"""
    from datetime import datetime as dt

    target = dt.strptime(target_date, "%Y-%m-%d")
    sorted_eps = sorted(quarterly_eps, key=lambda x: x["report_date"])

    def is_published(rd_str, at_date):
        rd = dt.strptime(rd_str, "%Y-%m-%d")
        m = rd.month
        if m == 3:   pub = rd.replace(month=4, day=30)
        elif m == 6: pub = rd.replace(month=8, day=31)
        elif m == 9: pub = rd.replace(month=10, day=31)
        elif m == 12:pub = rd.replace(year=rd.year+1, month=3, day=31)
        else:        pub = rd + timedelta(days=90)
        return at_date >= pub

    available = [e for e in sorted_eps if is_published(e["report_date"], target)]
    if not available:
        return None

    all_quarters = set()
    eps_by_yq: dict = {}
    for e in sorted_eps:
        rd = dt.strptime(e["report_date"], "%Y-%m-%d")
        q = (rd.month - 1) // 3 + 1
        all_quarters.add(q)
        eps_by_yq[(rd.year, q)] = e["eps"]

    is_semi = all_quarters.issubset({2, 4})
    if not is_semi and 1 not in all_quarters and 3 in all_quarters and 2 in all_quarters:
        q2_eq_q3 = all(
            abs(eps_by_yq[(y, 2)] - eps_by_yq.get((y, 3), float("inf"))) < 0.001
            for (y, q) in eps_by_yq if q == 2 and (y, 3) in eps_by_yq
        )
        if q2_eq_q3:
            is_semi = True
            sorted_eps = [e for e in sorted_eps
                          if (dt.strptime(e["report_date"], "%Y-%m-%d").month - 1) // 3 + 1 != 3]
            available = [e for e in sorted_eps if is_published(e["report_date"], target)]

    eps_by_year: dict = {}
    for e in sorted_eps:
        rd = dt.strptime(e["report_date"], "%Y-%m-%d")
        eps_by_year.setdefault(rd.year, {})[(rd.month - 1) // 3 + 1] = e["eps"]

    if is_semi:
        half: dict = {}
        for year, periods in eps_by_year.items():
            if 2 in periods: half[(year, "H1")] = periods[2]
            if 4 in periods:
                half[(year, "H2")] = periods[4] - periods.get(2, 0)
        latest = available[-1]
        rd = dt.strptime(latest["report_date"], "%Y-%m-%d")
        y, q = rd.year, (rd.month - 1) // 3 + 1
        if q == 4:
            h1 = half.get((y, "H1")); h2 = half.get((y, "H2"))
            return (h1 + h2) if h1 is not None and h2 is not None else None
        elif q == 2:
            h1 = half.get((y, "H1")); h2 = half.get((y-1, "H2"))
            return (h1 + h2) if h1 is not None and h2 is not None else None
        return None
    else:
        sq: dict = {}
        for year, quarters in eps_by_year.items():
            for q in sorted(quarters):
                sq[(year, q)] = quarters[q] if q == 1 else (
                    quarters[q] - quarters.get(q-1, 0))
        latest = available[-1]
        rd = dt.strptime(latest["report_date"], "%Y-%m-%d")
        y, q = rd.year, (rd.month - 1) // 3 + 1
        ttm, count = 0.0, 0
        for _ in range(4):
            if (y, q) in sq:
                ttm += sq[(y, q)]; count += 1
            q -= 1
            if q == 0: q, y = 4, y - 1
        return ttm if count == 4 else None


def calc_pe_series(quarterly_eps: list[dict], records: list[dict]) -> list[dict]:
    """为记录列表中每条计算 pe 字段（已有 pe 的跳过）。"""
    ttm_cache: dict = {}
    result = []
    for r in records:
        d = r["date"]
        c = r.get("close")
        if c is None or c <= 0:
            result.append(r); continue

        mk = d[:7]
        if mk not in ttm_cache:
            ttm_cache[mk] = _calc_ttm_eps(quarterly_eps, d)
        ttm = ttm_cache[mk]

        new_r = dict(r)
        if ttm and ttm != 0 and r.get("pe") is None:
            new_r["pe"] = round(c / ttm, 2)
        result.append(new_r)
    return result


def fetch_full_with_pe(item: dict) -> list[dict]:
    """层2：akshare 拉历史 K 线 + 东方财富计算 PE，返回完整记录。"""
    records = fetch_akshare_kline(item)
    if not records:
        return []

    time.sleep(BATCH_SLEEP)
    eps = _fetch_quarterly_eps(item["market"], item["symbol"])
    if eps:
        records = calc_pe_series(eps, records)

    return records


# ══════════════════════════════════════════
#  层 3：stock-data 增量更新
# ══════════════════════════════════════════

def to_stockdata_code(item: dict) -> str:
    market, symbol = item["market"], item["symbol"]
    if market == "us":
        return f"us{symbol.split('.')[-1]}"
    elif market == "hk":
        return f"hk{symbol}"
    elif market == "a":
        return f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}"
    return symbol


def _run_stockdata(args: list) -> dict | None:
    try:
        r = subprocess.run([STOCK_DATA_BIN] + args,
                           capture_output=True, text=True, timeout=30)
        lines = [l for l in r.stdout.splitlines() if not l.startswith("[")]
        clean = "\n".join(lines).strip()
        if not clean or clean == "null":
            return None
        return json.loads(clean)
    except Exception:
        return None


def fetch_stockdata_incr(item: dict) -> list[dict]:
    """用 stock-data kline 拉最近 KLINE_INCR_COUNT 条日线。"""
    code = to_stockdata_code(item)
    time.sleep(REQUEST_INTERVAL)
    data = _run_stockdata(["kline", code, "day", str(KLINE_INCR_COUNT), "qfq"])
    if not data or data.get("code", -1) != 0:
        return []
    nodes = data.get("data", {}).get("nodes", [])
    records = []
    for node in nodes:
        d = node.get("date", "")
        if not d:
            continue
        records.append({
            "date":   d,
            "open":   _to_float(node.get("open")),
            "high":   _to_float(node.get("high")),
            "low":    _to_float(node.get("low")),
            "close":  _to_float(node.get("last")),
            "volume": _to_float(node.get("volume")),
            "amount": _to_float(node.get("amount")),
            "pe":     None,
        })
    records.sort(key=lambda x: x["date"])
    return records


def fetch_stockdata_pe(item: dict) -> float | None:
    """用 stock-data quote 获取当前 PE。"""
    code = to_stockdata_code(item)
    time.sleep(REQUEST_INTERVAL)
    data = _run_stockdata(["quote", code])
    if not data:
        return None
    try:
        stock = list(data.values())[0]
        pe = stock.get("pe_ratio")
        return round(float(pe), 2) if pe and pe != 0 else None
    except Exception:
        return None


def get_last_trading_date() -> str:
    d = datetime.now()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


# ══════════════════════════════════════════
#  主逻辑：处理单只股票
# ══════════════════════════════════════════

def process_item(item: dict, incr_only: bool = False) -> dict:
    name   = item["name"]
    market = item["market"]
    out    = new_path(item)

    if market == "crypto":
        return {"name": name, "status": "skipped", "detail": "crypto暂不支持"}

    # ── 读取已有 newdata ──
    existing = load_json(out)

    # ── 检查是否需要增量 ──
    last_td = get_last_trading_date()
    if existing:
        dates = [r["date"] for r in existing if r.get("date")]
        last_date = max(dates) if dates else ""
        need_incr = last_date < last_td
    else:
        need_incr = True
        last_date = ""

    # ════════════════════════════
    #  全量建立阶段（首次 / 无存量）
    # ════════════════════════════
    if not existing and not incr_only:
        base = []

        # 层1：存量迁移
        migrated = migrate_from_old_dirs(item)
        if migrated:
            base = migrated
            source = f"迁移 data/+pedata/ ({len(base)}条)"
        else:
            # 层2：akshare 兜底
            base = fetch_full_with_pe(item)
            source = f"akshare全量 ({len(base)}条)"

        if not base:
            return {"name": name, "status": "failed", "detail": "无法获取历史数据"}

        existing = base
        save_json(existing, out)
        dates = [r["date"] for r in existing if r.get("date")]
        last_date = max(dates) if dates else ""
        # 更新 need_incr
        need_incr = last_date < last_td
    elif not existing and incr_only:
        return {"name": name, "status": "skipped", "detail": "无本地数据，跳过（--incr模式）"}

    # ════════════════════════════
    #  增量更新阶段
    # ════════════════════════════
    added = 0
    if need_incr:
        incr_records = fetch_stockdata_incr(item)
        if incr_records:
            existing_dates = {r["date"] for r in existing}
            truly_new = [r for r in incr_records if r["date"] not in existing_dates]

            if truly_new:
                # 为最新记录补 PE
                pe_today = fetch_stockdata_pe(item)
                if pe_today is not None:
                    truly_new[-1]["pe"] = pe_today

                existing = merge_by_date(existing, truly_new)
                added = len(truly_new)
                save_json(existing, out)

    total = len(existing)
    date_range = f"{existing[0]['date']} ~ {existing[-1]['date']}" if existing else "N/A"

    if added:
        detail = f"+{added}条增量，共{total}条（{date_range}）"
        status = "updated"
    elif not need_incr:
        detail = f"已是最新（{last_date}），共{total}条"
        status = "skipped"
    else:
        detail = f"增量无新数据，共{total}条（{date_range}）"
        status = "skipped"

    return {"name": name, "status": status, "detail": detail}


# ══════════════════════════════════════════
#  入口
# ══════════════════════════════════════════

def load_config() -> list[dict]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["stocks"]


def run_update(incr_only: bool = False) -> str:
    os.makedirs(NEW_DATA_DIR, exist_ok=True)
    config = load_config()

    updated, skipped, failed = [], [], []

    for item in config:
        result = process_item(item, incr_only=incr_only)
        bucket = {"updated": updated, "skipped": skipped, "failed": failed}.get(
            result["status"], failed)
        bucket.append(result)
        # 简单进度打印（实时可见）
        sym = "✅" if result["status"] == "updated" else (
              "⏭" if result["status"] == "skipped" else "❌")
        print(f"  {sym} {result['name']}: {result['detail']}")

    lines = [
        f"\n[stock_history_v2] 完成：{len(updated)}已更新，"
        f"{len(skipped)}无需更新，{len(failed)}失败"
    ]
    if failed:
        lines.append(f"\n❌ 失败列表：")
        for r in failed:
            lines.append(f"  {r['name']}：{r['detail']}")

    report = "\n".join(lines)
    print(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incr", action="store_true",
                        help="仅增量更新，跳过迁移/akshare全量拉取")
    args = parser.parse_args()
    run_update(incr_only=args.incr)
