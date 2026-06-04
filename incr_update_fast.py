"""
incr_update_fast.py - 快速增量更新脚本
使用腾讯财经API（A股/港股）和 CoinGecko（BTC）进行增量更新
对于已有最新数据的股票直接跳过
"""
import json
import os
import glob
import time
import re
import requests
from datetime import datetime, timedelta

NEW_DATA_DIR = "/root/.openclaw/workspace/newdata"
CONFIG_FILE = "/root/.openclaw/workspace/config.json"
TARGET_DATE = "2026-04-24"

def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["stocks"]

def load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def to_tencent_code(item):
    market, symbol = item["market"], item["symbol"]
    if market == "a":
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        return f"{prefix}{symbol}"
    elif market == "hk":
        return f"hk{symbol}"
    elif market == "us":
        ticker = symbol.split(".")[-1]
        return f"us{ticker}"
    return None

def fetch_tencent_kline(item, start_date, end_date):
    """从腾讯财经拉取日线数据"""
    code = to_tencent_code(item)
    if not code:
        return []
    
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {
        "_var": "kline_dayfqkline",
        "param": f"{code},day,{start_date},{end_date},60,qfq",
        "r": "0.12345",
    }
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, params=params, timeout=15)
        text = resp.text
        # 去掉 JS 变量前缀
        if text.startswith("kline_dayfqkline="):
            text = text[len("kline_dayfqkline="):]
        data = json.loads(text)
        
        code_data = data.get("data", {}).get(code, {})
        # 优先取 qfqday（前复权），否则取 day
        klines = code_data.get("qfqday") or code_data.get("day") or []
        
        records = []
        for row in klines:
            if len(row) < 6:
                continue
            date_str = row[0]
            if not (start_date <= date_str <= end_date):
                continue
            records.append({
                "date": date_str,
                "open": float(row[1]) if row[1] else None,
                "close": float(row[2]) if row[2] else None,
                "high": float(row[3]) if row[3] else None,
                "low": float(row[4]) if row[4] else None,
                "volume": float(row[5]) if row[5] else None,
                "amount": None,
                "pe": None,
            })
        return records
    except Exception as e:
        return []

def fetch_coingecko_btc(days=3):
    """从 CoinGecko 拉取 BTC 数据"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        vol_map = {}
        for ts_ms, vol in volumes:
            d = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            vol_map[d] = vol
        seen = {}
        for ts_ms, price in prices:
            d = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            seen[d] = {
                "date": d, "open": price, "high": price,
                "low": price, "close": price,
                "volume": vol_map.get(d, 0), "amount": 0, "pe": None,
            }
        return sorted(seen.values(), key=lambda x: x["date"])
    except Exception as e:
        return []

def merge_by_date(base, overlay):
    merged = {r["date"]: r for r in base}
    merged.update({r["date"]: r for r in overlay})
    return sorted(merged.values(), key=lambda x: x["date"])

def main():
    config = load_config()
    os.makedirs(NEW_DATA_DIR, exist_ok=True)
    
    results = {"updated": [], "skipped": [], "failed": []}
    
    # 统计需要更新的
    need_update = []
    for item in config:
        symbol = item["symbol"]
        market = item["market"]
        name = item.get("name", symbol)
        
        # 确定输出文件名
        if market == "us":
            fname = f"{symbol.split('.')[-1]}.json"
        else:
            fname = f"{symbol}.json"
        
        out_path = os.path.join(NEW_DATA_DIR, fname)
        existing = load_json(out_path)
        
        if not existing:
            need_update.append((item, out_path, ""))
            continue
        
        last_date = max(r["date"] for r in existing if r.get("date"))
        if last_date >= TARGET_DATE:
            results["skipped"].append({"name": name, "detail": f"已最新({last_date})"})
        else:
            need_update.append((item, out_path, last_date))
    
    print(f"[incr_update_fast] 共 {len(config)} 只，{len(results['skipped'])} 已最新，{len(need_update)} 需更新")
    print()
    
    for idx, (item, out_path, last_date) in enumerate(need_update):
        symbol = item["symbol"]
        market = item["market"]
        name = item.get("name", symbol)
        
        existing = load_json(out_path)
        start_date = last_date if last_date else "2026-04-20"
        
        new_records = []
        
        if market == "crypto":
            new_records = fetch_coingecko_btc(days=5)
        else:
            new_records = fetch_tencent_kline(item, start_date, TARGET_DATE)
        
        if new_records:
            if existing:
                existing_dates = {r["date"] for r in existing}
                truly_new = [r for r in new_records if r["date"] not in existing_dates]
            else:
                truly_new = new_records
            
            if truly_new:
                merged = merge_by_date(existing, truly_new)
                save_json(merged, out_path)
                print(f"  ✅ {name}: +{len(truly_new)}条 → 最新{truly_new[-1]['date']}")
                results["updated"].append({"name": name, "detail": f"+{len(truly_new)}条"})
            else:
                print(f"  ⏭ {name}: 无新数据（已至{last_date}）")
                results["skipped"].append({"name": name, "detail": f"无新数据"})
        else:
            print(f"  ❌ {name}: 获取失败（{market}市场）")
            results["failed"].append({"name": name, "detail": f"数据获取失败"})
        
        # 简单限速
        if market != "crypto":
            time.sleep(0.3)
    
    print()
    print(f"[incr_update_fast] 完成：{len(results['updated'])}已更新，"
          f"{len(results['skipped'])}无需更新，{len(results['failed'])}失败")
    
    if results["failed"]:
        print(f"\n❌ 失败列表：")
        for r in results["failed"]:
            print(f"  {r['name']}：{r['detail']}")
    
    return results

if __name__ == "__main__":
    main()
