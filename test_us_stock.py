"""test_us_stock.py — 美股数据源逐一测试脚本

以 ADBE（Adobe）为例，测试所有可能获取美股历史数据的方式。
用法:
    python3 test_us_stock.py
    python3 test_us_stock.py --symbol AAPL
"""

import argparse
import json
import time

# ─── 测试项 ──────────────────────────────────────────

def test_akshare(symbol: str):
    """测试 akshare stock_us_hist"""
    print("\n" + "=" * 60)
    print(f"📡 [1] akshare  stock_us_hist  symbol=105.{symbol}")
    print("=" * 60)
    try:
        import akshare as ak
        df = ak.stock_us_hist(
            symbol=f"105.{symbol}", period="daily",
            start_date="20240101", end_date="21001231",
            adjust="qfq"
        )
        if df is not None and not df.empty:
            print(f"✅ 成功！获取 {len(df)} 条数据")
            print(df.tail(5).to_string(index=False))
            return True
        else:
            print("❌ 返回空数据")
            return False
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        return False


def test_yfinance(symbol: str):
    """测试 yfinance"""
    print("\n" + "=" * 60)
    print(f"📡 [2] yfinance  ticker={symbol}")
    print("=" * 60)
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max")
        if df is not None and not df.empty:
            print(f"✅ 成功！获取 {len(df)} 条数据")
            print(df.tail(5).to_string())
            return True
        else:
            print("❌ 返回空数据")
            return False
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        return False


def test_yfinance_download(symbol: str):
    """测试 yfinance download 方式（批量下载接口，有时比 Ticker.history 更稳定）"""
    print("\n" + "=" * 60)
    print(f"📡 [3] yfinance.download  ticker={symbol}")
    print("=" * 60)
    try:
        import yfinance as yf
        df = yf.download(symbol, start="2024-01-01", progress=False)
        if df is not None and not df.empty:
            print(f"✅ 成功！获取 {len(df)} 条数据")
            print(df.tail(5).to_string())
            return True
        else:
            print("❌ 返回空数据")
            return False
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        return False


def test_tencent(symbol: str):
    """测试腾讯财经"""
    print("\n" + "=" * 60)
    print(f"📡 [4] 腾讯财经  code=us{symbol}")
    print("=" * 60)
    try:
        import requests
        code = f"us{symbol}"
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {
            "param": f"{code},day,2024-01-01,2100-01-01,9999,qfq",
            "_var": "kline_dayqfq",
        }
        resp = requests.get(url, params=params,
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=20)
        text = resp.text
        json_str = text.split("=", 1)[-1] if "=" in text else text
        data = json.loads(json_str)

        raw_data = data.get("data", {})
        stock_data = raw_data.get(code, {}) if isinstance(raw_data, dict) else {}
        day_data = stock_data.get("qfqday") or stock_data.get("day") or []

        if day_data:
            print(f"✅ 成功！获取 {len(day_data)} 条数据")
            for row in day_data[-5:]:
                print(f"  {row}")
            return True
        else:
            print(f"❌ 返回空数据")
            print(f"  响应摘要: {text[:300]}")
            return False
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        return False


def test_eastmoney(symbol: str):
    """测试东方财富 HTTP 接口"""
    print("\n" + "=" * 60)
    print(f"📡 [5] 东方财富HTTP  secid=105.{symbol}")
    print("=" * 60)
    try:
        import requests
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": f"105.{symbol}",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": 101,
            "fqt": 1,
            "beg": "20240101",
            "end": "21000101",
        }
        resp = requests.get(url, params=params,
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=20)
        data = resp.json()
        klines = data.get("data", {}).get("klines", []) if data.get("data") else []

        if klines:
            print(f"✅ 成功！获取 {len(klines)} 条数据")
            for line in klines[-5:]:
                print(f"  {line}")
            return True
        else:
            # 再试 106 (纽约证交所)
            params["secid"] = f"106.{symbol}"
            resp = requests.get(url, params=params,
                                headers={"User-Agent": "Mozilla/5.0"},
                                timeout=20)
            data = resp.json()
            klines = data.get("data", {}).get("klines", []) if data.get("data") else []
            if klines:
                print(f"✅ 成功(106)！获取 {len(klines)} 条数据")
                for line in klines[-5:]:
                    print(f"  {line}")
                return True
            print(f"❌ 105 和 106 都返回空数据")
            print(f"  data 字段: {data.get('data')}")
            return False
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        return False


def test_alphavantage(symbol: str):
    """测试 Alpha Vantage（免费 key，每天 25 次调用）"""
    print("\n" + "=" * 60)
    print(f"📡 [6] Alpha Vantage (free demo key)  symbol={symbol}")
    print("=" * 60)
    try:
        import requests
        # 使用 demo key，实际使用需要注册获取免费 key
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": "demo",  # demo key 仅支持 IBM
        }
        resp = requests.get(url, params=params, timeout=20)
        data = resp.json()

        ts = data.get("Time Series (Daily)", {})
        if ts:
            print(f"✅ 成功！获取 {len(ts)} 条数据")
            for date in list(ts.keys())[:5]:
                print(f"  {date}: {ts[date]}")
            return True
        else:
            print(f"❌ 返回空或限制: {list(data.keys())}")
            if "Note" in data:
                print(f"  Note: {data['Note']}")
            if "Information" in data:
                print(f"  Info: {data['Information']}")
            return False
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        return False


def test_stooq(symbol: str):
    """测试 Stooq 免费接口（波兰站，支持美股历史数据，无需 API key）"""
    print("\n" + "=" * 60)
    print(f"📡 [7] Stooq CSV  symbol={symbol}.US")
    print("=" * 60)
    try:
        import pandas as pd
        url = f"https://stooq.com/q/d/l/?s={symbol}.us&d1=20240101&d2=20261231&i=d"
        df = pd.read_csv(url)
        if df is not None and not df.empty and "Close" in df.columns:
            print(f"✅ 成功！获取 {len(df)} 条数据")
            print(df.tail(5).to_string(index=False))
            return True
        else:
            print(f"❌ 返回空或格式异常")
            if df is not None:
                print(f"  columns: {list(df.columns)}")
                print(f"  head: {df.head()}")
            return False
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        return False


def test_yahoo_chart_api(symbol: str):
    """测试 Yahoo Finance Chart API（直接 HTTP 请求，不依赖 yfinance 库）"""
    print("\n" + "=" * 60)
    print(f"📡 [8] Yahoo Chart API (直接HTTP)  symbol={symbol}")
    print("=" * 60)
    try:
        import requests
        import datetime

        # 获取最近一年的数据
        end_ts = int(datetime.datetime.now().timestamp())
        start_ts = int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp())

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        data = resp.json()

        result = data.get("chart", {}).get("result", [])
        if result:
            timestamps = result[0].get("timestamp", [])
            quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
            print(f"✅ 成功！获取 {len(timestamps)} 条数据")
            # 打印最后5条
            for i in range(-5, 0):
                if i + len(timestamps) >= 0:
                    idx = i + len(timestamps)
                    dt = datetime.datetime.fromtimestamp(timestamps[idx]).strftime("%Y-%m-%d")
                    c = quotes.get("close", [])[idx]
                    v = quotes.get("volume", [])[idx]
                    print(f"  {dt}  close={c}  volume={v}")
            return True
        else:
            err = data.get("chart", {}).get("error", {})
            print(f"❌ 返回空数据: {err}")
            return False
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        return False


# ─── 主程序 ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="美股数据源测试")
    parser.add_argument("--symbol", default="ADBE", help="美股 ticker，默认 ADBE (Adobe)")
    args = parser.parse_args()
    symbol = args.symbol.upper()

    print(f"🔍 测试美股数据源 — 标的: {symbol}")
    print(f"{'=' * 60}")

    results = {}
    tests = [
        ("akshare", test_akshare),
        ("yfinance (Ticker.history)", test_yfinance),
        ("yfinance (download)", test_yfinance_download),
        ("腾讯财经", test_tencent),
        ("东方财富HTTP", test_eastmoney),
        ("Alpha Vantage (demo)", test_alphavantage),
        ("Stooq CSV", test_stooq),
        ("Yahoo Chart API", test_yahoo_chart_api),
    ]

    for name, test_func in tests:
        try:
            ok = test_func(symbol)
        except Exception as e:
            print(f"❌ 未捕获异常: {type(e).__name__}: {e}")
            ok = False
        results[name] = ok
        time.sleep(1)  # 避免请求太快

    # ─── 汇总 ───
    print("\n\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    for name, ok in results.items():
        status = "✅ 可用" if ok else "❌ 不可用"
        print(f"  {status}  {name}")

    available = [n for n, ok in results.items() if ok]
    print(f"\n可用数据源: {len(available)}/{len(results)}")
    if available:
        print(f"推荐优先级: {' → '.join(available)}")
    else:
        print("⚠️  所有数据源均不可用，请检查网络/代理设置")


if __name__ == "__main__":
    main()
