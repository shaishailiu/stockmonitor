import akshare as ak
import pandas as pd
import requests
import os
import json
from datetime import datetime, timedelta

CONFIG_FILE = "config.json"
DATA_DIR = "data"


# 统一字段映射：中文 -> 英文（东方财富接口）
COLUMN_MAP = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "change_pct",
    "涨跌额": "change",
    "换手率": "turnover_rate",
}

# 新浪接口字段映射
COLUMN_MAP_SINA = {
    "outstanding_share": "_outstanding_share",
    "turnover": "turnover_rate",
}


def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """统一 DataFrame 的字段名和日期格式。"""
    df = df.rename(columns=COLUMN_MAP)

    # 统一日期为 YYYY-MM-DD 字符串
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # 确保有 symbol 列
    if "symbol" not in df.columns:
        df.insert(0, "symbol", symbol)

    # 按最全字段集排列，缺失的填 None
    full_columns = [
        "date", "symbol", "open", "high", "low", "close",
        "volume", "amount", "amplitude", "change_pct", "change",
        "turnover_rate", "market_cap",
    ]
    for col in full_columns:
        if col not in df.columns:
            df[col] = None
    df = df[full_columns]
    df = df.reset_index(drop=True)
    return df


def _default_date_range():
    today = datetime.now()
    end_date = today.strftime("%Y%m%d")
    start_date = (today - timedelta(weeks=52)).strftime("%Y%m%d")
    return start_date, end_date


def get_us_stock_history(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """获取美股每日历史数据。"""
    if start_date is None or end_date is None:
        s, e = _default_date_range()
        start_date = start_date or s
        end_date = end_date or e
    df = ak.stock_us_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    ticker = symbol.split(".")[-1]
    return _normalize(df, ticker)


def get_a_stock_history(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """获取A股每日历史数据。"""
    if start_date is None or end_date is None:
        s, e = _default_date_range()
        start_date = start_date or s
        end_date = end_date or e
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    return _normalize(df, symbol)


def get_hk_stock_history(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """获取港股每日历史数据。"""
    if start_date is None or end_date is None:
        s, e = _default_date_range()
        start_date = start_date or s
        end_date = end_date or e
    df = ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    return _normalize(df, symbol)


# ===== 备用接口：新浪财经 =====

def _sina_normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """新浪接口返回的数据标准化。"""
    df = df.rename(columns=COLUMN_MAP_SINA)
    # 新浪接口字段已经是英文(date/open/high/low/close/volume/amount)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "symbol" not in df.columns:
        df.insert(0, "symbol", symbol)
    # 删除内部辅助字段
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    full_columns = [
        "date", "symbol", "open", "high", "low", "close",
        "volume", "amount", "amplitude", "change_pct", "change",
        "turnover_rate", "market_cap",
    ]
    for col in full_columns:
        if col not in df.columns:
            df[col] = None
    df = df[full_columns]
    return df.reset_index(drop=True)


def get_us_stock_history_sina(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """美股备用接口：新浪财经。symbol 格式如 105.AAPL，只取 ticker。"""
    ticker = symbol.split(".")[-1]
    df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
    # 新浪不支持 start/end 参数，手动过滤
    if start_date:
        sd = pd.to_datetime(start_date, format="%Y%m%d")
        df = df[pd.to_datetime(df["date"]) >= sd]
    if end_date:
        ed = pd.to_datetime(end_date, format="%Y%m%d")
        df = df[pd.to_datetime(df["date"]) <= ed]
    return _sina_normalize(df, ticker)


def get_a_stock_history_sina(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """A股备用接口：新浪财经。symbol 格式如 600519。"""
    prefix = "sh" if symbol.startswith("6") or symbol.startswith("9") else "sz"
    sina_symbol = f"{prefix}{symbol}"
    kwargs = {"symbol": sina_symbol, "adjust": "qfq"}
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        kwargs["end_date"] = end_date
    df = ak.stock_zh_a_daily(**kwargs)
    return _sina_normalize(df, symbol)


def get_hk_stock_history_sina(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """港股备用接口：新浪财经。symbol 格式如 00700。"""
    df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
    # 新浪不支持 start/end 参数，手动过滤
    if start_date:
        sd = pd.to_datetime(start_date, format="%Y%m%d")
        df = df[pd.to_datetime(df["date"]) >= sd]
    if end_date:
        ed = pd.to_datetime(end_date, format="%Y%m%d")
        df = df[pd.to_datetime(df["date"]) <= ed]
    return _sina_normalize(df, symbol)


def get_btc_history(days: int = 365) -> pd.DataFrame:
    """通过 CoinGecko 获取 BTC/USD 每日历史数据。"""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    prices = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
    caps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])

    df = prices.merge(volumes, on="timestamp").merge(caps, on="timestamp")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset="date", keep="first")
    df = df[["date", "close", "volume", "market_cap"]]
    return _normalize(df, "BTC")


def load_local_data(filename: str) -> pd.DataFrame:
    """从本地 JSON 文件加载已有数据。"""
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        df = pd.read_json(filename)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception:
        return pd.DataFrame()


def _last_trading_date(market: str) -> datetime:
    """获取最近一个交易日的日期。

    crypto 市场 7x24 交易，最近交易日就是今天（或昨天，若当前时间还很早）。
    其他市场只在周一至周五开盘（不考虑各国节假日，仅排除周末）。
    """
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if market == "crypto":
        return today

    # 股票市场：往回找最近的工作日
    d = today
    while d.weekday() >= 5:  # 5=周六, 6=周日
        d -= timedelta(days=1)
    return d


def get_missing_date_range(local_df: pd.DataFrame, market: str = ""):
    """根据本地数据计算需要补充的日期范围。

    Args:
        market: 市场类型，用于判断交易日

    Returns:
        (start_date, end_date) 格式 YYYYMMDD，如果无需更新返回 (None, None)
    """
    start_52w, end_today = _default_date_range()
    end_today_dt = datetime.strptime(end_today, "%Y%m%d")

    if local_df.empty:
        return start_52w, end_today

    local_dates = pd.to_datetime(local_df["date"])
    local_max = local_dates.max()
    local_min = local_dates.min()
    start_52w_dt = datetime.strptime(start_52w, "%Y%m%d")

    # 判断本地最新日期是否已覆盖到最近一个交易日
    last_td = _last_trading_date(market)
    if local_max.date() >= last_td.date():
        return None, None

    next_day = local_max + timedelta(days=1)
    if next_day.date() > end_today_dt.date():
        return None, None

    new_start = next_day.strftime("%Y%m%d")

    # 如果本地数据起始日期晚于52周起始，也需要补前面的数据
    if local_min.date() > start_52w_dt.date():
        # 需要补前面的，直接全量查
        return start_52w, end_today

    return new_start, end_today


def merge_data(local_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """合并本地数据和新数据，按日期去重并排序。"""
    if local_df.empty:
        return new_df
    if new_df.empty:
        return local_df
    merged = pd.concat([local_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset="date", keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def save_to_json(df: pd.DataFrame, filename: str):
    df.to_json(filename, orient="records", force_ascii=False, indent=2)


def fetch_and_save(name: str, fetch_fns: list, filename: str, market: str = "", **kwargs) -> dict:
    """通用的获取并保存逻辑，支持增量更新，支持多数据源 fallback。

    Returns:
        dict with keys: name, status ("updated"|"skipped"|"failed"), detail (str)
    """
    local_df = load_local_data(filename)
    is_btc = any(fn == get_btc_history for fn, _ in fetch_fns)

    fetch_kwargs = dict(kwargs)

    if not local_df.empty:
        start, end = get_missing_date_range(local_df, market=market)
        if start is None:
            return {"name": name, "status": "skipped", "detail": f"已是最新（{local_df['date'].iloc[-1]}）"}

        local_max = local_df["date"].max()

        if not is_btc:
            fetch_kwargs["start_date"] = start
            fetch_kwargs["end_date"] = end
        if is_btc:
            days_diff = (datetime.now() - datetime.strptime(local_max, "%Y-%m-%d")).days
            if days_diff <= 0:
                return {"name": name, "status": "skipped", "detail": f"已是最新（{local_max}）"}
            fetch_kwargs["days"] = days_diff + 1

    errors = []
    for i, (fetch_fn, source_name) in enumerate(fetch_fns):
        try:
            new_df = fetch_fn(**fetch_kwargs)
            merged_df = merge_data(local_df, new_df)

            if not merged_df.empty:
                save_to_json(merged_df, filename)
                new_count = len(new_df)
                total = len(merged_df)
                date_range = f"{merged_df['date'].iloc[0]} ~ {merged_df['date'].iloc[-1]}"
                return {"name": name, "status": "updated", "detail": f"+{new_count}条，共{total}条（{date_range}）"}
            else:
                return {"name": name, "status": "failed", "detail": "未获取到数据"}
        except Exception as e:
            errors.append(f"{source_name}: {e}")

    return {"name": name, "status": "failed", "detail": "；".join(errors)}


# 市场类型 -> 数据源列表（按优先级排列）
MARKET_FETCH_MAP = {
    "us": lambda item: [
        (get_us_stock_history_sina,  "新浪财经", {"symbol": item["symbol"]}),
        (get_us_stock_history,      "东方财富", {"symbol": item["symbol"]}),
    ],
    "a": lambda item: [
        (get_a_stock_history_sina,   "新浪财经", {"symbol": item["symbol"]}),
        (get_a_stock_history,       "东方财富", {"symbol": item["symbol"]}),
    ],
    "hk": lambda item: [
        (get_hk_stock_history_sina,  "新浪财经", {"symbol": item["symbol"]}),
        (get_hk_stock_history,      "东方财富", {"symbol": item["symbol"]}),
    ],
    "crypto": lambda item: [
        (get_btc_history,           "CoinGecko", {"days": 1095}),
    ],
}


def load_config():
    """从 config.json 加载股票配置列表。"""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["stocks"]


def get_output_filename(item: dict) -> str:
    """根据配置项生成输出文件名（存放在 data 目录下）。"""
    symbol = item["symbol"]
    if item["market"] == "us":
        name = f"{symbol.split('.')[-1]}.json"
    else:
        name = f"{symbol}.json"
    return os.path.join(DATA_DIR, name)


def run_update() -> str:
    """执行数据更新，返回汇总字符串。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    config = load_config()

    updated = []
    skipped = []
    failed = []

    for item in config:
        name = item["name"]
        market = item["market"]
        filename = get_output_filename(item)

        if market not in MARKET_FETCH_MAP:
            failed.append({"name": name, "status": "failed", "detail": f"未知market类型: {market}"})
            continue

        sources = MARKET_FETCH_MAP[market](item)
        fetch_fns = [(fn, sname) for fn, sname, _ in sources]
        base_kwargs = sources[0][2]
        result = fetch_and_save(name, fetch_fns, filename, market=market, **base_kwargs)

        if result["status"] == "updated":
            updated.append(result)
        elif result["status"] == "skipped":
            skipped.append(result)
        else:
            failed.append(result)

    lines = []
    lines.append(f"数据更新完成：{len(updated)}个已更新，{len(skipped)}个无需更新，{len(failed)}个失败")

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
