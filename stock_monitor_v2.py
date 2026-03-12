import pandas as pd
import numpy as np
import json
import os
import io
from datetime import datetime, timedelta
CONFIG_FILE = "config.json"
BLOCK_FILE = "block.json"
DATA_DIR = "newdata"
PE_DATA_DIR = "newdata"  # PE 直接内嵌在 newdata/ 的每条记录中

# 日期模糊匹配允许的最大偏移天数（查询 PE 时使用）
PE_DATE_TOLERANCE_DAYS = 3

# ── 触发条件：回撤 + 至少一个技术指标确认 ──
# (最大ratio, 日线RSI上限, 周线RSI上限, emoji, 预警名)
ALERT_LEVELS = [
    (0.30, 40, 40, "🔴", "红色预警"),
    (0.50, 35, 35, "🟠", "橙色预警"),
    (0.70, 30, 30, "🟡", "黄色预警"),
]


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["stocks"]


def load_block_symbols() -> set:
    """加载 block.json 中的屏蔽股票 symbol 集合。"""
    if not os.path.exists(BLOCK_FILE):
        return set()
    try:
        with open(BLOCK_FILE, "r", encoding="utf-8") as f:
            return {s["symbol"] for s in json.load(f).get("stocks", [])}
    except (json.JSONDecodeError, KeyError):
        return set()


def get_data_path(item: dict) -> str:
    symbol = item["symbol"]
    if item["market"] == "us":
        name = f"{symbol.split('.')[-1]}.json"
    else:
        name = f"{symbol}.json"
    return os.path.join(DATA_DIR, name)


def get_pe_path(item: dict) -> str:
    """获取数据文件路径（PE 已内嵌在 newdata/ 的每条记录中）。"""
    symbol = item["symbol"]
    if item["market"] == "us":
        name = f"{symbol.split('.')[-1]}.json"
    else:
        name = f"{symbol}.json"
    return os.path.join(DATA_DIR, name)


def _load_pe_data(pe_path: str) -> dict[str, float]:
    """从 newdata/ 文件中读取 {date: pe} 字典（PE 直接内嵌在每条记录）。"""
    if not os.path.exists(pe_path):
        return {}
    try:
        with open(pe_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        # newdata 格式：每条记录含 pe 字段
        return {r["date"]: r["pe"] for r in records if "date" in r and r.get("pe") is not None}
    except Exception:
        return {}


def lookup_pe(pe_map: dict[str, float], date: str) -> float | None:
    """从 PE 字典中查询指定日期的 PE 值，支持 ±3 天模糊匹配。"""
    if not date or not pe_map:
        return None

    # 精确匹配
    if date in pe_map:
        val = pe_map[date]
        return round(float(val), 2) if pd.notna(val) else None

    # 模糊匹配：前后 PE_DATE_TOLERANCE_DAYS 天内最近的日期
    target = datetime.strptime(date, "%Y-%m-%d")
    best_val = None
    best_diff = timedelta(days=PE_DATE_TOLERANCE_DAYS + 1)

    for d_str, pe_val in pe_map.items():
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
            diff = abs(d - target)
            if diff <= timedelta(days=PE_DATE_TOLERANCE_DAYS) and diff < best_diff:
                best_diff = diff
                best_val = pe_val
        except (ValueError, TypeError):
            continue

    if best_val is not None and pd.notna(best_val):
        return round(float(best_val), 2)
    return None


# ──────────────────────────────────────
#  技术指标计算
# ──────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> float | None:
    """计算RSI。"""
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(val, 1) if not pd.isna(val) else None


def calc_weekly_rsi(df: pd.DataFrame, period: int = 14) -> float | None:
    """计算周线RSI。"""
    df_w = df.set_index("date").resample("W-FRI").agg({"close": "last"}).dropna()
    if len(df_w) < period + 1:
        return None
    return calc_rsi(df_w["close"], period)


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算MACD，返回 (dif, dea, macd_hist) 的最新值。"""
    if len(series) < slow + signal:
        return None, None, None
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return (
        round(dif.iloc[-1], 4),
        round(dea.iloc[-1], 4),
        round(hist.iloc[-1], 4),
    )


def detect_macd_divergence(df: pd.DataFrame, lookback: int = 60) -> bool:
    """检测MACD底背离：价格创近期新低但MACD柱状图没有创新低。
    在最近 lookback 根K线中寻找底背离。"""
    close = df["close"].values
    if len(close) < 35 + lookback:
        data = df.copy()
    else:
        data = df.iloc[-(35 + lookback):].copy()

    s = data["close"].reset_index(drop=True)
    if len(s) < 35:
        return False

    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2

    prices = s.values
    hists = hist.values

    # 在最近 lookback 范围内找两个低点
    scan = min(lookback, len(prices) - 1)
    recent_prices = prices[-scan:]
    recent_hists = hists[-scan:]

    # 找近期价格的最低点
    min_idx_recent = np.argmin(recent_prices[-20:])  # 最近20根
    min_idx_earlier = np.argmin(recent_prices[:-20]) if scan > 20 else None

    if min_idx_earlier is None:
        return False

    price_recent_low = recent_prices[-20:][min_idx_recent]
    price_earlier_low = recent_prices[:-20][min_idx_earlier]

    hist_recent_low = recent_hists[-20:][min_idx_recent]
    hist_earlier_low = recent_hists[:-20][min_idx_earlier]

    # 底背离：价格新低更低，但MACD柱状图的低点更高
    if price_recent_low <= price_earlier_low and hist_recent_low > hist_earlier_low:
        return True

    return False


def calc_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """计算布林带，返回 (当前价位置百分比, 是否跌破下轨)。
    位置百分比: 0%=下轨, 50%=中轨, 100%=上轨, <0%=跌破下轨。"""
    if len(series) < period:
        return None, False
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std

    cur = series.iloc[-1]
    up = upper.iloc[-1]
    lo = lower.iloc[-1]

    if pd.isna(up) or pd.isna(lo) or up == lo:
        return None, False

    pct = round((cur - lo) / (up - lo) * 100, 1)
    below = cur < lo
    return pct, below


def calc_bias(series: pd.Series, period: int = 120) -> float | None:
    """计算均线偏离率 BIAS = (当前价 - MA) / MA * 100。"""
    if len(series) < period:
        return None
    ma = series.rolling(window=period).mean().iloc[-1]
    if pd.isna(ma) or ma <= 0:
        return None
    return round((series.iloc[-1] - ma) / ma * 100, 1)


def calc_volume_ratio(df: pd.DataFrame, days: int = 60) -> float | None:
    """计算最新成交量相对于N日均量的百分比。"""
    if "volume" not in df.columns or len(df) < 2:
        return None
    vol = df["volume"].dropna()
    if len(vol) < 2:
        return None
    avg_vol = vol.iloc[-days - 1:-1].mean() if len(vol) > days else vol.iloc[:-1].mean()
    if pd.isna(avg_vol) or avg_vol <= 0:
        return None
    return round(vol.iloc[-1] / avg_vol * 100, 1)


def get_currency_symbol(market: str) -> str:
    if market == "us":
        return "$"
    elif market == "a":
        return "¥"
    elif market == "hk":
        return "HK$"
    elif market == "crypto":
        return "$"
    return ""






# 东财交易所编号 -> Google Finance 交易所代码
_US_EXCHANGE_MAP = {
    "105": "NASDAQ",
    "106": "NYSE",
    "107": "NYSEAMERICAN",
}


def get_google_finance_url(item: dict) -> str:
    """生成 Google Finance 链接。"""
    market = item["market"]
    symbol = item["symbol"]

    if market == "us":
        parts = symbol.split(".")
        exchange_code = _US_EXCHANGE_MAP.get(parts[0], "NASDAQ")
        ticker = parts[-1]
        return f"https://www.google.com/finance/quote/{ticker}:{exchange_code}?window=5Y"
    elif market == "hk":
        # 去掉前导0，如 03690 -> 3690
        ticker = str(int(symbol))
        return f"https://www.google.com/finance/quote/{ticker}:HKG?window=5Y"
    elif market == "a":
        # 沪市6开头 -> SHA，深市0/3开头 -> SHE
        if symbol.startswith("6"):
            exchange = "SHA"
        else:
            exchange = "SHE"
        return f"https://www.google.com/finance/quote/{symbol}:{exchange}?window=5Y"
    elif market == "crypto":
        return f"https://www.google.com/finance/quote/{symbol}-USD?window=5Y"
    return ""


def make_link(name: str, item_or_info: dict) -> str:
    """生成 Markdown 超链接格式的股票名。"""
    url = get_google_finance_url(item_or_info)
    if url:
        return f"[{name}]({url})"
    return name


# ──────────────────────────────────────
#  阶段性高点识别
# ──────────────────────────────────────

# 有效反弹阈值（保留供其他函数使用）
SIGNIFICANT_REBOUND = 0.20

# 显著低点的最小涨幅阈值：从低点到高点涨幅需达到此比例才算"显著低点"
MIN_RISE_FOR_TROUGH = 0.30


def find_cycle_high(df: pd.DataFrame) -> dict:
    """找到当前下跌周期的显著高点和前一轮低谷。

    算法（简化版）：
    1. 在全部数据中找到全局最高点作为"显著高点"
    2. 从全局最高点往前（更早时间）扫描，找到最低价作为候选低谷
    3. 验证：从低谷到高点的涨幅需 >= MIN_RISE_FOR_TROUGH，否则低谷无效

    Returns:
        dict with keys:
        - peak_price: 本轮下跌的起始高点价格
        - peak_date: 高点日期
        - trough_price: 前一次低谷价格（可能为 None）
        - trough_date: 低谷日期（可能为 None）
    """
    highs = df["high"].values if "high" in df.columns else df["close"].values
    lows = df["low"].values if "low" in df.columns else df["close"].values
    dates = df["date"].values

    if len(highs) == 0:
        return {"peak_price": None, "peak_date": None,
                "trough_price": None, "trough_date": None}

    # 1. 找全局最高点 → 显著高点
    peak_idx = int(np.argmax(highs))
    peak_price = float(highs[peak_idx])

    # 2. 从高点往前找最低点 → 候选显著低点
    prev_trough_price = None
    prev_trough_date = None

    if peak_idx > 0:
        segment_lows = lows[:peak_idx]
        min_idx = int(np.argmin(segment_lows))
        candidate_trough = float(segment_lows[min_idx])

        # 3. 验证涨幅是否足够大
        if candidate_trough > 0:
            rise = (peak_price - candidate_trough) / candidate_trough
            if rise >= MIN_RISE_FOR_TROUGH:
                prev_trough_price = candidate_trough
                prev_trough_date = pd.Timestamp(dates[min_idx]).strftime("%Y-%m-%d")

    peak_date = pd.Timestamp(dates[peak_idx]).strftime("%Y-%m-%d")

    return {
        "peak_price": peak_price,
        "peak_date": peak_date,
        "trough_price": prev_trough_price,
        "trough_date": prev_trough_date,
    }


# ──────────────────────────────────────
#  分析主逻辑
# ──────────────────────────────────────

def analyze(item: dict) -> dict | None:
    path = get_data_path(item)
    if not os.path.exists(path):
        return None

    df = pd.read_json(path)
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 技术指标仍用52周数据（保持时效性）
    cutoff_52w = datetime.now() - timedelta(weeks=52)
    df_52w = df[df["date"] >= cutoff_52w].copy()
    if df_52w.empty:
        return None

    latest = df_52w.iloc[-1]
    current_price = latest["close"]
    latest_date = latest["date"].strftime("%Y-%m-%d")
    if pd.isna(current_price) or current_price <= 0:
        return None

    # 用全量数据（最多3年）找阶段性高点
    cutoff_full = datetime.now() - timedelta(weeks=156)
    df_full = df[df["date"] >= cutoff_full].copy()
    if df_full.empty:
        df_full = df_52w

    cycle_info = find_cycle_high(df_full)
    stage_high = cycle_info["peak_price"]

    if stage_high is None or stage_high <= 0:
        # 退化为52周高点
        stage_high = df_52w["high"].max() if "high" in df_52w.columns else df_52w["close"].max()
    if pd.isna(stage_high) or stage_high <= 0:
        return None

    # ── 从本地 PE 数据文件获取 PE 值 ──
    pe_path = get_pe_path(item)
    pe_map = _load_pe_data(pe_path)
    peak_pe = lookup_pe(pe_map, cycle_info["peak_date"])
    trough_pe = lookup_pe(pe_map, cycle_info["trough_date"])
    current_pe = lookup_pe(pe_map, latest_date)

    ratio = current_price / stage_high
    drop_pct = round((1 - ratio) * 100, 1)

    # ── 计算所有技术指标（基于52周数据） ──
    rsi_14 = calc_rsi(df_52w["close"], 14)
    weekly_rsi = calc_weekly_rsi(df_52w, 14)
    dif, dea, macd_hist = calc_macd(df_52w["close"])
    macd_divergence = detect_macd_divergence(df_52w)
    boll_pct, boll_below = calc_bollinger(df_52w["close"], 20, 2.0)
    bias_120 = calc_bias(df_52w["close"], 120)
    vol_ratio = calc_volume_ratio(df_52w, 60)

    # ── 触发判定：回撤阈值 + RSI超卖 ──
    triggered_level = None
    for max_ratio, rsi_cap, w_rsi_cap, emoji, level_name in ALERT_LEVELS:
        if ratio > max_ratio:
            continue
        rsi_ok = (rsi_14 is not None and rsi_14 <= rsi_cap)
        w_rsi_ok = (weekly_rsi is not None and weekly_rsi <= w_rsi_cap)
        if rsi_ok or w_rsi_ok:
            triggered_level = (max_ratio, rsi_cap, w_rsi_cap, emoji, level_name)
            break
    if triggered_level is None:
        return None

    max_ratio_t, rsi_cap_t, w_rsi_cap_t, emoji, level_name = triggered_level

    # ── 构造触发原因 ──
    reason_parts = [f"回撤{drop_pct}%"]
    if rsi_14 is not None and rsi_14 <= rsi_cap_t:
        reason_parts.append(f"日线RSI({rsi_14})<{rsi_cap_t}")
    if weekly_rsi is not None and weekly_rsi <= w_rsi_cap_t:
        reason_parts.append(f"周线RSI({weekly_rsi})<{w_rsi_cap_t}")
    if macd_divergence:
        reason_parts.append("MACD底背离")
    if boll_below:
        reason_parts.append("跌破布林带下轨")
    reason = " + ".join(reason_parts)

    # ──────────────────────────────────────
    #  四维综合评分（满分100）
    # ──────────────────────────────────────

    # 1) 回撤深度 (0-30分)
    if drop_pct >= 80:
        score_drop = 30
    elif drop_pct >= 30:
        score_drop = 6 + (drop_pct - 30) / 50 * 24
    else:
        score_drop = 0

    # 2) RSI超卖 (0-25分) = 日线(15) + 周线(10)
    score_rsi = 0
    if rsi_14 is not None:
        if rsi_14 <= 10:
            score_rsi += 15
        elif rsi_14 <= 50:
            score_rsi += (50 - rsi_14) / 40 * 15
    if weekly_rsi is not None:
        if weekly_rsi <= 10:
            score_rsi += 10
        elif weekly_rsi <= 50:
            score_rsi += (50 - weekly_rsi) / 40 * 10

    # 3) MACD底背离 (0-25分)：有底背离=25分，无底背离但MACD柱缩短也给部分分
    score_macd = 0
    if macd_divergence:
        score_macd = 25
    elif macd_hist is not None and dif is not None:
        # MACD柱状图为负但在收窄（DIF > DEA趋势接近），给部分分
        if macd_hist < 0 and dif > dea:
            score_macd = 12
        elif macd_hist < 0:
            score_macd = 5

    # 4) 布林带 + 均线偏离 (0-20分) = 布林带(12) + BIAS(8)
    score_boll = 0
    if boll_pct is not None:
        if boll_below:
            score_boll += 12
        elif boll_pct <= 5:
            score_boll += 10
        elif boll_pct <= 15:
            score_boll += 7
        elif boll_pct <= 25:
            score_boll += 3

    score_bias = 0
    if bias_120 is not None:
        if bias_120 <= -30:
            score_bias = 8
        elif bias_120 <= -20:
            score_bias = 6
        elif bias_120 <= -10:
            score_bias = 4
        elif bias_120 <= -5:
            score_bias = 2

    score_position = score_boll + score_bias

    # 成交量加分 (bonus 0-5)
    score_vol = 0
    if vol_ratio is not None:
        if vol_ratio >= 200:
            score_vol = 5
        elif vol_ratio >= 150:
            score_vol = 3
        elif vol_ratio >= 120:
            score_vol = 2

    raw_score = score_drop + score_rsi + score_macd + score_position + score_vol
    score = min(round(raw_score), 100)

    # ── 操作建议 ──
    if score >= 80:
        advice_emoji = "💡 💎"
        advice = "强底部信号，可分批建仓（20-30%）"
    elif score >= 60:
        advice_emoji = "💡 📊"
        advice = "重点关注，可小仓位介入（10-20%）"
    elif score >= 40:
        advice_emoji = "💡 📋"
        advice = "进入观察，等待进一步确认"
    else:
        advice_emoji = "💡 ⏳"
        advice = "信号较弱，继续等待"

    return {
        "name": item["name"],
        "market": item["market"],
        "symbol": item["symbol"],
        "latest_date": latest_date,
        "current_price": round(current_price, 2),
        "high_52w": round(stage_high, 2),
        "peak_date": cycle_info["peak_date"],
        "trough_price": round(cycle_info["trough_price"], 2) if cycle_info["trough_price"] is not None else None,
        "trough_date": cycle_info["trough_date"],
        "peak_pe": peak_pe,
        "trough_pe": trough_pe,
        "current_pe": current_pe,
        "ratio": round(ratio, 4),
        "drop_pct": drop_pct,
        "emoji": emoji,
        "level_name": level_name,
        "reason": reason,
        "rsi_14": rsi_14,
        "weekly_rsi": weekly_rsi,
        "dif": dif,
        "dea": dea,
        "macd_hist": macd_hist,
        "macd_divergence": macd_divergence,
        "boll_pct": boll_pct,
        "boll_below": boll_below,
        "bias_120": bias_120,
        "vol_ratio": vol_ratio,
        "score": score,
        "score_detail": {
            "回撤": round(score_drop, 1),
            "RSI": round(score_rsi, 1),
            "MACD": round(score_macd, 1),
            "布林/偏离": round(score_position, 1),
            "成交量": round(score_vol, 1),
        },
        "advice_emoji": advice_emoji,
        "advice": advice,
    }


# ──────────────────────────────────────
#  卡片格式化
# ──────────────────────────────────────

def format_card(s: dict) -> str:
    buf = io.StringIO()
    currency = get_currency_symbol(s["market"])

    name_link = make_link(s["name"], s)
    buf.write(f"{s['emoji']} {s['level_name']} {name_link} 触发底部信号\n")

    # 触发原因
    buf.write(f"🧿 触发原因：\n")
    buf.write(f"  · {s['reason']}\n")

    # 价格信息
    buf.write(f"📉 价格信息：\n")
    current_pe_str = f" | PE {s['current_pe']}" if s.get('current_pe') is not None else ""
    buf.write(f"  · 当前价：{currency}{s['current_price']}{current_pe_str}\n")
    peak_date_str = f"（{s['peak_date']}）" if s.get('peak_date') else ""
    peak_pe_str = f" | PE {s['peak_pe']}" if s.get('peak_pe') is not None else ""
    buf.write(f"  · 本轮高点：{currency}{s['high_52w']}{peak_date_str}{peak_pe_str}\n")
    if s.get('trough_price') is not None:
        trough_date_str = f"（{s['trough_date']}）" if s.get('trough_date') else ""
        trough_pe_str = f" | PE {s['trough_pe']}" if s.get('trough_pe') is not None else ""
        buf.write(f"  · 前次低谷：{currency}{s['trough_price']}{trough_date_str}{trough_pe_str}\n")
    buf.write(f"  · 回撤幅度：{s['drop_pct']}%\n")

    # 技术指标
    buf.write(f"📊 技术指标：\n")
    rsi_str = f"{s['rsi_14']}" if s['rsi_14'] is not None else "N/A"
    w_rsi_str = f"{s['weekly_rsi']}" if s['weekly_rsi'] is not None else "N/A"
    buf.write(f"  · RSI(14)：{rsi_str} | 周线RSI：{w_rsi_str}\n")

    # MACD
    macd_status = "底背离 ✅" if s['macd_divergence'] else "无背离"
    if s['dif'] is not None:
        buf.write(f"  · MACD：DIF {s['dif']} | DEA {s['dea']} | {macd_status}\n")
    else:
        buf.write(f"  · MACD：N/A\n")

    # 布林带
    if s['boll_pct'] is not None:
        boll_status = "⚠️ 跌破下轨" if s['boll_below'] else f"位于{s['boll_pct']}%位置"
        buf.write(f"  · 布林带：{boll_status}\n")

    # 均线偏离
    if s['bias_120'] is not None:
        buf.write(f"  · 120日均线偏离：{s['bias_120']}%\n")

    # 成交量
    vol_str = f"{s['vol_ratio']}% (相对60日均量)" if s['vol_ratio'] is not None else "N/A"
    buf.write(f"  · 成交量：{vol_str}\n")

    # 评分明细
    d = s["score_detail"]
    buf.write(f"💯 综合评分：{s['score']}/100\n")
    buf.write(f"  · 回撤{d['回撤']} + RSI{d['RSI']} + MACD{d['MACD']} + 布林/偏离{d['布林/偏离']} + 量能{d['成交量']}\n")

    # 操作建议
    buf.write(f"{s['advice_emoji']} {s['advice']}\n")

    # 时间戳
    buf.write(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return buf.getvalue()


def generate_report() -> str:
    config = load_config()
    blocked = load_block_symbols()
    results = []
    all_drops = []  # 所有股票的回撤数据

    for item in config:
        if item["symbol"] in blocked:
            continue
        # 计算基础回撤数据
        drop_info = calc_drop_info(item)
        if drop_info:
            all_drops.append(drop_info)
        # 完整分析（触发阈值的）
        r = analyze(item)
        if r:
            results.append(r)

    buf = io.StringIO()

    # 触发阈值的卡片
    if results:
        results.sort(key=lambda x: x["ratio"])
        cards = [format_card(s) for s in results]
        buf.write(("\n" + "─" * 40 + "\n\n").join(cards))

    # 回撤超过25%的表格
    drop_40 = [d for d in all_drops if d["drop_pct"] >= 25]
    if drop_40:
        drop_40.sort(key=lambda x: -x["drop_pct"])
        if results:
            buf.write("\n\n")
        buf.write("回撤超过25%的股票一览\n\n")
        # 计算名字列的显示宽度（中文算2，英文算1）
        def display_width(s: str) -> int:
            w = 0
            for c in s:
                w += 2 if '\u4e00' <= c <= '\u9fff' or '\uff00' <= c <= '\uffef' or '\u3000' <= c <= '\u303f' else 1
            return w

        col_w = 22  # 名字列目标显示宽度
        header_pad = " " * (col_w - display_width("股票"))
        buf.write(f"  股票{header_pad} {'当前价':>10s} {'阶段高点':>10s} {'回撤':>8s}\n")
        buf.write(f"  {'─' * (col_w // 2)}  {'─' * 10} {'─' * 10} {'─' * 8}\n")
        for d in drop_40:
            cur = get_currency_symbol(d["market"])
            name_link = make_link(d["name"], d)
            # 补齐：目标宽度 - 显示名宽度 = 需要额外补的空格（链接部分不显示）
            name_dw = display_width(d["name"])
            pad = " " * (col_w - name_dw + len(name_link) - len(name_link))
            # 链接字符串比显示名长，需要减少format的宽度
            extra = len(name_link) - name_dw
            total_pad = col_w + extra
            buf.write(f"  {name_link:<{total_pad}s} {cur}{d['current_price']:>8.2f} {cur}{d['high_52w']:>8.2f} {d['drop_pct']:>7.1f}%\n")

    if not results and not drop_40:
        buf.write(f"✅ 当前没有股票触发预警，也没有回撤超过25%的股票。\n")
        buf.write(f"   监控的股票数量：{len(config)} | 有回撤数据的：{len(all_drops)}\n")
        if all_drops:
            max_drop = max(all_drops, key=lambda x: x["drop_pct"])
            buf.write(f"   当前最大回撤：{max_drop['name']} {max_drop['drop_pct']}%\n")

    return buf.getvalue()


def calc_drop_info(item: dict) -> dict | None:
    """计算单只股票的阶段性回撤基础数据。"""
    path = get_data_path(item)
    if not os.path.exists(path):
        return None

    df = pd.read_json(path)
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 用全量数据（最多3年）找阶段性高点
    cutoff_full = datetime.now() - timedelta(weeks=156)
    df_full = df[df["date"] >= cutoff_full].copy()

    cutoff_52w = datetime.now() - timedelta(weeks=52)
    df_52w = df[df["date"] >= cutoff_52w]
    if df_52w.empty:
        return None

    if df_full.empty:
        df_full = df_52w

    current_price = df_52w.iloc[-1]["close"]
    if pd.isna(current_price) or current_price <= 0:
        return None

    cycle_info = find_cycle_high(df_full)
    stage_high = cycle_info["peak_price"]

    if stage_high is None or stage_high <= 0:
        stage_high = df_52w["high"].max() if "high" in df_52w.columns else df_52w["close"].max()
    if pd.isna(stage_high) or stage_high <= 0:
        return None

    drop_pct = round((1 - current_price / stage_high) * 100, 1)

    return {
        "name": item["name"],
        "market": item["market"],
        "symbol": item["symbol"],
        "current_price": round(current_price, 2),
        "high_52w": round(stage_high, 2),
        "peak_date": cycle_info["peak_date"],
        "trough_price": round(cycle_info["trough_price"], 2) if cycle_info["trough_price"] is not None else None,
        "trough_date": cycle_info["trough_date"],
        "drop_pct": drop_pct,
    }


def main() -> str:
    """生成报告，打印并返回。"""
    report: str = generate_report()
    print(report, end="")
    return report


if __name__ == "__main__":
    main()
