"""resilience_screener.py — 韧性筛选脚本 v2

扫描 config.json 中所有股票的 newdata/ 历史日线数据，
检测 "高峰 → 腰斩(≥50%) → 恢复(至前高70%)" 的 V/U 型周期，
输出结构化 JSON 数据（含 OHLC K线数据）到 resilience_report/ 目录。
HTML 页面使用 ECharts 渲染交互式 K 线图。

运行方式：
  python3 resilience_screener.py
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ══════════════════════════════════════════
#  配置参数
# ══════════════════════════════════════════

CONFIG_FILE = "config.json"
NEW_DATA_DIR = "newdata"
OUTPUT_DIR = "resilience_report"

# 筛选参数
MIN_DROP_PCT = 0.50          # 最小跌幅 50%
MIN_RECOVERY_PCT = 0.70      # 恢复到前高的 70%
LOOKBACK_START = "2019-01-01" # 回溯起点（近7年）
MIN_DATA_POINTS = 250        # 最少数据点（约1年）
MA_WINDOW = 5                # 移动均线窗口
PEAK_WINDOW = 60             # 峰值检测窗口（约3个月）
MIN_DROP_DAYS = 60           # 最小下跌持续交易日
MIN_RECOVERY_DAYS = 60       # 最小恢复持续交易日

# K线数据采样 —— 限制每只股票最多输出的数据点数（避免 JSON 过大）
MAX_KLINE_POINTS = 1500      # 约6年日线


# ══════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════

def _base_name(item: dict) -> str:
    """复用 stock_history_v2.py 的文件名映射逻辑"""
    symbol = item["symbol"]
    if item["market"] == "us":
        return f"{symbol.split('.')[-1]}.json"
    return f"{symbol}.json"


def load_config() -> list[dict]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["stocks"]


def load_stock_data(item: dict) -> pd.DataFrame | None:
    """加载单只股票的历史数据，过滤并截取近7年"""
    path = os.path.join(NEW_DATA_DIR, _base_name(item))
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return None

    if not records:
        return None

    df = pd.DataFrame(records)

    # 确保必要列存在
    required_cols = ["date", "open", "high", "low", "close"]
    if not all(c in df.columns for c in required_cols):
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ["open", "high", "low", "close", "volume", "amount", "pe"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 过滤 close 为 None、NaN、<=0 的记录
    df = df[df["close"].notna() & (df["close"] > 0)]

    if df.empty:
        return None

    # 截取近7年数据
    cutoff = pd.Timestamp(LOOKBACK_START)
    df_recent = df[df["date"] >= cutoff].copy()

    # 如果近7年数据不足，使用全量
    if len(df_recent) < MIN_DATA_POINTS:
        df_recent = df.copy()

    # 仍然不足则放弃
    if len(df_recent) < MIN_DATA_POINTS:
        return None

    df_recent = df_recent.sort_values("date").reset_index(drop=True)
    return df_recent


# ══════════════════════════════════════════
#  核心算法：V型周期检测
# ══════════════════════════════════════════

def detect_peaks(smoothed: np.ndarray, window: int = PEAK_WINDOW) -> list[int]:
    """检测局部峰值：某点是前后各 window 个点中的最大值"""
    n = len(smoothed)
    peaks = []
    for i in range(window, n - window):
        left = smoothed[max(0, i - window):i]
        right = smoothed[i + 1:min(n, i + window + 1)]
        if smoothed[i] >= max(left) and smoothed[i] >= max(right):
            peaks.append(i)
    return peaks


def find_v_cycles(df: pd.DataFrame) -> list[dict]:
    """在数据中检测所有 V 型周期

    算法：
    1. 找到所有局部峰值
    2. 对每个峰值，在其之后找到全局最低谷（在价格恢复到峰值70%之前）
    3. 关键：必须先让价格跌破峰值的50%后，才开始检查恢复条件
    4. 从谷底之后找恢复到峰值70%的点
    """
    df = df.copy()
    df["smoothed"] = df["close"].rolling(window=MA_WINDOW, min_periods=1).mean()

    smoothed = df["smoothed"].values
    dates = df["date"].values
    closes = df["close"].values
    n = len(smoothed)

    # 检测所有局部峰值
    peak_indices = detect_peaks(smoothed)

    if not peak_indices:
        return []

    cycles = []
    search_start = 0

    for peak_idx in peak_indices:
        if peak_idx < search_start:
            continue

        peak_price = smoothed[peak_idx]
        drop_threshold = peak_price * (1 - MIN_DROP_PCT)  # 跌破此价格才算腰斩
        recovery_target = peak_price * MIN_RECOVERY_PCT     # 恢复到此价格算走出来

        # ── 阶段1：从峰值往后，找到跌破50%的谷底区域 ──
        trough_idx = None
        trough_price = float("inf")
        has_dropped_enough = False  # 是否曾跌破50%

        for j in range(peak_idx + 1, n):
            if smoothed[j] < trough_price:
                trough_price = smoothed[j]
                trough_idx = j

            # 检查是否已经跌破50%
            if trough_price <= drop_threshold:
                has_dropped_enough = True

            # 只有在已经跌破50%之后，价格又恢复到峰值70%以上，才中断搜索
            if has_dropped_enough and smoothed[j] >= recovery_target:
                break

        if not has_dropped_enough or trough_idx is None:
            continue

        # 检查下跌持续天数
        drop_days = trough_idx - peak_idx
        if drop_days < MIN_DROP_DAYS:
            continue

        drop_pct = (peak_price - trough_price) / peak_price

        # ── 阶段2：从谷底往后，找恢复点 ──
        recovery_idx = None
        for j in range(trough_idx + 1, n):
            if smoothed[j] >= recovery_target:
                recovery_idx = j
                break

        if recovery_idx is None:
            continue

        # 检查恢复持续天数
        recovery_days = recovery_idx - trough_idx
        if recovery_days < MIN_RECOVERY_DAYS:
            continue

        # 记录完整 V 型周期
        recovery_price = smoothed[recovery_idx]
        recovery_pct = recovery_price / peak_price

        cycle = {
            "peak_date": str(pd.Timestamp(dates[peak_idx]).date()),
            "peak_price": round(float(peak_price), 4),
            "trough_date": str(pd.Timestamp(dates[trough_idx]).date()),
            "trough_price": round(float(trough_price), 4),
            "recovery_date": str(pd.Timestamp(dates[recovery_idx]).date()),
            "recovery_price": round(float(recovery_price), 4),
            "drop_pct": round(float(drop_pct * 100), 1),
            "recovery_pct": round(float(recovery_pct * 100), 1),
            "drop_days": int(drop_days),
            "recovery_days": int(recovery_days),
            "_peak_idx": int(peak_idx),
            "_trough_idx": int(trough_idx),
            "_recovery_idx": int(recovery_idx),
        }
        cycles.append(cycle)

        # 从恢复点之后继续搜索下一个周期
        search_start = recovery_idx + 1

    return cycles


# ══════════════════════════════════════════
#  韧性评分
# ══════════════════════════════════════════

def calc_resilience_score(cycles: list[dict]) -> float:
    """计算韧性评分 (0-100)"""
    if not cycles:
        return 0.0

    # 周期数量评分 (权重30%)
    n = len(cycles)
    if n >= 3:
        cycle_score = 100
    elif n == 2:
        cycle_score = 80
    else:
        cycle_score = 50

    # 平均恢复比例评分 (权重40%)
    avg_recovery = np.mean([c["recovery_pct"] for c in cycles])
    # 70% -> 60分, 100% -> 100分, >100% 还是100分
    recovery_score = min(100, 60 + (avg_recovery - 70) / 30 * 40)
    recovery_score = max(0, recovery_score)

    # 恢复速度评分 (权重30%)
    avg_recovery_days = np.mean([c["recovery_days"] for c in cycles])
    # 60日 -> 100分, 300日 -> 60分
    if avg_recovery_days <= 60:
        speed_score = 100
    elif avg_recovery_days >= 300:
        speed_score = 60
    else:
        speed_score = 100 - (avg_recovery_days - 60) / 240 * 40

    score = cycle_score * 0.3 + recovery_score * 0.4 + speed_score * 0.3
    return round(score, 1)


# ══════════════════════════════════════════
#  提取 K 线数据 & 关键信息
# ══════════════════════════════════════════

def extract_kline_data(df: pd.DataFrame) -> list[list]:
    """提取 OHLC K线数据，格式：[[date, open, close, low, high, volume], ...]
    注意：ECharts candlestick 需要 [open, close, low, high] 的顺序
    """
    # 如果数据过长，做尾部截取
    if len(df) > MAX_KLINE_POINTS:
        df = df.tail(MAX_KLINE_POINTS).reset_index(drop=True)

    result = []
    for _, row in df.iterrows():
        date_str = str(pd.Timestamp(row["date"]).date())
        o = round(float(row["open"]), 4) if pd.notna(row.get("open")) else None
        c = round(float(row["close"]), 4) if pd.notna(row.get("close")) else None
        l = round(float(row["low"]), 4) if pd.notna(row.get("low")) else None
        h = round(float(row["high"]), 4) if pd.notna(row.get("high")) else None
        v = round(float(row["volume"]), 0) if pd.notna(row.get("volume")) else 0

        if o is not None and c is not None and l is not None and h is not None:
            result.append([date_str, o, c, l, h, v])

    return result


def extract_key_info(df: pd.DataFrame) -> dict:
    """提取当前关键信息"""
    if df.empty:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    close = float(latest["close"])
    prev_close = float(prev["close"])
    change = close - prev_close
    change_pct = (change / prev_close * 100) if prev_close != 0 else 0

    # 最近价格统计
    high_52w = float(df.tail(250)["high"].max()) if "high" in df.columns else None
    low_52w = float(df.tail(250)["low"].min()) if "low" in df.columns else None

    # 均线
    ma5 = round(float(df["close"].tail(5).mean()), 4)
    ma20 = round(float(df["close"].tail(20).mean()), 4) if len(df) >= 20 else None
    ma60 = round(float(df["close"].tail(60).mean()), 4) if len(df) >= 60 else None

    # 当前 PE
    pe = round(float(latest["pe"]), 2) if pd.notna(latest.get("pe")) and latest.get("pe") else None

    # 成交量
    vol = round(float(latest["volume"]), 0) if pd.notna(latest.get("volume")) else None
    amount = round(float(latest["amount"]), 0) if pd.notna(latest.get("amount")) else None

    # 距52周高点跌幅
    from_high = round((close / high_52w - 1) * 100, 1) if high_52w and high_52w > 0 else None
    # 距52周低点涨幅
    from_low = round((close / low_52w - 1) * 100, 1) if low_52w and low_52w > 0 else None

    info = {
        "latest_date": str(pd.Timestamp(latest["date"]).date()),
        "close": round(close, 4),
        "open": round(float(latest["open"]), 4) if pd.notna(latest.get("open")) else None,
        "high": round(float(latest["high"]), 4) if pd.notna(latest.get("high")) else None,
        "low": round(float(latest["low"]), 4) if pd.notna(latest.get("low")) else None,
        "change": round(change, 4),
        "change_pct": round(change_pct, 2),
        "volume": vol,
        "amount": amount,
        "pe": pe,
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "high_52w": round(high_52w, 4) if high_52w else None,
        "low_52w": round(low_52w, 4) if low_52w else None,
        "from_high_pct": from_high,
        "from_low_pct": from_low,
    }

    return info


# ══════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stocks = load_config()
    total = len(stocks)

    passed = []
    failed = []
    skipped = []

    for idx, item in enumerate(stocks, 1):
        name = item["name"]
        market = item["market"]
        symbol = item["symbol"]
        stock_type = item.get("stock_type", "unclassified")
        duplicate_of = item.get("duplicate_of", "")

        tag = f"[{idx}/{total}]"

        # 跳过 crypto
        if market == "crypto":
            skipped.append({
                "name": name, "symbol": symbol, "market": market,
                "stock_type": stock_type, "duplicate_of": duplicate_of,
                "reason": "crypto类型，不适用",
                "kline": [],
                "key_info": {},
            })
            print(f"  ⏭ {tag} {name}: 跳过(crypto)")
            continue

        # 加载数据
        df = load_stock_data(item)
        if df is None:
            skipped.append({
                "name": name, "symbol": symbol, "market": market,
                "stock_type": stock_type, "duplicate_of": duplicate_of,
                "reason": "数据不足或不存在",
                "kline": [],
                "key_info": {},
            })
            print(f"  ⏭ {tag} {name}: 跳过(数据不足)")
            continue

        # 提取 K线数据 和 关键信息（所有有数据的股票都提取）
        kline = extract_kline_data(df)
        key_info = extract_key_info(df)

        # 检测 V 型周期
        cycles = find_v_cycles(df)

        if not cycles:
            failed.append({
                "name": name, "symbol": symbol, "market": market,
                "stock_type": stock_type, "duplicate_of": duplicate_of,
                "reason": "未检测到符合条件的V型周期",
                "kline": kline,
                "key_info": key_info,
            })
            print(f"  ❌ {tag} {name}: 未通过(无V型周期)")
            continue

        # 计算韧性评分
        score = calc_resilience_score(cycles)

        # 清理内部索引字段
        clean_cycles = []
        for c in cycles:
            cc = {k: v for k, v in c.items() if not k.startswith("_")}
            clean_cycles.append(cc)

        passed.append({
            "name": name,
            "symbol": symbol,
            "market": market,
            "stock_type": stock_type,
            "duplicate_of": duplicate_of,
            "score": score,
            "cycle_count": len(cycles),
            "cycles": clean_cycles,
            "kline": kline,
            "key_info": key_info,
        })
        print(f"  ✅ {tag} {name}: 通过 (评分={score}, 周期数={len(cycles)})")

    # 按评分降序排序
    passed.sort(key=lambda x: x["score"], reverse=True)

    # 构建输出 JSON
    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "min_drop_pct": MIN_DROP_PCT * 100,
            "min_recovery_pct": MIN_RECOVERY_PCT * 100,
            "lookback_start": LOOKBACK_START,
            "ma_window": MA_WINDOW,
            "peak_window": PEAK_WINDOW,
            "min_drop_days": MIN_DROP_DAYS,
            "min_recovery_days": MIN_RECOVERY_DAYS,
        },
        "summary": {
            "total": total,
            "passed": len(passed),
            "failed": len(failed),
            "skipped": len(skipped),
        },
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }

    # 写入 JSON
    json_path = os.path.join(OUTPUT_DIR, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))

    file_size_mb = os.path.getsize(json_path) / 1024 / 1024

    # 打印总结
    print(f"\n{'='*60}")
    print(f"韧性筛选完成!")
    print(f"  总计: {total}")
    print(f"  通过: {len(passed)}")
    print(f"  未通过: {len(failed)}")
    print(f"  跳过: {len(skipped)}")
    print(f"\n输出: {json_path} ({file_size_mb:.1f} MB)")
    print(f"\n提示: 用以下命令启动本地服务器查看报告:")
    print(f"  python3 -m http.server -d {OUTPUT_DIR} 8080")
    print(f"  然后打开 http://localhost:8080")


if __name__ == "__main__":
    main()
