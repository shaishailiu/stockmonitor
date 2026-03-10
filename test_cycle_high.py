"""
测试脚本：验证 find_cycle_high() 阶段性高点识别算法
以腾讯控股（00700.HK）为例
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# ─── 从 stock_monitor.py 复制核心函数（避免导入整个模块的依赖问题） ───

SIGNIFICANT_REBOUND = 0.20

def find_cycle_high(df: pd.DataFrame) -> dict:
    """找到当前下跌周期的阶段性高点。"""
    highs = df["high"].values if "high" in df.columns else df["close"].values
    dates = df["date"].values

    if len(highs) == 0:
        return {"peak_price": None, "peak_date": None,
                "trough_price": None, "trough_date": None}

    global_high_idx = int(np.argmax(highs))
    stage_high = highs[global_high_idx]
    stage_high_idx = global_high_idx
    trough = highs[global_high_idx]
    trough_idx = global_high_idx

    for i in range(global_high_idx + 1, len(highs)):
        if highs[i] < trough:
            trough = highs[i]
            trough_idx = i
        if trough > 0 and highs[i] >= trough * (1 + SIGNIFICANT_REBOUND):
            if highs[i] > stage_high or i > stage_high_idx:
                stage_high = highs[i]
                stage_high_idx = i
            trough = highs[i]
            trough_idx = i

    for i in range(stage_high_idx, len(highs)):
        if highs[i] >= stage_high:
            stage_high = highs[i]
            stage_high_idx = i

    prev_trough_price = None
    prev_trough_date = None
    if stage_high_idx > 0:
        search_start = max(0, global_high_idx)
        segment = highs[search_start:stage_high_idx]
        if len(segment) > 0:
            min_idx_in_seg = int(np.argmin(segment))
            prev_trough_price = float(segment[min_idx_in_seg])
            prev_trough_date = pd.Timestamp(dates[search_start + min_idx_in_seg]).strftime("%Y-%m-%d")

    peak_date = pd.Timestamp(dates[stage_high_idx]).strftime("%Y-%m-%d")

    return {
        "peak_price": float(stage_high),
        "peak_date": peak_date,
        "trough_price": prev_trough_price,
        "trough_date": prev_trough_date,
    }


# ─── 测试逻辑 ───

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_stock_data(symbol: str) -> pd.DataFrame:
    """从本地 data 目录加载股票数据"""
    path = os.path.join(DATA_DIR, f"{symbol}.json")
    if not os.path.exists(path):
        print(f"❌ 数据文件不存在: {path}")
        print(f"   请先运行 stock_history.py 获取数据")
        return pd.DataFrame()

    df = pd.read_json(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def test_stock(name: str, symbol: str):
    """测试指定股票的阶段性高点识别"""
    print(f"\n{'='*60}")
    print(f"  测试股票: {name} ({symbol})")
    print(f"{'='*60}")

    df = load_stock_data(symbol)
    if df.empty:
        return

    # 基本数据信息
    print(f"\n📊 数据概览:")
    print(f"   数据范围: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"   数据条数: {len(df)}")
    print(f"   当前价格: {df['close'].iloc[-1]:.2f}")

    # 全局高点和低点（用 high 列）
    high_col = "high" if "high" in df.columns else "close"
    low_col = "low" if "low" in df.columns else "close"

    global_high_idx = df[high_col].idxmax()
    global_low_idx = df[low_col].idxmin()

    print(f"\n📈 全局最高点:")
    print(f"   价格: {df.loc[global_high_idx, high_col]:.2f}")
    print(f"   日期: {df.loc[global_high_idx, 'date'].strftime('%Y-%m-%d')}")

    print(f"\n📉 全局最低点:")
    print(f"   价格: {df.loc[global_low_idx, low_col]:.2f}")
    print(f"   日期: {df.loc[global_low_idx, 'date'].strftime('%Y-%m-%d')}")

    # 使用 find_cycle_high 找阶段性高点
    # 用最近2年数据
    cutoff_2y = datetime.now() - timedelta(weeks=104)
    df_2y = df[df["date"] >= cutoff_2y].copy()

    if df_2y.empty:
        print("\n⚠️ 最近2年无数据")
        return

    print(f"\n📊 最近2年数据:")
    print(f"   范围: {df_2y['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df_2y['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"   条数: {len(df_2y)}")

    # 2年数据中的简单最高/最低
    high_2y_idx = df_2y[high_col].idxmax()
    low_2y_idx = df_2y[low_col].idxmin()
    high_2y = df_2y.loc[high_2y_idx, high_col]
    low_2y = df_2y.loc[low_2y_idx, low_col]

    print(f"\n   2年最高: {high_2y:.2f} ({df_2y.loc[high_2y_idx, 'date'].strftime('%Y-%m-%d')})")
    print(f"   2年最低: {low_2y:.2f} ({df_2y.loc[low_2y_idx, 'date'].strftime('%Y-%m-%d')})")

    # 调用 find_cycle_high
    result = find_cycle_high(df_2y)
    current_price = df_2y["close"].iloc[-1]

    print(f"\n🎯 find_cycle_high() 结果:")
    print(f"   本轮高点: {result['peak_price']:.2f} ({result['peak_date']})")
    if result["trough_price"] is not None:
        print(f"   前次低谷: {result['trough_price']:.2f} ({result['trough_date']})")
    else:
        print(f"   前次低谷: 无（高点即全局高点）")

    drop_pct = (1 - current_price / result["peak_price"]) * 100
    print(f"\n   当前价格: {current_price:.2f}")
    print(f"   距本轮高点回撤: {drop_pct:.1f}%")

    # 对比：如果用简单52周高点
    cutoff_52w = datetime.now() - timedelta(weeks=52)
    df_52w = df[df["date"] >= cutoff_52w].copy()
    if not df_52w.empty:
        simple_high_52w = df_52w[high_col].max()
        simple_drop = (1 - current_price / simple_high_52w) * 100
        print(f"\n📊 对比 - 简单52周高点:")
        print(f"   52周最高: {simple_high_52w:.2f}")
        print(f"   距52周高点回撤: {simple_drop:.1f}%")
        diff = drop_pct - simple_drop
        if abs(diff) > 0.1:
            print(f"   ⚡ 差异: {diff:+.1f}%（阶段高点法回撤{'更大' if diff > 0 else '更小'}）")
        else:
            print(f"   ✅ 两种方法结果一致")

    # 显示近期价格走势关键点
    print(f"\n📋 近30天价格走势（采样）:")
    df_recent = df_2y.tail(30)
    for i in range(0, len(df_recent), 5):
        row = df_recent.iloc[i]
        print(f"   {row['date'].strftime('%Y-%m-%d')}  "
              f"开:{row.get('open', 'N/A'):>8.2f}  "
              f"高:{row.get('high', 'N/A'):>8.2f}  "
              f"低:{row.get('low', 'N/A'):>8.2f}  "
              f"收:{row['close']:>8.2f}")
    # 最后一天
    last = df_recent.iloc[-1]
    print(f"   {last['date'].strftime('%Y-%m-%d')}  "
          f"开:{last.get('open', 'N/A'):>8.2f}  "
          f"高:{last.get('high', 'N/A'):>8.2f}  "
          f"低:{last.get('low', 'N/A'):>8.2f}  "
          f"收:{last['close']:>8.2f}  ← 最新")


if __name__ == "__main__":
    import sys
    # 将输出同时写到文件
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, s):
            for f in self.files:
                f.write(s)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    # 腾讯控股
    test_stock("腾讯控股", "00700")

    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"\n结果已保存到: {log_path}")
