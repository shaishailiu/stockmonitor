"""PE（市盈率）数据管理模块。

与 stock_history.py 类似的架构：
- 读取 config.json 遍历所有股票
- 每只股票独立文件：pedata/{symbol}.json
- 增量更新：已有数据只计算差值，没有则拉全量历史
- 利用 data/ 目录已有的收盘价 + 东方财富 EPS 计算 PE(TTM)

文件格式：[{date, pe}, ...]（按日期升序）

直接运行本脚本可批量更新所有股票的 PE 数据：
    python pe_store.py
"""

import json
import os
import traceback
import requests
import pandas as pd
from datetime import datetime, timedelta

CONFIG_FILE = "config.json"
DATA_DIR = "data"
PE_DATA_DIR = "pedata"

# 批量更新时的请求间隔（秒），避免触发接口限流
BATCH_SLEEP = 1.0


# ──────────────────────────────────────
#  配置 & 文件路径
# ──────────────────────────────────────

def load_config():
    """从 config.json 加载股票配置列表。"""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["stocks"]


def get_pe_filename(item: dict) -> str:
    """根据配置项生成 PE 数据文件名（存放在 pedata 目录下）。"""
    symbol = item["symbol"]
    if item["market"] == "us":
        name = f"{symbol.split('.')[-1]}.json"
    else:
        name = f"{symbol}.json"
    return os.path.join(PE_DATA_DIR, name)


def get_history_filename(item: dict) -> str:
    """根据配置项获取行情数据文件路径（data 目录下）。"""
    symbol = item["symbol"]
    if item["market"] == "us":
        name = f"{symbol.split('.')[-1]}.json"
    else:
        name = f"{symbol}.json"
    return os.path.join(DATA_DIR, name)


# ──────────────────────────────────────
#  PE 本地文件读写
# ──────────────────────────────────────

def load_pe_data(filename: str) -> pd.DataFrame:
    """从本地 JSON 文件加载 PE 数据。"""
    if not os.path.exists(filename):
        return pd.DataFrame(columns=["date", "pe"])
    try:
        with open(filename, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not records:
            return pd.DataFrame(columns=["date", "pe"])
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "pe"])


def save_pe_data(df: pd.DataFrame, filename: str):
    """将 PE 数据保存到 JSON 文件。"""
    if df.empty:
        return
    # 确保日期排序
    df = df.sort_values("date").reset_index(drop=True)
    records = df[["date", "pe"]].to_dict(orient="records")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_history_prices(filename: str) -> pd.DataFrame:
    """从行情 JSON 文件加载日期和收盘价。"""
    if not os.path.exists(filename):
        return pd.DataFrame(columns=["date", "close"])
    try:
        df = pd.read_json(filename)
        if df.empty or "close" not in df.columns or "date" not in df.columns:
            return pd.DataFrame(columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df[["date", "close"]].dropna(subset=["close"])
    except Exception:
        return pd.DataFrame(columns=["date", "close"])


# ──────────────────────────────────────
#  EPS 获取 & TTM 计算
# ──────────────────────────────────────

def _get_eastmoney_secucode(market: str, symbol: str) -> str | list[str]:
    """生成东方财富财报接口的 SECUCODE。

    美股返回列表（先 .O 后 .N），其他市场返回字符串。
    """
    if market == "hk":
        return f"{symbol}.HK"
    elif market == "us":
        ticker = symbol.split(".")[-1]
        # 美股可能在 NASDAQ(.O) 或 NYSE(.N) 上市，返回两个候选
        return [f"{ticker}.O", f"{ticker}.N"]
    else:
        suffix = "SH" if symbol.startswith(("6", "9")) else "SZ"
        return f"{symbol}.{suffix}"


def _fetch_quarterly_eps(market: str, symbol: str) -> list[dict] | None:
    """从东方财富获取每季度 EPS 数据。

    Returns:
        按报告日期降序排列的 [{report_date, eps}, ...] 列表，或 None。
    """
    secucode_raw = _get_eastmoney_secucode(market, symbol)
    # 美股返回列表（多个候选），其他返回字符串
    secucodes = secucode_raw if isinstance(secucode_raw, list) else [secucode_raw]

    # A 股用 RPT_LICO_FN_CPD（字段名是 REPORTDATE），港股/美股用不同报表（字段名是 REPORT_DATE）
    if market == "hk":
        report_name = "RPT_HKF10_FN_MAININDICATOR"
        date_field = "REPORT_DATE"
        sort_col = "REPORT_DATE"
    elif market == "us":
        report_name = "RPT_USF10_FN_GMAININDICATOR"
        date_field = "REPORT_DATE"
        sort_col = "REPORT_DATE"
    else:
        report_name = "RPT_LICO_FN_CPD"
        date_field = "REPORTDATE"
        sort_col = "REPORTDATE"

    for secucode in secucodes:
        try:
            url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
            params = {
                "reportName": report_name,
                "columns": f"SECUCODE,{date_field},BASIC_EPS",
                "filter": f'(SECUCODE="{secucode}")',
                "pageNumber": 1,
                "pageSize": 50,
                "sortTypes": "-1",
                "sortColumns": sort_col,
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            data = resp.json()

            if not data.get("result") or not data["result"].get("data"):
                continue  # 尝试下一个 secucode

            records = data["result"]["data"]
            result = []
            for r in records:
                rd = r.get(date_field, "")[:10]
                eps = r.get("BASIC_EPS")
                if rd and eps is not None:
                    result.append({"report_date": rd, "eps": float(eps)})
            if result:
                return result
        except Exception:
            traceback.print_exc()
            continue

    return None


def _calc_ttm_eps(quarterly_eps: list[dict], target_date: str) -> float | None:
    """根据季度/半年度 EPS 数据，计算目标日期对应的 TTM EPS。

    自动检测报告频率：
    - A 股通常有 Q1/Q2/Q3/Q4 四个季度报告
    - 港股通常只有中报(6月) + 年报(12月)
    对于半年度报告，用最近2个半年度数据（中报+下半年）计算 TTM。
    """
    from datetime import datetime as dt

    target = dt.strptime(target_date, "%Y-%m-%d")

    # 按报告日期排序（升序方便计算）
    sorted_eps = sorted(quarterly_eps, key=lambda x: x["report_date"])

    # 财报发布时间估算
    # Q1(3-31) → 4月底, Q2/中报(6-30) → 8月底, Q3(9-30) → 10月底, Q4/年报(12-31) → 次年3月底
    def is_published(report_date_str: str, at_date) -> bool:
        rd = dt.strptime(report_date_str, "%Y-%m-%d")
        month = rd.month
        if month == 3:    # Q1 → 4月底
            pub = rd.replace(month=4, day=30)
        elif month == 6:  # Q2/中报 → 8月底
            pub = rd.replace(month=8, day=31)
        elif month == 9:  # Q3 → 10月底
            pub = rd.replace(month=10, day=31)
        elif month == 12: # Q4/年报 → 次年3月底
            pub = rd.replace(year=rd.year + 1, month=3, day=31)
        else:
            pub = rd + timedelta(days=90)
        return at_date >= pub

    available = [e for e in sorted_eps if is_published(e["report_date"], target)]
    if not available:
        return None

    # 检测报告频率：看出现了哪些季度
    all_quarters = set()
    eps_by_yq: dict[tuple, float] = {}  # (year, quarter) -> eps
    for e in sorted_eps:
        rd = dt.strptime(e["report_date"], "%Y-%m-%d")
        q = (rd.month - 1) // 3 + 1
        all_quarters.add(q)
        eps_by_yq[(rd.year, q)] = e["eps"]

    # 判断是否半年度报告（典型港股）：
    # 1. 只有 Q2 和 Q4
    # 2. 或虽有 Q3 但 Q2=Q3 值完全相同（东方财富将中报重复展示到 Q3）且没有 Q1
    is_semi_annual = all_quarters.issubset({2, 4})

    if not is_semi_annual and 1 not in all_quarters and 3 in all_quarters and 2 in all_quarters:
        # 检查是否 Q2 = Q3 同值（所有同年份）
        q2_eq_q3 = True
        for (y, q), v in eps_by_yq.items():
            if q == 2 and (y, 3) in eps_by_yq:
                if abs(v - eps_by_yq[(y, 3)]) > 0.001:
                    q2_eq_q3 = False
                    break
        if q2_eq_q3:
            is_semi_annual = True
            # 移除重复的 Q3 数据，只保留 Q2 和 Q4
            def _not_q3(e):
                return (dt.strptime(e["report_date"], "%Y-%m-%d").month - 1) // 3 + 1 != 3
            sorted_eps = [e for e in sorted_eps if _not_q3(e)]
            available = [e for e in available if _not_q3(e)]

    # 建立按年-期间索引的 EPS
    eps_by_year: dict[int, dict[int, float]] = {}
    for e in sorted_eps:
        rd = dt.strptime(e["report_date"], "%Y-%m-%d")
        year = rd.year
        quarter = (rd.month - 1) // 3 + 1
        if year not in eps_by_year:
            eps_by_year[year] = {}
        eps_by_year[year][quarter] = e["eps"]

    if is_semi_annual:
        # ── 半年度模式 ──
        # 中报(Q2) = 上半年累计, 年报(Q4) = 全年累计
        # 下半年 EPS = 年报 - 中报
        single_half_eps: dict[tuple, float] = {}
        for year, periods in eps_by_year.items():
            if 2 in periods:
                single_half_eps[(year, "H1")] = periods[2]  # 上半年
            if 4 in periods:
                if 2 in periods:
                    single_half_eps[(year, "H2")] = periods[4] - periods[2]  # 下半年
                else:
                    # 只有年报没中报，无法拆分，整年当 TTM
                    single_half_eps[(year, "FULL")] = periods[4]

        # 最新已发布的报告
        latest = available[-1]
        latest_rd = dt.strptime(latest["report_date"], "%Y-%m-%d")
        latest_year = latest_rd.year
        latest_q = (latest_rd.month - 1) // 3 + 1

        ttm_eps = 0.0
        count = 0

        if latest_q == 4:
            # 最新是年报 → TTM = 全年 EPS
            if (latest_year, "H1") in single_half_eps and (latest_year, "H2") in single_half_eps:
                ttm_eps = single_half_eps[(latest_year, "H1")] + single_half_eps[(latest_year, "H2")]
                count = 2
            elif (latest_year, "FULL") in single_half_eps:
                ttm_eps = single_half_eps[(latest_year, "FULL")]
                count = 2
        elif latest_q == 2:
            # 最新是中报 → TTM = 今年上半年 + 去年下半年
            h1 = single_half_eps.get((latest_year, "H1"))
            h2 = single_half_eps.get((latest_year - 1, "H2"))
            if h1 is not None and h2 is not None:
                ttm_eps = h1 + h2
                count = 2

        if count < 2:
            return None
        return ttm_eps

    else:
        # ── 季度模式（A 股等） ──
        # 年报EPS是全年累计，Q3是前三季累计，Q2是半年累计，Q1是第一季
        single_quarter_eps: dict[tuple, float] = {}
        for year, quarters in eps_by_year.items():
            for q in sorted(quarters.keys()):
                if q == 1:
                    single_quarter_eps[(year, q)] = quarters[q]
                else:
                    prev_q = q - 1
                    if prev_q in quarters:
                        single_quarter_eps[(year, q)] = quarters[q] - quarters[prev_q]
                    else:
                        single_quarter_eps[(year, q)] = quarters[q]

        latest = available[-1]
        latest_rd = dt.strptime(latest["report_date"], "%Y-%m-%d")
        latest_year = latest_rd.year
        latest_quarter = (latest_rd.month - 1) // 3 + 1

        ttm_eps = 0.0
        count = 0
        y, q = latest_year, latest_quarter
        for _ in range(4):
            key = (y, q)
            if key in single_quarter_eps:
                ttm_eps += single_quarter_eps[key]
                count += 1
            q -= 1
            if q == 0:
                q = 4
                y -= 1

        if count < 4:
            return None
        return ttm_eps


# ──────────────────────────────────────
#  PE 批量计算核心逻辑
# ──────────────────────────────────────

def calc_pe_for_dates(quarterly_eps: list[dict], prices_df: pd.DataFrame) -> pd.DataFrame:
    """根据 EPS 数据和价格数据，为每个日期计算 PE(TTM)。

    Args:
        quarterly_eps: 季度 EPS 数据列表
        prices_df: DataFrame with columns ['date', 'close']

    Returns:
        DataFrame with columns ['date', 'pe']
    """
    results = []

    # 预计算：找到 EPS 数据覆盖的最早可算日期
    # （需要至少一个已发布的报告才能算 TTM）
    if not quarterly_eps:
        return pd.DataFrame(columns=["date", "pe"])

    # 缓存 TTM EPS：同一报告期内 TTM EPS 相同，不需要每天重算
    # 按月缓存（同月内 TTM EPS 通常不变）
    ttm_cache: dict[str, float | None] = {}

    for _, row in prices_df.iterrows():
        date_str = row["date"]
        close = row["close"]
        if pd.isna(close) or close <= 0:
            continue

        # 用月份做缓存键（同月内 TTM EPS 通常相同）
        month_key = date_str[:7]  # "YYYY-MM"
        if month_key not in ttm_cache:
            ttm_cache[month_key] = _calc_ttm_eps(quarterly_eps, date_str)

        ttm_eps = ttm_cache[month_key]
        if ttm_eps is None or ttm_eps == 0:
            continue

        pe = round(close / ttm_eps, 2)
        results.append({"date": date_str, "pe": pe})

    return pd.DataFrame(results) if results else pd.DataFrame(columns=["date", "pe"])


def update_pe_for_item(item: dict) -> dict:
    """为单只股票更新 PE 数据（增量更新）。

    流程：
    1. 读取本地已有 PE 数据
    2. 读取行情数据（价格）
    3. 如果已有 PE 数据，只计算缺失的日期
    4. 如果没有 PE 数据，全量计算
    5. 拉取 EPS，结合价格计算 PE(TTM)
    6. 合并保存

    Returns:
        dict with keys: name, status ("updated"|"skipped"|"failed"), detail (str)
    """
    name = item["name"]
    market = item["market"]

    if market == "crypto":
        return {"name": name, "status": "skipped", "detail": "加密货币无PE"}

    pe_file = get_pe_filename(item)
    history_file = get_history_filename(item)

    # 1. 加载行情价格数据
    prices_df = load_history_prices(history_file)
    if prices_df.empty:
        return {"name": name, "status": "failed", "detail": "无行情数据"}

    # 2. 加载已有 PE 数据
    pe_df = load_pe_data(pe_file)
    existing_dates = set(pe_df["date"].tolist()) if not pe_df.empty else set()

    # 3. 计算需要补充的日期
    all_price_dates = set(prices_df["date"].tolist())

    if not pe_df.empty:
        # 已有数据 → 只关心比最晚 PE 日期更新的价格日期（增量更新）
        # 远历史中 EPS 不覆盖的日期不再反复重试
        latest_pe_date = pe_df["date"].max()
        missing_dates = {d for d in all_price_dates if d > latest_pe_date and d not in existing_dates}
    else:
        # 全新股票 → 全量计算
        missing_dates = all_price_dates - existing_dates

    if not missing_dates:
        latest_pe_date = pe_df["date"].max() if not pe_df.empty else "N/A"
        return {"name": name, "status": "skipped", "detail": f"已是最新（{latest_pe_date}）"}

    # 4. 只取需要补充的价格数据
    missing_prices = prices_df[prices_df["date"].isin(missing_dates)].copy()

    # 5. 获取 EPS 数据
    symbol = item["symbol"]
    quarterly_eps = _fetch_quarterly_eps(market, symbol)
    if not quarterly_eps:
        return {"name": name, "status": "failed", "detail": "获取EPS失败"}

    # 6. 计算 PE
    new_pe_df = calc_pe_for_dates(quarterly_eps, missing_prices)

    # 7. 合并并保存（即使只有部分日期计算成功也保存）
    if not new_pe_df.empty:
        if not pe_df.empty:
            merged = pd.concat([pe_df, new_pe_df], ignore_index=True)
            merged = merged.drop_duplicates(subset="date", keep="last")
            merged = merged.sort_values("date").reset_index(drop=True)
        else:
            merged = new_pe_df.sort_values("date").reset_index(drop=True)

        save_pe_data(merged, pe_file)

        new_count = len(new_pe_df)
        uncovered = len(missing_dates) - new_count
        total = len(merged)
        date_range = f"{merged['date'].iloc[0]} ~ {merged['date'].iloc[-1]}"
        detail = f"+{new_count}条，共{total}条（{date_range}）"
        if uncovered > 0:
            detail += f"，{uncovered}条因EPS不覆盖跳过"
        return {"name": name, "status": "updated", "detail": detail}
    else:
        return {"name": name, "status": "failed", "detail": f"EPS数据不覆盖缺失的{len(missing_dates)}个日期"}


# ──────────────────────────────────────
#  批量更新入口（类似 stock_history.run_update）
# ──────────────────────────────────────

def run_update() -> str:
    """遍历 config.json 中所有股票，批量更新 PE 数据。"""
    import time

    os.makedirs(PE_DATA_DIR, exist_ok=True)
    config = load_config()

    updated = []
    skipped = []
    failed = []

    total = len(config)
    print(f"开始更新 {total} 只股票的 PE 数据...\n")

    for i, item in enumerate(config):
        result = update_pe_for_item(item)

        if result["status"] == "updated":
            updated.append(result)
            print(f"  [{i+1}/{total}] ✅ {result['name']}：{result['detail']}")
        elif result["status"] == "skipped":
            skipped.append(result)
            print(f"  [{i+1}/{total}] ⏭️ {result['name']}：{result['detail']}")
        else:
            failed.append(result)
            print(f"  [{i+1}/{total}] ❌ {result['name']}：{result['detail']}")

        # 避免请求过快（只在实际请求了远程API时等待）
        if result["status"] != "skipped":
            time.sleep(BATCH_SLEEP)

    lines = []
    lines.append(f"PE数据更新完成：{len(updated)}个已更新，{len(skipped)}个无需更新，{len(failed)}个失败")

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


# ──────────────────────────────────────
#  命令行入口
# ──────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    run_update()
