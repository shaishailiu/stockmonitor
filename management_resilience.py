"""management_resilience.py — 管理层韧性评估系统

基于 K 线特征 + 腾讯云 DeepSeek API（联网搜索）逐只分析管理层韧性。
结果即时写入 JSON，支持中断后重新运行（已有结果的股票自动跳过）。

运行方式：
  python management_resilience.py

环境变量：
  LLM_API_KEY  — 腾讯云 API Key（也可在脚本内配置）
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ══════════════════════════════════════════
#  配置
# ══════════════════════════════════════════

CONFIG_FILE = "config.json"
NEW_DATA_DIR = "newdata"
OUTPUT_DIR = "resilience_report"
RESULT_FILE = os.path.join(OUTPUT_DIR, "management_resilience.json")

# 腾讯云 DeepSeek API
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # 在此处填写或通过环境变量设置
LLM_BASE_URL = "https://api.lkeap.cloud.tencent.com/v1"
LLM_MODEL = "deepseek-v3"

# 算法筛选参数（比 resilience_screener.py 更宽松）
MIN_DRAWDOWN_PCT = 30       # 最小回撤 30%（宽松门槛）
LOOKBACK_START = "2019-01-01"
MIN_DATA_POINTS = 250
MA_WINDOW = 5

# 请求间隔（秒），避免 API 限流
REQUEST_INTERVAL = 2


# ══════════════════════════════════════════
#  数据加载
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
    """加载单只股票的历史日线数据"""
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
    required_cols = ["date", "open", "high", "low", "close"]
    if not all(c in df.columns for c in required_cols):
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ["open", "high", "low", "close", "volume", "amount", "pe"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["close"].notna() & (df["close"] > 0)]
    if df.empty:
        return None

    cutoff = pd.Timestamp(LOOKBACK_START)
    df_recent = df[df["date"] >= cutoff].copy()
    if len(df_recent) < MIN_DATA_POINTS:
        df_recent = df.copy()
    if len(df_recent) < MIN_DATA_POINTS:
        return None

    df_recent = df_recent.sort_values("date").reset_index(drop=True)
    return df_recent


# ══════════════════════════════════════════
#  V 型周期检测（宽松版）
# ══════════════════════════════════════════

PEAK_WINDOW = 60          # 峰值检测窗口（约3个月）
MIN_DROP_PCT_CYCLE = 0.30 # 至少跌30%
MIN_RECOVERY_PCT = 0.50   # 恢复到前高的50%算走出来（比 resilience_screener 的70%更宽松）
MIN_DROP_DAYS = 40        # 最小下跌持续交易日
MIN_RECOVERY_DAYS = 40    # 最小恢复持续交易日


def detect_peaks(smoothed: np.ndarray, window: int = PEAK_WINDOW) -> list[int]:
    """检测局部峰值"""
    n = len(smoothed)
    peaks = []
    for i in range(window, n - window):
        left = smoothed[max(0, i - window):i]
        right = smoothed[i + 1:min(n, i + window + 1)]
        if smoothed[i] >= max(left) and smoothed[i] >= max(right):
            peaks.append(i)
    return peaks


def find_v_cycles(df: pd.DataFrame) -> list[dict]:
    """检测所有完整的 V 型周期：峰值 → 跌≥30% → 恢复到前高50%
    按峰值从高到低排序处理，避免小峰值吞掉大周期。
    """
    df = df.copy()
    df["smoothed"] = df["close"].rolling(window=MA_WINDOW, min_periods=1).mean()

    smoothed = df["smoothed"].values
    dates = df["date"].values
    n = len(smoothed)

    peak_indices = detect_peaks(smoothed)
    if not peak_indices:
        return []

    # 按峰值价格从高到低排序，优先处理最显著的峰值
    peak_indices.sort(key=lambda i: smoothed[i], reverse=True)

    cycles = []
    used_ranges = []  # 已被占用的索引范围

    for peak_idx in peak_indices:
        # 检查是否与已有周期重叠
        if any(start <= peak_idx <= end for start, end in used_ranges):
            continue

        peak_price = smoothed[peak_idx]
        drop_threshold = peak_price * (1 - MIN_DROP_PCT_CYCLE)
        recovery_target = peak_price * MIN_RECOVERY_PCT

        # 阶段1：从峰值往后扫描，找到跌破门槛后的最低谷底
        # 用很高的恢复门槛(90%)来决定何时停止搜索，确保找到真正的最深谷底
        trough_idx = None
        trough_price = peak_price
        has_dropped_enough = False
        scan_stop = peak_price * 0.90  # 只有恢复到峰值90%以上才停止搜索谷底

        for j in range(peak_idx + 1, n):
            if any(start <= j <= end for start, end in used_ranges):
                break
            if smoothed[j] < trough_price:
                trough_price = smoothed[j]
                trough_idx = j
            if trough_price <= drop_threshold:
                has_dropped_enough = True
            # 只有跌够了，且从谷底过了足够久，且价格恢复到接近峰值，才停止搜索
            if (has_dropped_enough and trough_idx is not None
                    and (j - trough_idx) >= MIN_RECOVERY_DAYS
                    and smoothed[j] >= scan_stop):
                break

        if not has_dropped_enough or trough_idx is None:
            continue

        drop_days = trough_idx - peak_idx
        if drop_days < MIN_DROP_DAYS:
            continue

        # 阶段2：找恢复终点——从谷底之后追踪到这轮反弹的实际最高点
        # 先确认首次过50%恢复门槛的点存在
        first_recovery_idx = None
        for j in range(trough_idx + 1, n):
            if any(start <= j <= end for start, end in used_ranges):
                break
            if smoothed[j] >= recovery_target:
                first_recovery_idx = j
                break

        if first_recovery_idx is None:
            continue

        # 从谷底往后找最高点，作为恢复终点
        rally_high = trough_price
        recovery_idx = trough_idx
        for j in range(trough_idx + 1, n):
            if any(start <= j <= end for start, end in used_ranges):
                break
            if smoothed[j] > rally_high:
                rally_high = smoothed[j]
                recovery_idx = j

        recovery_days = recovery_idx - trough_idx
        if recovery_days < MIN_RECOVERY_DAYS:
            continue

        drop_pct = (peak_price - trough_price) / peak_price
        recovery_price = smoothed[recovery_idx]

        cycles.append({
            "peak_date": str(pd.Timestamp(dates[peak_idx]).date()),
            "peak_price": round(float(peak_price), 2),
            "trough_date": str(pd.Timestamp(dates[trough_idx]).date()),
            "trough_price": round(float(trough_price), 2),
            "recovery_date": str(pd.Timestamp(dates[recovery_idx]).date()),
            "recovery_price": round(float(recovery_price), 2),
            "drop_pct": round(float(drop_pct * 100), 1),
            "recovery_pct": round(float(recovery_price / peak_price * 100), 1),
            "drop_days": int(drop_days),
            "recovery_days": int(recovery_days),
            "_peak_idx": int(peak_idx),
            "_trough_idx": int(trough_idx),
            "_recovery_idx": int(recovery_idx),
        })
        used_ranges.append((peak_idx, recovery_idx))

    # 按时间排序输出
    cycles.sort(key=lambda c: c["_peak_idx"])
    return cycles


# ══════════════════════════════════════════
#  K 线特征提取（基于 V 型周期）
# ══════════════════════════════════════════

def extract_features(item: dict, df: pd.DataFrame) -> list[dict] | None:
    """对每个完整的 V 型周期生成独立特征，返回特征列表（每个周期一条）。
    只关注历史上"跌下去→爬回来"的完整过程。
    """
    cycles = find_v_cycles(df)
    if not cycles:
        return None

    features_list = []
    for cycle in cycles:
        peak_idx = cycle["_peak_idx"]
        recovery_idx = cycle["_recovery_idx"]

        # 取周期范围内的数据，前后各扩展30个交易日做上下文
        start_idx = max(0, peak_idx - 30)
        end_idx = min(len(df), recovery_idx + 30)
        df_period = df.iloc[start_idx:end_idx].copy()

        # 该周期范围内的季度股价
        df_q = df_period.copy().set_index("date")
        quarterly = df_q["close"].resample("QE").last().dropna()
        quarterly_str = ", ".join([f"{d.strftime('%Y-Q')}{(d.month-1)//3+1}:{v:.2f}" for d, v in quarterly.items()])

        # 该周期范围内的月线走势
        monthly = df_q["close"].resample("ME").last().dropna()
        monthly_str = ", ".join([f"{d.strftime('%Y-%m')}:{v:.2f}" for d, v in monthly.items()])

        # PE（周期结束时的PE）
        pe = None
        if "pe" in df.columns:
            pe_vals = df_period["pe"].dropna()
            pe_vals = pe_vals[pe_vals > 0]
            if not pe_vals.empty:
                pe = round(float(pe_vals.iloc[-1]), 2)

        features_list.append({
            "name": item["name"],
            "symbol": item["symbol"],
            "market": item["market"],
            "stock_type": item.get("stock_type", "unclassified"),
            "cycle_index": len(features_list) + 1,
            "peak_price": cycle["peak_price"],
            "peak_date": cycle["peak_date"],
            "trough_price": cycle["trough_price"],
            "trough_date": cycle["trough_date"],
            "recovery_price": cycle["recovery_price"],
            "recovery_date": cycle["recovery_date"],
            "drop_pct": cycle["drop_pct"],
            "recovery_pct": cycle["recovery_pct"],
            "drop_days": cycle["drop_days"],
            "recovery_days": cycle["recovery_days"],
            "quarterly_prices": quarterly_str,
            "monthly_prices": monthly_str,
            "pe": pe,
        })

    return features_list if features_list else None


# ══════════════════════════════════════════
#  腾讯云 DeepSeek API 调用（流式）
# ══════════════════════════════════════════

def call_deepseek(prompt: str, enable_search: bool = True) -> str:
    """调用腾讯云 DeepSeek API（流式模式，支持联网搜索）"""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": True,
    }
    if enable_search:
        payload["enable_search"] = True

    resp = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    full_content = ""
    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if text.startswith("data: "):
            text = text[6:]
        if text.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(text)
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            full_content += delta
        except (json.JSONDecodeError, IndexError, KeyError):
            continue

    return full_content


def build_prompt(features: dict) -> str:
    """构造给 LLM 的分析提示词——只聚焦一段完整的 V 型周期"""
    market_map = {"a": "A股", "hk": "港股", "us": "美股"}
    market_label = market_map.get(features["market"], features["market"])

    return f"""你是一位专注于管理层能力评估的深度研究分析师。请搜索该公司的公开信息，然后基于事实进行分析。

## 公司信息
- 公司：{features['name']}（{features['symbol']}，{market_label}）
- 类型：{features['stock_type']}

## 分析范围：一段完整的"跌下去→爬回来"的V型周期

**重要：只分析以下这段时间范围内发生的事情，不要混入其他时期的信息。**

- 📈 峰值：{features['peak_price']}（{features['peak_date']}）
- 📉 谷底：{features['trough_price']}（{features['trough_date']}），下跌 {features['drop_pct']}%，耗时 {features['drop_days']} 个交易日
- 🔄 恢复：{features['recovery_price']}（{features['recovery_date']}），恢复到峰值的 {features['recovery_pct']}%，耗时 {features['recovery_days']} 个交易日
{"- 当前PE：" + str(features['pe']) if features['pe'] else ""}

**这段周期内的股价走势（按季度）：**
{features['quarterly_prices']}

**这段周期内的月线收盘价：**
{features['monthly_prices']}

## 核心分析要求

请讲清楚这段**从 {features['peak_date']} 到 {features['recovery_date']}** 期间发生的完整故事：

1. **下跌阶段（{features['peak_date']} → {features['trough_date']}）**：股价从 {features['peak_price']} 跌到 {features['trough_price']}，期间到底发生了什么？
2. **恢复阶段（{features['trough_date']} → {features['recovery_date']}）**：股价从 {features['trough_price']} 恢复到 {features['recovery_price']}，管理层做了什么让公司走出来的？
3. 管理层的行动是主动的还是被动的？

请务必给出**具体的时间、事件、人物、数字、对应的股价**，例如：
- ✅ "2022-03，CEO张某宣布裁员30%（约5000人），当时股价约4.5元，此后半年股价企稳并回升至7元"
- ❌ "管理层实施了降本增效措施"（太笼统）

## 输出格式
请严格用以下 JSON 格式输出（不要包含其他文字）：
```json
{{
  "crisis_type": "行业下行/经营失误/政策打击/竞争失利/财务危机/多重因素",
  "crisis_timeline": [
    {{
      "date": "YYYY-MM",
      "stock_price": "当时股价约XX元",
      "event": "具体发生了什么导致股价下跌"
    }}
  ],
  "management_response_timeline": [
    {{
      "date": "YYYY-MM",
      "stock_price": "决策时股价约XX元",
      "action": "谁做了什么具体决策，规模多大",
      "result": "实际效果如何",
      "stock_price_after": "此后股价变化到约XX元"
    }}
  ],
  "action_quality": "主动求变/被迫应对/躺平等待",
  "recovery_driver": "股价恢复的核心驱动力是什么（管理层行动/行业回暖/政策利好/多重因素）",
  "management_resilience_score": 65,
  "key_insight": "这段V型周期最值得关注的一点",
  "one_line_summary": "一句话概括：从XX元跌到XX元（原因），管理层做了XX，股价恢复到XX元"
}}
```

注意：
- 只分析 {features['peak_date']} 到 {features['recovery_date']} 这段时间，不要把之后的事情混进来
- crisis_timeline 中的事件必须发生在下跌阶段（{features['peak_date']}~{features['trough_date']}）
- management_response_timeline 中的行动可以在谷底前后（管理层可能在下跌途中就开始行动）
- stock_price 必须从上面的季度/月线数据中找对应值，不要编造
- 评分标准：90+ 管理层主导的翻盘，70-89 积极应对有成效，50-69 有行动但效果一般，30-49 迟缓，<30 躺平/纯靠行业回暖"""


def parse_llm_response(content: str) -> dict | None:
    """解析 LLM 返回的 JSON"""
    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    import re
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 { 和最后一个 }
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ══════════════════════════════════════════
#  结果持久化（断点续跑）
# ══════════════════════════════════════════

def load_existing_results() -> dict:
    """加载已有的分析结果"""
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            pass
    return {
        "generated_at": "",
        "config": {
            "model": LLM_MODEL,
            "min_drawdown_pct": MIN_DRAWDOWN_PCT,
            "lookback_start": LOOKBACK_START,
        },
        "progress": {
            "total_candidates": 0,
            "analyzed": 0,
            "skipped": 0,
            "failed": 0,
        },
        "results": {},   # symbol -> analysis result
        "skipped": {},    # symbol -> reason
    }


def save_results(data: dict):
    """写入结果到 JSON 文件"""
    data["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════

def main():
    # 检查 API Key
    if not LLM_API_KEY:
        print("❌ 请设置 LLM_API_KEY 环境变量或在脚本中填写")
        print("   export LLM_API_KEY=sk-xxxxxxxx")
        print("   或在 Windows: set LLM_API_KEY=sk-xxxxxxxx")
        sys.exit(1)

    # 加载已有结果（支持断点续跑）
    results = load_existing_results()

    # 清除所有旧格式结果（重新分析）
    old_keys = [k for k, v in results["results"].items()
                if not v.get("cycle") and not v.get("analysis", {}).get("parse_error")]
    if old_keys:
        print(f"发现 {len(old_keys)} 条旧格式结果，将重新分析。")
        for k in old_keys:
            del results["results"][k]

    existing_keys = set(results["results"].keys())

    # 加载股票列表
    stocks = load_config()
    total = len(stocks)

    print(f"{'='*60}")
    print(f"  管理层韧性评估系统（V型周期版）")
    print(f"  模型: {LLM_MODEL} (腾讯云 DeepSeek)")
    print(f"  股票池: {total} 只")
    print(f"  已有结果: {len(results['results'])} 条")
    print(f"{'='*60}\n")

    # 阶段1: 算法宽筛 + V型周期检测
    all_features = []  # (result_key, features_dict)
    skip_count = 0

    for item in stocks:
        symbol = item["symbol"]
        name = item["name"]
        market = item["market"]

        # 跳过 crypto
        if market == "crypto":
            if symbol not in results["skipped"]:
                results["skipped"][symbol] = {"name": name, "reason": "crypto类型，不适用"}
            skip_count += 1
            continue

        # 跳过 duplicate
        if item.get("duplicate_of"):
            if symbol not in results["skipped"]:
                results["skipped"][symbol] = {"name": name, "reason": f"A+H重复（对应 {item['duplicate_of']}）"}
            skip_count += 1
            continue

        # 加载数据
        df = load_stock_data(item)
        if df is None:
            if symbol not in results["skipped"]:
                results["skipped"][symbol] = {"name": name, "reason": "数据不足或不存在"}
            skip_count += 1
            continue

        # 检测 V 型周期
        features_list = extract_features(item, df)
        if features_list is None:
            if symbol not in results["skipped"]:
                results["skipped"][symbol] = {"name": name, "reason": "未检测到完整的V型周期（跌≥30%且恢复≥50%）"}
            skip_count += 1
            continue

        for feat in features_list:
            result_key = f"{symbol}_cycle{feat['cycle_index']}"
            all_features.append((result_key, feat))

    # 筛选需要分析的
    to_analyze = [(k, f) for k, f in all_features if k not in existing_keys]

    print(f"阶段1完成: 检测到 {len(all_features)} 个V型周期 (来自 {len(set(f['symbol'] for _, f in all_features))} 只股票), 跳过 {skip_count} 只")
    print(f"  已有结果 {len(all_features) - len(to_analyze)} 个, 待分析 {len(to_analyze)} 个\n")

    if not to_analyze:
        print("所有V型周期已分析完毕，无需重新请求。")
        save_results(results)
        print_summary(results)
        return

    save_results(results)

    # 阶段2: 逐个调用 LLM 分析
    success_count = 0
    fail_count = 0

    for i, (result_key, features) in enumerate(to_analyze, 1):
        symbol = features["symbol"]
        name = features["name"]
        ci = features["cycle_index"]

        print(f"[{i}/{len(to_analyze)}] {name} 第{ci}个V型周期")
        print(f"  {features['peak_price']}({features['peak_date']}) -> {features['trough_price']}({features['trough_date']}) -> {features['recovery_price']}({features['recovery_date']})")
        print(f"  下跌 {features['drop_pct']}%, 恢复到峰值 {features['recovery_pct']}%")

        try:
            prompt = build_prompt(features)
            raw_response = call_deepseek(prompt, enable_search=True)

            analysis = parse_llm_response(raw_response)
            if analysis is None:
                print(f"  LLM 返回格式异常，保存原始回复")
                analysis = {"raw_response": raw_response, "parse_error": True}

            results["results"][result_key] = {
                "name": name,
                "symbol": symbol,
                "market": features["market"],
                "stock_type": features["stock_type"],
                "cycle": {
                    "index": ci,
                    "peak_price": features["peak_price"],
                    "peak_date": features["peak_date"],
                    "trough_price": features["trough_price"],
                    "trough_date": features["trough_date"],
                    "recovery_price": features["recovery_price"],
                    "recovery_date": features["recovery_date"],
                    "drop_pct": features["drop_pct"],
                    "recovery_pct": features["recovery_pct"],
                    "drop_days": features["drop_days"],
                    "recovery_days": features["recovery_days"],
                },
                "analysis": analysis,
                "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            results["progress"]["analyzed"] = len(results["results"])
            save_results(results)

            score = analysis.get("management_resilience_score", "?")
            summary = analysis.get("one_line_summary", "解析失败")
            print(f"  评分: {score}  {summary}")
            success_count += 1

        except requests.exceptions.HTTPError as e:
            print(f"  API 错误: {e}")
            results["results"][result_key] = {
                "name": name, "symbol": symbol,
                "market": features["market"], "stock_type": features["stock_type"],
                "cycle": {
                    "index": ci,
                    "peak_price": features["peak_price"], "peak_date": features["peak_date"],
                    "trough_price": features["trough_price"], "trough_date": features["trough_date"],
                    "recovery_price": features["recovery_price"], "recovery_date": features["recovery_date"],
                    "drop_pct": features["drop_pct"], "recovery_pct": features["recovery_pct"],
                    "drop_days": features["drop_days"], "recovery_days": features["recovery_days"],
                },
                "analysis": {"error": str(e), "parse_error": True},
                "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_results(results)
            fail_count += 1

        except Exception as e:
            print(f"  异常: {e}")
            fail_count += 1

        if i < len(to_analyze):
            time.sleep(REQUEST_INTERVAL)

    # 最终保存
    results["progress"]["analyzed"] = len(results["results"])
    results["progress"]["failed"] = fail_count
    save_results(results)

    print(f"\n{'='*60}")
    print(f"本轮分析完成: 成功 {success_count}, 失败 {fail_count}")
    print_summary(results)


def print_summary(results: dict):
    """打印结果摘要"""
    analyzed = results["results"]
    if not analyzed:
        print("暂无分析结果。")
        return

    scored = []
    errors = []
    for key, data in analyzed.items():
        analysis = data.get("analysis", {})
        if analysis.get("parse_error"):
            errors.append(data)
            continue
        score = analysis.get("management_resilience_score", 0)
        scored.append((score, key, data))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'='*60}")
    print(f"  管理层韧性评估结果排行")
    print(f"  共 {len(analyzed)} 个V型周期, 有效 {len(scored)}, 解析失败 {len(errors)}")
    print(f"{'='*60}")

    print(f"\n{'排名':<4} {'评分':<6} {'行动质量':<10} {'名称':<16} {'周期'}")
    print(f"{'-'*80}")

    for rank, (score, key, data) in enumerate(scored[:30], 1):
        a = data["analysis"]
        quality = a.get("action_quality", "?")
        name = data["name"]
        c = data.get("cycle", {})
        period = f"{c.get('peak_date','?')} -> {c.get('trough_date','?')} -> {c.get('recovery_date','?')}"
        print(f"{rank:<4} {score:<6} {quality:<10} {name:<16} {period}")

    print(f"\n输出文件: {RESULT_FILE}")
    print(f"可随时中断，重新运行将自动跳过已分析的周期。")


if __name__ == "__main__":
    main()
