"""定时调度：每天自动获取数据、生成报告、推送到企业微信机器人。

使用方式:
  python scheduler.py                        # 启动调度（前台运行）
  python scheduler.py --once                 # 立即执行一次（调试用）
  python scheduler.py --test-webhook         # 测试企业微信推送

环境变量:
  WECOM_WEBHOOK  企业微信机器人 Webhook URL（必填）

时间表:
  - 每天 10:00：增量更新数据 → 分别推送 G/T/O 三板块报告（3条消息）
  - 每天 22:00：增量更新数据 → 合并推送三板块报告（1条消息）
"""
import os
import sys
import subprocess
import json
import io
import time
import argparse
from datetime import datetime

# ── 配置 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_SCRIPT = os.path.join(SCRIPT_DIR, "stock_history_v2.py")
MONITOR_SCRIPT = os.path.join(SCRIPT_DIR, "stock_monitor_v2.py")
WECOM_WEBHOOK = os.environ.get("WECOM_WEBHOOK", "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=65bfafa0-b407-4846-9cf9-e1b877625af3")

SECTORS = [
    ("G", "🎮 游戏行业"),
    ("T", "💻 科技行业"),
    ("O", "📊 其他行业"),
]


# ═══════════════════════════════════════════
#  企业微信机器人推送
# ═══════════════════════════════════════════

def send_wecom(msg: str, webhook: str = "") -> bool:
    """发送 Markdown 消息到企业微信机器人。"""
    webhook = webhook or WECOM_WEBHOOK
    if not webhook:
        print("[wecom] ❌ 未设置 WECOM_WEBHOOK 环境变量，跳过推送")
        return False

    payload = {
        "msgtype": "markdown",
        "markdown": {"content": msg},
    }
    try:
        import urllib.request
        req = urllib.request.Request(
            webhook,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("errcode") == 0:
                print("[wecom] ✅ 推送成功")
                return True
            else:
                print(f"[wecom] ❌ 推送失败: {result}")
                return False
    except Exception as e:
        print(f"[wecom] ❌ 推送异常: {e}")
        return False


# ═══════════════════════════════════════════
#  任务执行
# ═══════════════════════════════════════════

def run_update() -> str:
    """增量更新数据。返回日志文本。"""
    print("\n" + "=" * 50)
    print(f"[{datetime.now():%H:%M:%S}] 开始增量更新数据...")
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", HISTORY_SCRIPT],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )
        log = result.stdout + result.stderr
        print(log)
        return log
    except subprocess.TimeoutExpired:
        print("[update] ❌ 更新超时")
        return "❌ 数据更新超时"
    except Exception as e:
        print(f"[update] ❌ 更新失败: {e}")
        return f"❌ 数据更新失败: {e}"


def run_monitor(sector: str, compact: bool = True) -> str:
    """运行监控报告。返回报告文本。"""
    label = dict(SECTORS).get(sector, sector)
    print(f"[{datetime.now():%H:%M:%S}] 生成 {label} 报告...")
    cmd = [sys.executable, "-X", "utf8", MONITOR_SCRIPT, "-s", sector]
    if compact:
        cmd.append("-c")
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"[monitor] ❌ {label} 超时")
        return f"❌ {label} 报告生成超时"
    except Exception as e:
        print(f"[monitor] ❌ {label} 失败: {e}")
        return f"❌ {label} 报告生成失败: {e}"


def _chunk_lines(lines: list[str], header: str, max_body: int, label: str = "") -> list[str]:
    """将行列表按 max_body 字节上限切分为多个带 header 的消息块。"""
    chunks: list[str] = []
    current_lines: list[str] = []
    current_size = len(header.encode("utf-8"))

    for line in lines:
        line_size = len(line.encode("utf-8")) + 1  # +1 for \n
        if current_lines and current_size + line_size > max_body:
            chunks.append(header + "\n".join(current_lines))
            current_lines = [line]
            current_size = len(header.encode("utf-8")) + line_size
        else:
            current_lines.append(line)
            current_size += line_size

    if current_lines:
        chunks.append(header + "\n".join(current_lines))

    return chunks


def format_wecom_markdown(title: str, report: str, timestamp: str = "") -> list[str]:
    """将监控报告拆分为多条企业微信 Markdown 消息（每条 ≤ 4096 字节）。
    先按 卡片 / 表格 两大区域分开，各自内部按边界拆分。
    """
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    MAX_BODY = 3600
    HEADER = f"# {title}\n> 更新时间：{timestamp}\n\n"

    body = report.strip()

    # 不需要拆分
    if len(HEADER.encode("utf-8")) + len(body.encode("utf-8")) <= 4096:
        return [HEADER + body]

    # ── 1. 分离卡片区和表格区 ──
    table_marker = "回撤超过25%的股票一览"
    lines = body.split("\n")

    marker_idx = None
    for i, line in enumerate(lines):
        if table_marker in line:
            marker_idx = i
            break

    if marker_idx is not None:
        cards_lines = lines[:marker_idx]
        table_lines = lines[marker_idx:]  # 含标题行
    else:
        cards_lines = lines
        table_lines = []

    all_chunks: list[str] = []

    # ── 2. 卡片区按逻辑边界拆分 ──
    if cards_lines:
        cards_text = "\n".join(cards_lines).strip()
        # 检测分隔符：完整模式用分隔线，精简模式用双空行
        sep = "─" * 40
        if sep in cards_text:
            card_parts = [p.strip() for p in cards_text.split(sep) if p.strip()]
        else:
            card_parts = [p.strip() for p in cards_text.split("\n\n") if p.strip()]

        # 将卡片累积到 block 中
        card_blocks: list[str] = []
        current_block: list[str] = []
        for card in card_parts:
            card_lines_list = card.split("\n")
            block_size = sum(len(l.encode("utf-8")) + 1 for l in current_block) + sum(len(l.encode("utf-8")) + 1 for l in card_lines_list)
            if current_block and block_size > MAX_BODY - 200:
                card_blocks.append("\n".join(current_block))
                current_block = card_lines_list
            else:
                if current_block:
                    current_block.append("")  # 卡片间空行
                current_block.extend(card_lines_list)
        if current_block:
            card_blocks.append("\n".join(current_block))

        # 生成头部 chunk：第一块包含板块标题行
        for i, block in enumerate(card_blocks):
            block_lines = block.split("\n")
            card_chunks = _chunk_lines(block_lines, HEADER, MAX_BODY)
            if i == 0 and marker_idx is not None:
                label_text = "底部信号"
            else:
                label_text = "底部信号（续）"
            for ci, c in enumerate(card_chunks):
                hdr = f"# {title} · {label_text}\n> 更新时间：{timestamp}\n\n"
                updated = c.replace(HEADER, hdr, 1)
                all_chunks.append(updated)

    # ── 3. 表格区按行拆分 ──
    if table_lines:
        table_header = f"# {title} · 回撤一览\n> 更新时间：{timestamp}\n\n"
        table_chunks = _chunk_lines(table_lines, table_header, MAX_BODY)
        all_chunks.extend(table_chunks)

    # ── 分页标注 ──
    if len(all_chunks) > 1:
        for i, chunk in enumerate(all_chunks):
            head, s, tail = chunk.partition("\n")
            all_chunks[i] = f"{head} ({i + 1}/{len(all_chunks)}){s}{tail}"

    return all_chunks


def send_report(title: str, report: str, timestamp: str = ""):
    """生成并分段发送报告。自动按卡片边界拆分为多条消息。"""
    msgs = format_wecom_markdown(title, report, timestamp)
    for msg in msgs:
        send_wecom(msg)
        time.sleep(0.5)  # 避免频率限制


def morning_job():
    """上午任务：更新数据 → 3 板块分别推送。"""
    print(f"\n{'=' * 50}")
    print(f"[{datetime.now():%H:%M:%S}] 🌅 上午推送开始")
    run_update()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for sector, label in SECTORS:
        report = run_monitor(sector, compact=True)
        if report.strip():
            send_report(f"📊 {label} 底部信号", report, now)
    print(f"[{datetime.now():%H:%M:%S}] 🌅 上午推送完成")


def evening_job():
    """晚上任务：更新数据 → 三板块合并推送。"""
    print(f"\n{'=' * 50}")
    print(f"[{datetime.now():%H:%M:%S}] 🌙 晚上推送开始")
    update_log = run_update()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_reports = []

    for sector, label in SECTORS:
        report = run_monitor(sector, compact=True)
        if report.strip():
            all_reports.append(f"## {label}\n\n{report.strip()}")

    if all_reports:
        combined = "\n\n---\n\n".join(all_reports)
        send_report("📈 今日底部信号汇总", combined, now)

    # 附带更新日志
    if update_log.strip():
        log_msg = f"## ⚙️ 数据更新日志\n```\n{update_log.strip()[-500:]}\n```"
        send_wecom(log_msg)

    print(f"[{datetime.now():%H:%M:%S}] 🌙 晚上推送完成")


def once():
    """立即执行一次完整流程（调试用）。"""
    print(f"[{datetime.now():%H:%M:%S}] ⚡ 立即执行（含数据更新）")
    run_update()
    push_now()
    print(f"[{datetime.now():%H:%M:%S}] ⚡ 执行完成")


def push_now():
    """仅生成报告并推送（不更新数据）。"""
    print(f"[{datetime.now():%H:%M:%S}] 📨 生成报告并推送...")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for sector, label in SECTORS:
        report = run_monitor(sector, compact=True)
        if report.strip():
            send_report(f"📊 {label} 底部信号", report, now)
    print(f"[{datetime.now():%H:%M:%S}] 📨 推送完成")


def test_webhook():
    """测试企业微信推送。"""
    if not WECOM_WEBHOOK:
        print("❌ 请先设置环境变量 WECOM_WEBHOOK")
        print("   PowerShell: $env:WECOM_WEBHOOK='https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx'")
        return
    ok = send_wecom(f"## ✅ 股票监控推送测试\n> 时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n\n连接正常！")
    print("测试完成" if ok else "测试失败")


# ═══════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="股票监控定时调度")
    parser.add_argument("--once", action="store_true", help="立即执行一次完整流程（更新数据 + 推送报告）")
    parser.add_argument("--push", action="store_true", help="仅生成报告并推送（不更新数据）")
    parser.add_argument("--test-webhook", action="store_true", help="测试企业微信推送")
    args = parser.parse_args()

    if args.test_webhook:
        test_webhook()
        return

    if args.once:
        once()
        return

    if args.push:
        push_now()
        return

    # ── 定时调度 ──
    try:
        import schedule
    except ImportError:
        print("正在安装 schedule...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "schedule", "-q"])
        import schedule

    if not WECOM_WEBHOOK:
        print("⚠️  未设置 WECOM_WEBHOOK 环境变量，推送功能不可用")
        print("   PowerShell: $env:WECOM_WEBHOOK='https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx'")

    schedule.every().day.at("10:00").do(morning_job)
    schedule.every().day.at("22:00").do(evening_job)

    print("=" * 50)
    print("🚀 股票监控调度器已启动")
    print(f"   上午推送：每天 10:00（3 板块分别推送）")
    print(f"   晚上推送：每天 22:00（合并推送 + 更新日志）")
    print("   按 Ctrl+C 停止")
    print("=" * 50)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
