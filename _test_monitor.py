"""调用 stock_monitor.generate_report() 打印最新监控报告。"""

import sys
import os
import io

# Windows 终端 GBK 编码不支持 emoji，强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 确保工作目录正确（stock_monitor 内部用相对路径读 config/data）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    from stock_monitor import generate_report

    print("=" * 60)
    print("  调用 stock_monitor.generate_report()")
    print("=" * 60)
    print()

    report = generate_report()
    if report and report.strip():
        print(report)
    else:
        print("⚠️  报告为空：没有股票触发预警，也没有回撤超过25%的股票。")
        print("    可能原因：find_cycle_high 的反弹重置逻辑低估了回撤幅度。")

except Exception as e:
    print(f"\n❌ 运行出错：{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc(file=sys.stdout)
