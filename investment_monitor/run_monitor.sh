#!/bin/bash
# 投资监控定时任务包装脚本 v2

cd /root/.openclaw/workspace/investment_monitor

# 记录日志
LOG_FILE="./logs/cron_$(date +%Y%m%d).log"

echo "======================================" >> $LOG_FILE
echo "开始执行监控: $(date)" >> $LOG_FILE
echo "版本: v2.0 混合数据源" >> $LOG_FILE
echo "======================================" >> $LOG_FILE

# 执行监控（v2版本，启用推送）
python3 monitor_v2.py >> $LOG_FILE 2>&1

EXIT_CODE=$?

echo "执行完成: $(date), 退出码: $EXIT_CODE" >> $LOG_FILE
echo "" >> $LOG_FILE

exit $EXIT_CODE
