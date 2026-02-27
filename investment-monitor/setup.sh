#!/bin/bash
# 投资监控系统环境配置脚本

set -e

echo "🚀 开始配置投资监控环境..."

# 检查 Python 版本
echo "检查 Python 版本..."
python3 --version

# 安装 pip (如果没有)
if ! command -v pip3 &> /dev/null; then
    echo "正在安装 pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py --user
    rm /tmp/get-pip.py
    export PATH="$HOME/.local/bin:$PATH"
fi

# 安装依赖包
echo "安装 Python 依赖包..."
python3 -m pip install --user -r requirements.txt

# 验证安装
echo ""
echo "验证安装..."
python3 -c "import akshare; print(f'✅ AkShare 版本: {akshare.__version__}')"
python3 -c "import pandas; print(f'✅ Pandas 版本: {pandas.__version__}')"

echo ""
echo "✅ 环境配置完成！"
echo ""
echo "接下来可以运行："
echo "  python3 monitor.py          # 手动执行一次扫描"
echo "  python3 monitor.py --test   # 测试模式（只扫描少量资产）"
