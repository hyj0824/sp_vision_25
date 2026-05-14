#!/bin/bash

# 串口设备
SERIAL_PORT="/dev/ttyACM0"
# 波特率
BAUD_RATE="921600"
# 输出的原始二进制文件
OUTPUT_FILE="serial_dump.raw"

# 检查串口是否存在
if [ ! -e "$SERIAL_PORT" ]; then
    echo "错误: 串口 $SERIAL_PORT 不存在"
    exit 1
fi

# 设置权限
echo "配置串口权限..."
sudo chmod 666 "$SERIAL_PORT"

# 配置串口参数（关键！）
echo "设置波特率 $BAUD_RATE..."
stty -F $SERIAL_PORT \
    $BAUD_RATE \
    raw \
    -echo \
    cs8 \
    -cstopb \
    -parenb \
    ignbrk \
    -icrnl \
    -opost \
    -onlcr \
    -isig \
    -icanon

# 开始读取原始数据（直接写入二进制文件）
echo "===================================="
echo "正在捕获原始二进制数据 → $OUTPUT_FILE"
echo "按 Ctrl+C 停止"
echo "===================================="

# 直接读取串口原始数据，写入文件（无任何转换）
cat $SERIAL_PORT > $OUTPUT_FILE

# 退出后恢复终端
stty sane
echo -e "\n已停止，数据保存在: $OUTPUT_FILE"
