#!/bin/bash
# set -e就是只要某条命令执行失败并返回非 0，脚本原则上立即退出
set -e

# ============================================================================
# G1 Deploy - Deployment Script
# ============================================================================
# This script handles the complete setup and deployment process for g1_deploy
# Following the steps from the README.md
# 这里说明了脚本支持四种启动模式：sim、real、指定接口名(eth0)、指定IP地址(192.168.123.100)
# 如果不带任何参数./deploy.sh这样就默认是real也就是./deploy.sh real
# Usage: ./deploy.sh [sim|real|<interface_name>|<ip_address>]
#   sim   - Use loopback interface for simulation (MuJoCo)
#   real  - Auto-detect robot network interface (192.168.123.x)
#   <interface_name> - Use specific interface (e.g., enP8p1s0, eth0)
#   <ip_address> - Use interface with specific IP
#
# Default: real
# ============================================================================

# Colors for output
# 定义终端颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory (where this script is located)
# 确定脚本自己的目录，假设脚本路径是：/home/user/g1_deploy/deploy.sh
# 那么${BASH_SOURCE[0]}就是/home/user/g1_deploy/deploy.sh，dirname就是取目录/home/user/g1_deploy
# 所以得到的SCRIPT_DIR就是/home/user/g1_deploy，然后cd进入该目录，这么做是为了后续方便相对路径引用其他文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Interface Resolution Functions
# ============================================================================

# Get all network interfaces and their IPs
# Returns lines of: interface_name:ip_address
# 函数：扫描电脑的所有网络接口
# 输出形式：
# lo:127.0.0.1
# eth0:192.168.123.10
# wlan0:192.168.1.20
get_network_interfaces() {
    # 如果是macOS系统，使用ifconfig命令获取网络接口和IP地址（macos系统通常返回Darwin）
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        ifconfig | awk '
            /^[a-z]/ { iface=$1; gsub(/:$/, "", iface) }
            /inet / { print iface ":" $2 }
        '
    else
    # 如果是linux系统则使用ip -4 addr show命令获取网络接口和IP地址
        # Linux
        ip -4 addr show 2>/dev/null | awk '
            /^[0-9]+:/ { gsub(/:$/, "", $2); iface=$2 }
            /inet / { split($2, a, "/"); print iface ":" a[1] }
        ' 2>/dev/null || \
        ifconfig 2>/dev/null | awk '
            /^[a-z]/ { iface=$1; gsub(/:$/, "", iface) }
            /inet / { 
                for (i=1; i<=NF; i++) {
                    if ($i == "inet") { print iface ":" $(i+1); break }
                    if ($i ~ /^addr:/) { split($i, a, ":"); print iface ":" a[2]; break }
                }
            }
        '
    fi
}

# Find interface by IP address
# Returns interface name or empty string
# 函数：根据指定的IP地址查找对应的网络接口
# 例如lo:127.0.0.1，eth0:192.168.123.10，则执行find_interface_by_ip "127.0.0.1"返回lo
find_interface_by_ip() {
    local target_ip="$1"
    get_network_interfaces | while IFS=: read -r iface ip; do
        if [[ "$ip" == "$target_ip" ]]; then
            echo "$iface"
            return 0
        fi
    done
}

# Find interface with IP matching a prefix
# Returns interface name or empty string
# 函数：根据指定的IP前缀查找对应的网络接口，例如eth0:192.168.123.10
# 则执行find_interface_by_ip_prefix "192.168.123."返回eth0
find_interface_by_ip_prefix() {
    local prefix="$1"
    get_network_interfaces | while IFS=: read -r iface ip; do
        if [[ "$ip" == "$prefix"* ]]; then
            echo "$iface"
            return 0
        fi
    done
}

# Check if string is an IP address
# 判断是否是ip地址，判断方式就是是否是数字.数字.数字.数字的形式
is_ip_address() {
    local input="$1"
    if [[ "$input" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        return 0
    fi
    return 1
}

# Check if interface has a specific IP
# 检查一个网卡是不是拥有某个 IP
interface_has_ip() {
    local iface="$1"
    local target_ip="$2"
    get_network_interfaces | while IFS=: read -r name ip; do
        if [[ "$name" == "$iface" ]] && [[ "$ip" == "$target_ip" ]]; then
            echo "yes"
            return 0
        fi
    done
}

# Resolve interface parameter to actual network interface name and environment type
# Arguments: interface - "sim", "real", or direct interface name or IP address
# Outputs: Sets TARGET and ENV_TYPE variables
# 最重要的网络函数，会调用上述的一堆函数，最后得到两个变量TARGET和ENV_TYPE
# 其中TARGET是最终要使用的网络接口名比如lo或者eth0，ENV_TYPE是环境类型sim或real，通常sim的话就是lo
# 如果是real的话就是eth0，例如./deploy.sh 127.0.0.1就最后得到TARGET=lo，ENV_TYPE=sim
resolve_interface() {
    local interface="$1"
    local os_type="$(uname)"
    
    # Check if interface is an IP address
    if is_ip_address "$interface"; then
        if [[ "$interface" == "127.0.0.1" ]]; then
            TARGET="$interface"
            ENV_TYPE="sim"
        else
            TARGET="$interface"
            ENV_TYPE="real"
        fi
        return 0
    fi
    
    if [[ "$interface" == "sim" ]]; then
        local lo_interface
        lo_interface=$(find_interface_by_ip "127.0.0.1")
        
        if [[ -n "$lo_interface" ]]; then
            # macOS uses lo0 instead of lo
            if [[ "$os_type" == "Darwin" ]] && [[ "$lo_interface" == "lo" ]]; then
                TARGET="lo0"
            else
                TARGET="$lo_interface"
            fi
        else
            # Fallback
            if [[ "$os_type" == "Darwin" ]]; then
                TARGET="lo0"
            else
                TARGET="lo"
            fi
        fi
        ENV_TYPE="sim"
        return 0
    
    elif [[ "$interface" == "real" ]]; then
        # Try to find interface with 192.168.123.x IP (Unitree robot network)
        local real_interface
        real_interface=$(find_interface_by_ip_prefix "192.168.123.")
        
        if [[ -n "$real_interface" ]]; then
            TARGET="$real_interface"
        else
            # Fallback to common interface names
            # Try to find any non-loopback interface
            local fallback_interface
            fallback_interface=$(get_network_interfaces | grep -v "127.0.0.1" | head -1 | cut -d: -f1)
            
            if [[ -n "$fallback_interface" ]]; then
                TARGET="$fallback_interface"
                echo -e "${YELLOW}⚠️  Could not find 192.168.123.x interface, using: $TARGET${NC}" >&2
            else
                # Ultimate fallback
                TARGET="enP8p1s0"
                echo -e "${YELLOW}⚠️  Could not auto-detect interface, using default: $TARGET${NC}" >&2
            fi
        fi
        ENV_TYPE="real"
        return 0
    
    else
        # Direct interface name - check if it has 127.0.0.1 to determine env_type
        local has_loopback
        has_loopback=$(interface_has_ip "$interface" "127.0.0.1")
        
        if [[ "$has_loopback" == "yes" ]]; then
            TARGET="$interface"
            ENV_TYPE="sim"
            return 0
        fi
        
        # macOS lo interface handling
        if [[ "$os_type" == "Darwin" ]] && [[ "$interface" == "lo" ]]; then
            TARGET="lo0"
            ENV_TYPE="sim"
            return 0
        fi
        
        # Default to real for unknown interfaces
        TARGET="$interface"
        ENV_TYPE="real"
        return 0
    fi
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================
# 帮助菜单
show_usage() {
    echo "Usage: $0 [OPTIONS] [sim|real|<interface>]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  --cp, --checkpoint PATH Set the checkpoint path (default: policy/checkpoints/example/model_step_000000)"
    echo "  --obs-config PATH       Set the observation config file (default: policy/configs/example.yaml)"
    echo "  --planner PATH          Set the planner model path (default: planner/example.onnx)"
    echo "  --motion-data PATH      Set the motion data path (default: reference/example_motion/)"
    echo "  --input-type TYPE       Set the input type (default: zmq_manager)"
    echo "  --output-type TYPE      Set the output type (default: ros2)"
    echo "  --zmq-host HOST         Set the ZMQ host (default: localhost)"
    echo "  --motor-kp-scale SPEC   Scale Kp for hardware motor indices/ranges"
    echo "  --motor-kd-scale SPEC   Scale Kd for hardware motor indices/ranges"
    echo ""
    echo "Interface modes:"
    echo "  sim              Use loopback interface for simulation (MuJoCo)"
    echo "  real             Auto-detect robot network (192.168.123.x)"
    echo "  <interface>      Use specific interface (e.g., enP8p1s0, eth0)"
    echo "  <ip_address>     Use interface by IP address"
    echo ""
    echo "Default: real"
    echo ""
    echo "Examples:"
    echo "  $0 sim           # Run in simulation mode"
    echo "  $0 real          # Auto-detect real robot interface"
    echo "  $0 enP8p1s0      # Use specific interface"
    echo "  $0 192.168.x.x # Use interface with this IP"
    echo "  $0 --cp policy/checkpoints/custom/model_step_123456 real  # Use custom checkpoint"
    echo "  $0 --obs-config policy/configs/custom.yaml sim  # Use custom obs config"
    echo "  $0 --planner planner/custom.onnx --input-type keyboard real  # Use custom planner and input"
    echo "  $0 --motion-data reference/custom_motion/ sim  # Use custom motion data"
}
# 默认配置，默认是real模式
# Default interface mode
INTERFACE_MODE="real"

# Default configuration values (can be overridden by command line)
# 模型默认路径，从huggingface上下载的模型就是放在policy/release/model目录下
CHECKPOINT_DEFAULT="policy/release/model"
# 观测默认配置
OBS_CONFIG_DEFAULT="policy/release/observation_config.yaml"
# 规划器默认路径，这个是分层架构，能够根据速度来生成运动序列
PLANNER_DEFAULT="planner/target_vel/V2/planner_sonic.onnx"
# 运动序列默认路径，这里不是指向一个具体的文件，而是指向一个目录，里面有很多运动序列文件
MOTION_DATA_DEFAULT="reference/example/"
# 默认创建输入管理器，它把键盘、手柄、ZMQ 和 ROS 2 输入整合起来，可在运行时切换
INPUT_TYPE_DEFAULT="manager"
# 创建所有当前构建可用的输出接口，主要是 ZMQ
OUTPUT_TYPE_DEFAULT="all"
# 指定ZMQ主机，默认是localhost，也就是ZMQ发布者和部署程序在同一台机器
ZMQ_HOST_DEFAULT="localhost"

# Initialize with defaults (will be set after parsing)
# 初始化变量，初始化为默认值
CHECKPOINT="$CHECKPOINT_DEFAULT"
OBS_CONFIG="$OBS_CONFIG_DEFAULT"
PLANNER="$PLANNER_DEFAULT"
MOTION_DATA="$MOTION_DATA_DEFAULT"
INPUT_TYPE="$INPUT_TYPE_DEFAULT"
OUTPUT_TYPE="$OUTPUT_TYPE_DEFAULT"
ZMQ_HOST="$ZMQ_HOST_DEFAULT"
# 初始化kp和kd的数组
MOTOR_KP_SCALES=()
MOTOR_KD_SCALES=()

# Parse arguments
# 命令行参数解析，这里参数可以指定是仿真还是实机
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --cp|--checkpoint)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --cp/--checkpoint requires a path argument${NC}" >&2
                exit 1
            fi
            CHECKPOINT="$2"
            shift 2
            ;;
        --obs-config)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --obs-config requires a path argument${NC}" >&2
                exit 1
            fi
            OBS_CONFIG="$2"
            shift 2
            ;;
        --planner)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --planner requires a path argument${NC}" >&2
                exit 1
            fi
            PLANNER="$2"
            shift 2
            ;;
        --motion-data)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --motion-data requires a path argument${NC}" >&2
                exit 1
            fi
            MOTION_DATA="$2"
            shift 2
            ;;
        --input-type)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --input-type requires a type argument${NC}" >&2
                exit 1
            fi
            INPUT_TYPE="$2"
            shift 2
            ;;
        --output-type)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --output-type requires a type argument${NC}" >&2
                exit 1
            fi
            OUTPUT_TYPE="$2"
            shift 2
            ;;
        --zmq-host)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --zmq-host requires a host argument${NC}" >&2
                exit 1
            fi
            ZMQ_HOST="$2"
            shift 2
            ;;
        --motor-kp-scale)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --motor-kp-scale requires <motor-list>=<factor>${NC}" >&2
                exit 1
            fi
            MOTOR_KP_SCALES+=("$2")
            shift 2
            ;;
        --motor-kd-scale)
            if [[ -z "$2" ]]; then
                echo -e "${RED}Error: --motor-kd-scale requires <motor-list>=<factor>${NC}" >&2
                exit 1
            fi
            MOTOR_KD_SCALES+=("$2")
            shift 2
            ;;
        sim|real)
            INTERFACE_MODE="$1"
            shift
            ;;
        *)
            # Could be interface name or IP
            INTERFACE_MODE="$1"
            shift
            ;;
    esac
done

# ============================================================================
# Display Header
# ============================================================================
# 打印启动logo
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                         G1 DEPLOY LAUNCHER                           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# Resolve Interface
# ============================================================================

echo -e "${BLUE}[Interface Resolution]${NC}"
echo "Requested mode: $INTERFACE_MODE"
# 这里调用了上面定义的resolve_interface函数，最终得到TARGET和ENV_TYPE两个变量
resolve_interface "$INTERFACE_MODE"
# 打印得到的这两个变量TARGET和ENV_TYPE
echo -e "Resolved interface: ${GREEN}$TARGET${NC}"
echo -e "Environment type:   ${GREEN}$ENV_TYPE${NC}"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Model checkpoint path (set via command line or default)
# CHECKPOINT and OBS_CONFIG are already set from argument parsing above

# Decoder and Encoder ONNX models
# 把 checkpoint 拆成 encoder 和 decoder
CHECKPOINT_DECODER="${CHECKPOINT}_decoder.onnx"
CHECKPOINT_ENCODER="${CHECKPOINT}_encoder.onnx"

# Motion data path (set via command line or default)
# MOTION_DATA is already set from argument parsing above

# Observation config (set via command line or default)
# OBS_CONFIG is already set from argument parsing above

# Planner model (set via command line or default)
# PLANNER is already set from argument parsing above

# Input type (set via command line or default)
# INPUT_TYPE is already set from argument parsing above

# Output type (set via command line or default)
# OUTPUT_TYPE is already set from argument parsing above

# ZMQ host (set via command line or default)
# ZMQ_HOST is already set from argument parsing above

# Additional flags for simulation mode
# 额外参数统一存在这里，sim 模式自动加：--disable-crc-check，也就是mujoco不进行crc检查
EXTRA_ARGS=()
if [[ "$ENV_TYPE" == "sim" ]]; then
    EXTRA_ARGS+=("--disable-crc-check")
    echo -e "${YELLOW}📋 Simulation mode: CRC check will be disabled${NC}"
    echo ""
fi
# 把 KP/KD scale 转成最终程序参数
for scale in "${MOTOR_KP_SCALES[@]}"; do
    EXTRA_ARGS+=("--motor-kp-scale" "$scale")
done
for scale in "${MOTOR_KD_SCALES[@]}"; do
    EXTRA_ARGS+=("--motor-kd-scale" "$scale")
done

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================

echo -e "${BLUE}[Step 1/4]${NC} Checking prerequisites..."

# Check for TensorRT
# 检查TensorRT，首先检查有没有设置TensorRT_ROOT环境变量，如果没有设置就提示用户设置，并且给出下载地址
# 并且检查$HOME/TensorRT目录是否存在，如果存在就临时设置TensorRT_ROOT为$HOME/TensorRT
if [ -z "$TensorRT_ROOT" ]; then
    echo -e "${YELLOW}⚠️  TensorRT_ROOT is not set.${NC}"
    echo "   Please ensure TensorRT is installed and add to your ~/.bashrc:"
    echo "   export TensorRT_ROOT=\$HOME/TensorRT"
    echo ""
    echo "   Get TensorRT from: https://developer.nvidia.com/tensorrt/download/10x"
    
    # Check if it exists in common locations
    if [ -d "$HOME/TensorRT" ]; then
        echo -e "${GREEN}   Found TensorRT at ~/TensorRT - setting temporarily${NC}"
        export TensorRT_ROOT="$HOME/TensorRT"
    fi
fi

# Check for required model files
# 检查一个文件是否存在，如果不存在就提示用户缺少文件
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}❌ Missing file: $1${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Found: $1${NC}"
        return 0
    fi
}

echo ""
echo "Checking required model files..."
MISSING_FILES=0
# 检查部署需要的四个文件是否存在，encoder和decoder和planner和obs_config
check_file "$CHECKPOINT_DECODER" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$CHECKPOINT_ENCODER" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$OBS_CONFIG" || MISSING_FILES=$((MISSING_FILES + 1))
check_file "$PLANNER" || MISSING_FILES=$((MISSING_FILES + 1))

if [ -d "$MOTION_DATA" ]; then
    echo -e "${GREEN}✅ Found: $MOTION_DATA${NC}"
else
    echo -e "${RED}❌ Missing directory: $MOTION_DATA${NC}"
    MISSING_FILES=$((MISSING_FILES + 1))
fi

if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some files are missing. Make sure you have pulled the model files.${NC}"
    echo "   You may need to run: git lfs pull"
fi

echo ""

# ============================================================================
# Step 2: Install Dependencies (if needed)
# ============================================================================
# 第二步检查just
echo -e "${BLUE}[Step 2/4]${NC} Checking/Installing dependencies..."

# Check if just is installed
# 检查系统有没有just，just可以理解为就是make的任务执行工具，如果没有just这里直接执行scripts/install_deps.sh安装依赖
if ! command -v just &> /dev/null; then
    echo "Installing dependencies (just not found)..."
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
else
    echo -e "${GREEN}✅ just is already installed${NC}"
fi

# Check if other essential tools are available
DEPS_OK=true
# 再检查cmake、clang、git是否安装，如果没有安装就提示用户安装，并且执行scripts/install_deps.sh安装依赖
for cmd in cmake clang git; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${YELLOW}⚠️  $cmd not found, will run install_deps.sh${NC}"
        DEPS_OK=false
        break
    fi
done

if [ "$DEPS_OK" = false ]; then
    echo "Installing missing dependencies..."
    chmod +x scripts/install_deps.sh
    ./scripts/install_deps.sh
else
    echo -e "${GREEN}✅ All essential tools are installed${NC}"
fi

echo ""

# ============================================================================
# Step 3: Setup Environment & Build
# ============================================================================

echo -e "${BLUE}[Step 3/4]${NC} Setting up environment and building..."

# Source the environment setup script
echo "Sourcing environment setup..."
set +e  # Temporarily allow errors (for jetson_clocks on non-Jetson systems)
# 加载部署环境变量，主要是设置TensorRT_ROOT和LD_LIBRARY_PATH
source scripts/setup_env.sh
set -e  # Re-enable exit on error

# Always build to ensure we have the latest version
echo "Building the project..."
# 类似于make，执行just build命令，编译部署程序，我们知道make是根据Makefile文件来编译程序，而just是根据.justfile文件来编译程序
# .justfile文件中定义了各种任务和依赖关系
just build

echo ""

# ============================================================================
# Step 4: Deploy
# ============================================================================
# 打印最终 Deployment Configuration，便于用户确认
echo -e "${BLUE}[Step 4/4]${NC} Ready to deploy!"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                         DEPLOYMENT CONFIGURATION                       ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Environment:        ${GREEN}$ENV_TYPE${NC}"
echo -e "  Network Interface:  ${GREEN}$TARGET${NC}"
echo -e "  Decoder Model:      ${GREEN}$CHECKPOINT_DECODER${NC}"
echo -e "  Encoder Model:      ${GREEN}$CHECKPOINT_ENCODER${NC}"
echo -e "  Motion Data:        ${GREEN}$MOTION_DATA${NC}"
echo -e "  Obs Config:         ${GREEN}$OBS_CONFIG${NC}"
echo -e "  Planner:            ${GREEN}$PLANNER${NC}"
echo -e "  Input Type:         ${GREEN}$INPUT_TYPE${NC}"
echo -e "  Output Type:        ${GREEN}$OUTPUT_TYPE${NC}"
echo -e "  ZMQ Host:           ${GREEN}$ZMQ_HOST${NC}"
if (( ${#EXTRA_ARGS[@]} > 0 )); then
printf -v EXTRA_ARGS_DISPLAY ' %q' "${EXTRA_ARGS[@]}"
echo -e "  Extra Args:         ${GREEN}${EXTRA_ARGS_DISPLAY# }${NC}"
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}The following command will be executed:${NC}"
echo ""
echo -e "${BLUE}just run g1_deploy_onnx_ref $TARGET $CHECKPOINT_DECODER $MOTION_DATA \\${NC}"
echo -e "${BLUE}    --obs-config $OBS_CONFIG \\${NC}"
echo -e "${BLUE}    --encoder-file $CHECKPOINT_ENCODER \\${NC}"
echo -e "${BLUE}    --planner-file $PLANNER \\${NC}"
echo -e "${BLUE}    --input-type $INPUT_TYPE \\${NC}"
echo -e "${BLUE}    --output-type $OUTPUT_TYPE \\${NC}"
echo -e "${BLUE}    --zmq-host $ZMQ_HOST${NC}"
if (( ${#EXTRA_ARGS[@]} > 0 )); then
echo -e "${BLUE}    ${EXTRA_ARGS_DISPLAY# }${NC}"
fi
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Ask for confirmation
if [[ "$ENV_TYPE" == "real" ]]; then
    echo -e "${YELLOW}⚠️  WARNING: This will start the REAL robot control system!${NC}"
else
    echo -e "${YELLOW}📋 This will start the simulation control system.${NC}"
fi
echo ""
read -p "$(echo -e ${GREEN}Proceed with deployment? [Y/n]: ${NC})" confirm

if [[ "$confirm" =~ ^[Yy]$ ]] || [[ -z "$confirm" ]]; then
    echo ""
    echo -e "${GREEN}🚀 Starting deployment...${NC}"
    echo ""
    # 这句话就是整个部署的核心命令，执行just run g1_deploy_onnx_ref，后面跟着各种参数
    just run g1_deploy_onnx_ref "$TARGET" "$CHECKPOINT_DECODER" "$MOTION_DATA" \
        --obs-config "$OBS_CONFIG" \
        --encoder-file "$CHECKPOINT_ENCODER" \
        --planner-file "$PLANNER" \
        --input-type "$INPUT_TYPE" \
        --output-type "$OUTPUT_TYPE" \
        --zmq-host "$ZMQ_HOST" \
        "${EXTRA_ARGS[@]}"
else
    echo ""
    echo -e "${YELLOW}Deployment cancelled.${NC}"
    exit 0
fi
