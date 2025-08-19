
## �� 多节点分布式训练脚本详细解读

这个 `train_on_multi_nodes.sh` 脚本是一个完整的多节点分布式训练解决方案，分为两个主要部分：**环境准备** 和 **SLURM集群配置**。

## 🏗️ **第一部分：环境准备和编译**

### **1. GCC/G++ 版本更新**
```bash
# 更新 GCC 和 G++ 版本（如需）
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7 g++-7
# 更新系统的默认 gcc 和 g++ 指向
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc
```

**作用**：
- **兼容性**：确保编译环境与DeepSpeed兼容
- **性能优化**：使用较新的编译器版本
- **依赖满足**：满足DeepSpeed的编译要求

### **2. DeepSpeed 源码编译**
```bash
# 源代码安装 DeepSpeed
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="7.5" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 
python setup.py build_ext -j8 bdist_wheel
```

#### **关键编译参数**：

| 参数 | 作用 | 说明 |
|------|------|------|
| `TORCH_CUDA_ARCH_LIST="7.5"` | GPU架构版本 | 根据实际GPU设置（RTX 2080 Ti等） |
| `DS_BUILD_CPU_ADAM=1` | CPU Offload支持 | 启用优化器状态CPU卸载 |
| `DS_BUILD_UTILS=1` | NVMe Offload支持 | 启用NVMe SSD参数卸载 |
| `-j8` | 并行编译 | 使用8个线程加速编译 |

#### **GPU架构对应表**：
```
RTX 2080 Ti, RTX 2080, RTX 2070: 7.5
RTX 3080, RTX 3070: 8.6
RTX 4080, RTX 4070: 8.9
RTX 4090: 8.9
V100: 7.0
A100: 8.0
```

### **3. 生成Wheel包**
```bash
# 运行将生成类似于dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl的文件，
# 在其他节点安装：pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl。
```

**优势**：
- **一致性**：确保所有节点使用相同版本的DeepSpeed
- **离线安装**：避免网络依赖问题
- **版本控制**：精确控制DeepSpeed版本

### **4. Transformers 安装**
```bash
# 源代码安装 Transformers
pip install git+https://github.com/huggingface/transformers
```

**作用**：
- **最新特性**：获取最新的Transformers功能
- **兼容性**：确保与DeepSpeed的兼容性
- **Bug修复**：包含最新的修复

## 🚀 **第二部分：SLURM集群配置**

### **SLURM 作业脚本模板**
```bash
#SBATCH --job-name=test-nodes        # 作业名称
#SBATCH --nodes=2                    # 节点数量
#SBATCH --ntasks-per-node=1          # 每个节点的任务数（关键：每个节点只1个任务！）
#SBATCH --cpus-per-task=10           # 每个任务的CPU核心数
#SBATCH --gres=gpu:8                 # GPU数量
#SBATCH --time 20:00:00              # 最大执行时间（HH:MM:SS）
#SBATCH --output=%x-%j.out           # 输出文件名
```

#### **SLURM参数详解**：

| 参数 | 作用 | 示例值 | 说明 |
|------|------|--------|------|
| `--job-name` | 作业名称 | `test-nodes` | 用于识别和管理作业 |
| `--nodes` | 节点数量 | `2` | 参与训练的节点数 |
| `--ntasks-per-node` | 每节点任务数 | `1` | **关键**：分布式训练必须为1 |
| `--cpus-per-task` | 每任务CPU数 | `10` | 数据预处理和系统开销 |
| `--gres=gpu` | GPU数量 | `8` | 每个节点的GPU数量 |
| `--time` | 最大运行时间 | `20:00:00` | 防止作业无限运行 |
| `--output` | 输出文件 | `%x-%j.out` | `%x`=作业名，`%j`=作业ID |

### **环境变量设置**
```bash
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
```

#### **环境变量详解**：

| 变量 | 作用 | 说明 |
|------|------|------|
| `GPUS_PER_NODE` | 每节点GPU数 | 与SLURM的`--gres=gpu`保持一致 |
| `MASTER_ADDR` | 主节点地址 | 自动获取第一个节点的主机名 |
| `MASTER_PORT` | 通信端口 | 避免端口冲突，使用非标准端口 |

### **分布式训练启动命令**
```bash
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

#### **PyTorch分布式参数**：

| 参数 | 作用 | 说明 |
|------|------|------|
| `--nproc_per_node` | 每节点进程数 | 等于GPU数量 |
| `--nnodes` | 总节点数 | SLURM自动设置 |
| `--node_rank` | 节点排名 | SLURM自动设置 |
| `--master_addr` | 主节点地址 | 环境变量设置 |
| `--master_port` | 通信端口 | 环境变量设置 |

## �� **实际使用流程**

### **步骤1：环境准备**
```bash
# 在编译节点执行
bash train_on_multi_nodes.sh  # 执行第一部分
```

### **步骤2：分发Wheel包**
```bash
# 将生成的wheel包分发到所有节点
scp dist/deepspeed-*.whl node1:/tmp/
scp dist/deepspeed-*.whl node2:/tmp/
# 在所有节点安装
pip install /tmp/deepspeed-*.whl
```

### **步骤3：创建SLURM脚本**
```bash
# 创建launch.slurm文件
cat > launch.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=translation-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
 translation/run_translation.py \
 --deepspeed config/ds_config_zero2.json \
 --model_name_or_path t5-large \
 --per_device_train_batch_size 4 \
 --output_dir output_dir --overwrite_output_dir \
 --do_train --do_eval \
 --max_train_samples 500 --num_train_epochs 1 \
 --dataset_name wmt16 --dataset_config "ro-en" \
 --source_lang en --target_lang ro'
EOF
```

### **步骤4：提交作业**
```bash
sbatch launch.slurm
```

## �� **配置示例对比**

### **小规模训练（2节点×8GPU）**：
```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
export GPUS_PER_NODE=8
```

### **中等规模训练（4节点×8GPU）**：
```bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
export GPUS_PER_NODE=8
```

### **大规模训练（8节点×8GPU）**：
```bash
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
export GPUS_PER_NODE=8
```

## ⚠️ **重要注意事项**

### **1. 编译环境一致性**
- 所有节点必须使用相同的CUDA版本
- 编译节点和目标节点架构应一致
- 确保Python版本兼容

### **2. 网络配置**
- 节点间网络带宽充足
- 防火墙允许指定端口通信
- 确保节点间SSH免密登录

### **3. 资源分配**
- `--ntasks-per-node=1` 是分布式训练的关键
- GPU数量要与实际硬件匹配
- 预留足够的CPU资源用于数据预处理

### **4. 故障排除**
- 检查SLURM作业状态：`squeue -u $USER`
- 查看作业日志：`tail -f %x-%j.out`
- 验证节点连接：`scontrol show nodes`

这个脚本提供了一个完整的、生产级别的多节点分布式训练解决方案，适用于大规模模型训练场景。
