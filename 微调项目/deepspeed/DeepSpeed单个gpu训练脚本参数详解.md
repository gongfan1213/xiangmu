
## �� DeepSpeed 单GPU训练脚本参数详解

这个脚本包含了4个不同的训练配置示例，让我逐一详细讲解每个参数的含义和作用。

### 🚀 **示例1：T5-Small + ZeRO-2**

```bash
deepspeed --num_gpus=1 translation/run_translation.py \
--deepspeed config/ds_config_zero2.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

#### **DeepSpeed 启动参数**：
- `--num_gpus=1`：指定使用1个GPU进行训练

#### **DeepSpeed 配置参数**：
- `--deepspeed config/ds_config_zero2.json`：指定ZeRO-2优化配置文件

#### **模型参数**：
- `--model_name_or_path t5-small`：使用T5-Small预训练模型（60M参数）

#### **批次大小参数**：
- `--per_device_train_batch_size 1`：每个GPU的训练批次大小为1

#### **输出参数**：
- `--output_dir output_dir`：模型输出目录
- `--overwrite_output_dir`：覆盖已存在的输出目录

#### **精度参数**：
- `--fp16`：启用FP16混合精度训练

#### **训练控制参数**：
- `--do_train`：执行训练
- `--max_train_samples 500`：限制训练样本数量为500个（用于快速测试）
- `--num_train_epochs 1`：训练1个epoch

#### **数据集参数**：
- `--dataset_name wmt16`：使用WMT16数据集
- `--dataset_config "ro-en"`：使用罗马尼亚语-英语配置

#### **语言参数**：
- `--source_lang en`：源语言为英语
- `--target_lang ro`：目标语言为罗马尼亚语

### 🚀 **示例2：T5-Large + ZeRO-2**

```bash
deepspeed --num_gpus=1 translation/run_translation.py \
--deepspeed config/ds_config_zero2.json \
--model_name_or_path t5-large \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--output_dir output_dir --overwrite_output_dir \
--do_train \
--do_eval \
--max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

#### **主要差异**：
- `--model_name_or_path t5-large`：使用T5-Large模型（770M参数）
- `--per_device_train_batch_size 4`：批次大小增加到4
- `--per_device_eval_batch_size 4`：添加评估批次大小
- `--do_eval`：同时执行评估
- **移除了 `--fp16`**：T5-Large在ZeRO-2下可能不需要FP16

### �� **示例3：T5-3B + ZeRO-3**

```bash
deepspeed --num_gpus=1 translation/run_translation.py \
--deepspeed config/ds_config_zero3.json \
--model_name_or_path t5-3b --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

#### **关键差异**：
- `--deepspeed config/ds_config_zero3.json`：使用ZeRO-3优化
- `--model_name_or_path t5-3b`：使用T5-3B模型（3B参数）
- `--per_device_train_batch_size 1`：批次大小回到1（大模型需要）
- **重新启用 `--fp16`**：3B模型需要FP16节省内存

### 🚀 **示例4：Python直接启动（不使用DeepSpeed）**

```bash
python translation/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

#### **主要差异**：
- **使用 `python` 而不是 `deepspeed`**：不使用DeepSpeed优化
- `--source_prefix "translate English to Romanian: "`：T5模型必需的前缀
- `--dataset_config_name ro-en`：使用 `_name` 后缀
- `--predict_with_generate`：启用生成预测
- `--output_dir tmp/tst-translation`：不同的输出目录

## �� **参数分类详解**

### **1. DeepSpeed 相关参数**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `--num_gpus` | 指定GPU数量 | `1` |
| `--deepspeed` | DeepSpeed配置文件 | `config/ds_config_zero2.json` |

### **2. 模型参数**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `--model_name_or_path` | 预训练模型路径 | `t5-small`, `t5-large`, `t5-3b` |
| `--source_prefix` | T5模型任务前缀 | `"translate English to Romanian: "` |

### **3. 批次大小参数**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `--per_device_train_batch_size` | 每GPU训练批次大小 | `1`, `4` |
| `--per_device_eval_batch_size` | 每GPU评估批次大小 | `4` |

### **4. 训练控制参数**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `--do_train` | 执行训练 | 无值（标志） |
| `--do_eval` | 执行评估 | 无值（标志） |
| `--max_train_samples` | 最大训练样本数 | `500` |
| `--num_train_epochs` | 训练轮数 | `1` |

### **5. 数据集参数**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `--dataset_name` | 数据集名称 | `wmt16` |
| `--dataset_config` | 数据集配置 | `"ro-en"` |
| `--dataset_config_name` | 数据集配置名称 | `ro-en` |

### **6. 语言参数**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `--source_lang` | 源语言代码 | `en` |
| `--target_lang` | 目标语言代码 | `ro` |

### **7. 输出参数**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `--output_dir` | 输出目录 | `output_dir`, `tmp/tst-translation` |
| `--overwrite_output_dir` | 覆盖输出目录 | 无值（标志） |

### **8. 精度参数**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `--fp16` | 启用FP16混合精度 | 无值（标志） |

### **9. 评估参数**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `--predict_with_generate` | 启用生成预测 | 无值（标志） |

## �� **参数选择策略**

### **模型大小 vs 批次大小**：
- **T5-Small (60M)**：批次大小 1-4
- **T5-Large (770M)**：批次大小 1-4
- **T5-3B (3B)**：批次大小 1

### **优化策略选择**：
- **小模型**：可以使用普通Python启动
- **中等模型**：推荐ZeRO-2
- **大模型**：必须使用ZeRO-3

### **内存优化**：
- **FP16**：适用于大模型或内存受限情况
- **批次大小**：根据GPU内存调整
- **样本限制**：用于快速测试和调试

## 🔧 **实际使用建议**

1. **首次运行**：使用示例1（T5-Small + ZeRO-2）进行测试
2. **生产训练**：根据模型大小选择合适的优化策略
3. **内存不足**：减小批次大小或启用FP16
4. **快速迭代**：使用 `--max_train_samples` 限制样本数

这些参数配置展示了从简单到复杂的完整训练流程，可以根据实际需求和硬件条件进行调整。
