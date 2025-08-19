
## 📋 脚本概览

这个脚本实现了基于 Transformers 库的序列到序列（Seq2Seq）模型训练，专门用于机器翻译任务。它支持多种翻译模型架构，包括 T5、BART、mBART、MarianMT 等。

## ��️ 脚本结构分析

### **1. 导入和依赖**
```python
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    # ... 其他导入
)
```

**核心组件**：
- **dataclasses**：用于定义命令行参数的数据类
- **transformers**：Hugging Face 的核心库
- **datasets**：数据集加载和处理
- **evaluate**：评估指标计算

### **2. 参数定义**

#### **ModelArguments 类**
```python
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(...)
    tokenizer_name: Optional[str] = field(...)
    cache_dir: Optional[str] = field(...)
    use_fast_tokenizer: bool = field(default=True, ...)
    # ... 其他参数
```

**主要参数**：
- `model_name_or_path`：预训练模型路径或标识符
- `config_name`：模型配置文件
- `tokenizer_name`：分词器名称
- `cache_dir`：模型缓存目录
- `use_fast_tokenizer`：是否使用快速分词器

#### **DataTrainingArguments 类**
```python
@dataclass
class DataTrainingArguments:
    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})
    dataset_name: Optional[str] = field(...)
    train_file: Optional[str] = field(...)
    validation_file: Optional[str] = field(...)
    max_source_length: Optional[int] = field(default=1024, ...)
    max_target_length: Optional[int] = field(default=128, ...)
    # ... 其他参数
```

**关键参数**：
- `source_lang`/`target_lang`：源语言和目标语言
- `dataset_name`：数据集名称
- `max_source_length`：输入序列最大长度
- `max_target_length`：目标序列最大长度
- `source_prefix`：T5模型的前缀（如"translate English to German: "）

### **3. 主函数流程**

#### **A. 参数解析和初始化**
```python
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(...)
    
    # 设置随机种子
    set_seed(training_args.seed)
```

#### **B. 数据集加载**
```python
if data_args.dataset_name is not None:
    # 从 Hugging Face Hub 下载数据集
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
else:
    # 从本地文件加载数据集
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    # ... 处理验证和测试文件
    raw_datasets = load_dataset(builder_name, data_files=data_files, ...)
```

**支持的数据格式**：
- Hugging Face Hub 上的公开数据集
- 本地 JSON/JSONL 文件
- 自定义格式的数据

#### **C. 模型和分词器加载**
```python
# 加载配置
config = AutoConfig.from_pretrained(...)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(...)

# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained(...)

# 调整词嵌入大小
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))
```

#### **D. 多语言模型特殊处理**
```python
# 多语言分词器需要设置源语言和目标语言
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
    tokenizer.src_lang = data_args.source_lang
    tokenizer.tgt_lang = data_args.target_lang
    
    # 设置强制开始标记
    forced_bos_token_id = tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token else None
    model.config.forced_bos_token_id = forced_bos_token_id
```

### **4. 数据预处理**

#### **预处理函数**
```python
def preprocess_function(examples):
    # 提取源语言和目标语言文本
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    
    # 添加前缀（T5模型需要）
    inputs = [prefix + inp for inp in inputs]
    
    # 对输入进行分词
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    
    # 对目标进行分词
    labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
    
    # 处理填充标记
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

**预处理步骤**：
1. **文本提取**：从翻译对中提取源语言和目标语言文本
2. **前缀添加**：为T5模型添加任务前缀
3. **分词处理**：对输入和目标文本进行分词
4. **长度控制**：截断或填充到指定长度
5. **标签处理**：将填充标记替换为-100（忽略损失计算）

#### **数据集处理**
```python
if training_args.do_train:
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
```

### **5. 数据整理器**

```python
# 数据整理器
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
if data_args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
```

**数据整理器作用**：
- 将批次数据整理成模型输入格式
- 处理动态填充
- 支持混合精度训练

### **6. 评估指标**

```python
# 加载BLEU评估指标
metric = evaluate.load("sacrebleu", cache_dir=model_args.cache_dir)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # 替换填充标记
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 后处理
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    # 计算BLEU分数
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    
    # 计算生成长度
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
```

### **7. 训练器初始化**

```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
)
```

### **8. 训练和评估**

#### **训练阶段**
```python
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # 保存模型和分词器
    
    # 记录训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
```

#### **评估阶段**
```python
if training_args.do_eval:
    logger.info("*** Evaluate ***")
    
    max_length = training_args.generation_max_length or data_args.val_max_target_length
    num_beams = data_args.num_beams or training_args.generation_num_beams
    
    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
```

#### **预测阶段**
```python
if training_args.do_predict:
    logger.info("*** Predict ***")
    
    predict_results = trainer.predict(
        predict_dataset, 
        metric_key_prefix="predict", 
        max_length=max_length, 
        num_beams=num_beams
    )
    
    # 保存预测结果
    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions = [pred.strip() for pred in predictions]
            
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                writer.write("\n".join(predictions))
```

## �� 脚本特点

### **1. 高度可配置**
- 支持多种模型架构
- 灵活的数据源配置
- 丰富的训练参数

### **2. 生产就绪**
- 完整的日志记录
- 检查点保存和恢复
- 分布式训练支持

### **3. 多语言支持**
- 自动处理多语言分词器
- 支持语言代码设置
- 强制开始标记配置

### **4. 评估完善**
- BLEU分数计算
- 生成长度统计
- 预测结果保存

## �� 使用示例

```bash
# 基本训练命令
python run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name wmt16 \
    --dataset_config_name de-en \
    --output_dir ./output \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

这个脚本是一个完整的、企业级的翻译模型训练解决方案，支持从数据加载到模型部署的全流程。
