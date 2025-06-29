我来为您详细翻译这个知识图谱大数据处理文档，并提供详细的中文注释。让我先读取完整的文档内容：

Read file: Java面试/25_Big_data_with_KG - 副本.ipynb
由于文档内容很长，我将分段为您翻译。让我创建一个完整的中文翻译版本：

让我分段创建翻译文档。首先创建第一部分：


由于文档内容非常长，无法一次性全部输出。下面我将分段为您翻译和注释，先给出前几部分的翻译和详细中文注释，后续如需继续请告知。

---

## 端到端流程：大数据与知识图谱（参考书籍）

### 目标
将关于科技公司收购的新闻文章转化为结构化的知识图谱，利用现代信息抽取、精炼和推理技术，整个流程遵循一本概念性书籍中的基础原则。

### 数据集：CNN/DailyMail

### 方法概览
本notebook分为多个阶段：
1. **数据获取与准备**：获取并清洗原始新闻文本。
2. **信息抽取**：识别关键实体（如组织、人物、金额、日期）及其关系（如“收购”、“投资”）。
3. **知识图谱构建**：将抽取的信息结构化为RDF三元组，形成KG的节点和边。
4. **图谱优化（概念性）**：用嵌入表示KG组件，探索链接预测等知识发现方法。
5. **持久化与利用**：存储、查询（SPARQL）和可视化知识图谱。

我们将结合大语言模型（LLM）和传统NLP库（如spaCy）进行复杂的实体和关系抽取，使用rdflib进行知识图谱管理。

---

### 目录

- 端到端流程：大数据与知识图谱（参考书籍）
  - 初始设置：导入与配置
    - 初始化LLM客户端与spaCy模型
    - 定义RDF命名空间
- 阶段1：数据获取与准备
  - 步骤1.1：数据获取
    - 执行数据获取
  - 步骤1.2：数据清洗与预处理
    - 执行数据清洗
- 阶段2：信息抽取
  - 步骤2.1：实体抽取（命名实体识别NER）
    - spaCy实体探索
    - LLM实体类型选择
    - LLM目标实体抽取
  - 步骤2.2：关系抽取
- 阶段3：知识图谱构建
  - 步骤3.1：实体消歧与链接
  - 步骤3.2：本体对齐
  - 步骤3.3：三元组生成
- 阶段4：图谱优化与嵌入
  - 步骤4.1：生成KG嵌入
  - 步骤4.2：链接预测
  - 步骤4.3：添加预测链接
- 阶段5：持久化与利用
  - 步骤5.1：知识图谱存储
  - 步骤5.2：SPARQL查询与分析
  - 步骤5.3：可视化
- 结论与未来工作

---

### 初始设置：导入与配置

**理论说明：**
在进行任何数据处理或分析前，需要设置好环境，包括：
- **导入库**：如datasets（数据加载）、openai（LLM）、spacy（NLP）、rdflib（知识图谱）、re（正则）、json、matplotlib/pyvis（可视化）、os/collections/tqdm（系统与进度条）。
- **API配置**：设置外部服务的凭证和端点，尤其是Nebius LLM API。生产环境下API密钥应通过环境变量或安全管理系统管理。
- **模型初始化**：加载spaCy的英文模型`en_core_web_sm`，配置LLM客户端用于生成和嵌入。
- **命名空间定义**：RDF知识图谱中，命名空间用于避免命名冲突，保证实体和属性的唯一性，符合Linked Data原则。

---

#### 代码（含详细中文注释）：

```python
# 导入必要的库
import os
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import time

# NLP与知识图谱相关库
import spacy
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD, SKOS # SKOS用于可选标签

# OpenAI客户端（用于LLM）
from openai import OpenAI

# 可视化
from pyvis.network import Network

# Hugging Face数据集库
from datasets import load_dataset

# 嵌入相似度
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- API配置（重要：请替换为你的真实API密钥和模型名）---
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "your_nebius_api_key_here") # 建议用环境变量管理
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"

# --- 模型名（重要：请替换为你实际部署的模型名）---
TEXT_GEN_MODEL_NAME = "deepseek-ai/DeepSeek-V3" # 例如phi-4、deepseek等
EMBEDDING_MODEL_NAME = "BAAI/bge-multilingual-gemma2" # 例如text-embedding-ada-002等

print("库导入完成。")
```

---

如需继续后续内容（如数据获取、清洗、实体识别、关系抽取、知识图谱构建等），请回复“继续”或指定需要的部分。

继续为您翻译和注释后续内容：

---

#### 初始化LLM客户端与spaCy模型

**理论说明：**
这里我们实例化主要的NLP工具客户端：
- **OpenAI客户端**：配置指向Nebius API，用于发送请求到部署的LLM进行实体抽取、关系抽取和生成嵌入。会检查配置参数是否设置。
- **spaCy模型**：加载`en_core_web_sm`，spaCy的小型英文模型，提供分词、词性标注、词形还原和基础命名实体识别功能，用于初始文本探索，可补充LLM方法。

```python
client = None # 初始化客户端为None
if NEBIUS_API_KEY != "YOUR_NEBIUS_API_KEY" and NEBIUS_BASE_URL != "YOUR_NEBIUS_BASE_URL" and TEXT_GEN_MODEL_NAME != "YOUR_TEXT_GENERATION_MODEL_NAME":
    try:
        client = OpenAI(
            base_url=NEBIUS_BASE_URL,
            api_key=NEBIUS_API_KEY 
        )
        print(f"OpenAI客户端已初始化，base_url: {NEBIUS_BASE_URL}，使用模型: {TEXT_GEN_MODEL_NAME}")
    except Exception as e:
        print(f"初始化OpenAI客户端时出错: {e}")
        client = None # 确保初始化失败时客户端为None
else:
    print("警告: OpenAI客户端未完全配置。LLM功能将被禁用。请设置NEBIUS_API_KEY、NEBIUS_BASE_URL和TEXT_GEN_MODEL_NAME。")

nlp_spacy = None # 初始化nlp_spacy为None
try:
    nlp_spacy = spacy.load("en_core_web_sm")
    print("spaCy模型'en_core_web_sm'已加载。")
except OSError:
    print("spaCy模型'en_core_web_sm'未找到。正在下载...（可能需要一些时间）")
    try:
        spacy.cli.download("en_core_web_sm")
        nlp_spacy = spacy.load("en_core_web_sm")
        print("spaCy模型'en_core_web_sm'下载并加载成功。")
    except Exception as e:
        print(f"下载或加载spaCy模型失败: {e}")
        print("请在终端中尝试: python -m spacy download en_core_web_sm 然后重启内核。")
        nlp_spacy = None # 确保加载失败时nlp_spacy为None
```

---

#### 定义RDF命名空间

**理论说明：**
在RDF中，命名空间用于避免命名冲突并为术语（URI）提供上下文：
- `EX`：项目特定的自定义命名空间（如我们的实体和关系，如果未映射到标准本体）。
- `SCHEMA`：指Schema.org，互联网上广泛使用的结构化数据词汇表。我们将尝试将一些抽取的类型映射到Schema.org术语以提高互操作性。
- `RDFS`：RDF Schema，提供描述RDF词汇表的基本词汇（如`rdfs:label`、`rdfs:Class`）。
- `RDF`：核心RDF词汇（如`rdf:type`）。
- `XSD`：XML Schema数据类型，用于指定字面量数据类型（如`xsd:string`、`xsd:date`）。
- `SKOS`：简单知识组织系统，用于叙词表、分类法和受控词汇表（如`skos:altLabel`用于替代名称）。

```python
EX = Namespace("http://example.org/kg/")
SCHEMA = Namespace("http://schema.org/")

print(f"自定义命名空间EX定义为: {EX}")
print(f"Schema.org命名空间SCHEMA定义为: {SCHEMA}")
```

---

## 阶段1：数据获取与准备
**（参考：第1章-大数据生态系统；第3章-大数据处理价值链）**

**理论（阶段概览）：**
这个初始阶段在任何数据驱动项目中都至关重要。它对应大数据价值链的早期阶段："数据获取"和"数据准备/预处理"的部分。目标是获取原始数据并将其转换为适合进一步处理和信息抽取的状态。低质量的输入数据（"垃圾进，垃圾出"原则）必然导致低质量的知识图谱。

---

### 步骤1.1：数据获取
**任务：** 收集新闻文章集合。

**书籍概念：** （第1章，图1和2；第3章-数据获取阶段）
这一步代表大数据生态系统的"数据源"和"摄取"组件。我们使用现有数据集（通过Hugging Face `datasets`的CNN/DailyMail）而不是抓取实时新闻，但原理相同：将外部数据引入我们的处理管道。

**方法：**
我们将定义函数`acquire_articles`来加载CNN/DailyMail数据集。为了管理演示的处理时间和成本，并专注于潜在相关的文章，此函数将：
1. 加载数据集的指定分割（如'train'）。
2. 可选地基于关键词列表过滤文章。对于我们的科技公司收购目标，关键词如"acquire"、"merger"、"technology"、"startup"是相关的。这是一个简单的启发式方法；对于更大的数据集，可以使用更高级的主题建模或分类进行更好的过滤。
3. 取（过滤后）文章的小样本。

**输出：** 原始文章数据结构列表（通常是包含'id'、'article'文本等的字典）。

```python
def acquire_articles(dataset_name="cnn_dailymail", version="3.0.0", split='train', sample_size=1000, keyword_filter=None):
    """
    从指定的Hugging Face数据集加载文章，可选地过滤它们，并取样本。
    """
    print(f"尝试加载数据集: {dataset_name} (版本: {version}, 分割: '{split}')...")
    try:
        full_dataset = load_dataset(dataset_name, version, split=split, streaming=False) # 使用streaming=False便于小数据集切片
        print(f"成功加载数据集。分割中的总记录数: {len(full_dataset)}")
    except Exception as e:
        print(f"加载数据集{dataset_name}时出错: {e}")
        print("请确保数据集可用或你有网络连接。")
        return [] # 失败时返回空列表
    
    raw_articles_list = []
    if keyword_filter:
        print(f"过滤包含关键词的文章: {keyword_filter}...")
        # 这是简单的关键词搜索。对于非常大的数据集，这可能很慢。
        # 如果不使用流式处理，考虑使用Hugging Face数据集的.filter()方法以提高效率。
        count = 0
        # 为了避免在数据集很大且只需要小样本时遍历整个数据集：
        # 我们将遍历到某个点或直到有足够的过滤文章。
        # 这是在潜在大数据集上平衡过滤和性能的启发式方法。
        iteration_limit = min(len(full_dataset), sample_size * 20) # 最多查看20倍sample_size的文章
        for i in tqdm(range(iteration_limit), desc="过滤文章"):
            record = full_dataset[i]
            if any(keyword.lower() in record['article'].lower() for keyword in keyword_filter):
                raw_articles_list.append(record)
                count += 1
            if count >= sample_size:
                print(f"在前{i+1}条记录中找到{sample_size}篇符合过滤条件的文章。")
                break
        if not raw_articles_list:
            print(f"警告: 在前{iteration_limit}条记录中未找到包含关键词{keyword_filter}的文章。返回空列表。")
            return []
        # 如果找到文章但少于sample_size，我们取找到的。
        # 如果找到更多，我们仍然只取sample_size。
        raw_articles_list = raw_articles_list[:sample_size]
    else:
        print(f"不进行关键词过滤，取前{sample_size}篇文章。")
        # 确保sample_size不超过数据集长度
        actual_sample_size = min(sample_size, len(full_dataset))
        raw_articles_list = list(full_dataset.select(range(actual_sample_size)))
        
    print(f"获取了{len(raw_articles_list)}篇文章。")
    return raw_articles_list

print("函数'acquire_articles'已定义。")
```

---

#### 执行数据获取

**理论：**
现在我们调用`acquire_articles`函数。我们定义与目标相关的关键词（科技公司收购）来指导过滤过程。设置`SAMPLE_SIZE`来保持演示的数据量可管理。较小的样本允许更快的迭代，特别是使用LLM时，可能有关联的成本和延迟。

```python
# 定义与科技公司收购相关的关键词
ACQUISITION_KEYWORDS = ["acquire", "acquisition", "merger", "buyout", "purchased by", "acquired by", "takeover"]
TECH_KEYWORDS = ["technology", "software", "startup", "app", "platform", "digital", "AI", "cloud"]

# 对于这个演示，我们主要按收购相关术语过滤。
# 技术方面将通过LLM在实体/关系抽取期间的提示来加强。
FILTER_KEYWORDS = ACQUISITION_KEYWORDS

SAMPLE_SIZE = 10 # 保持很小以便在这个演示notebook中快速LLM处理

# 将raw_data_sample初始化为空列表
raw_data_sample = [] 
raw_data_sample = acquire_articles(sample_size=SAMPLE_SIZE, keyword_filter=FILTER_KEYWORDS)

if raw_data_sample: # 检查列表是否不为空
    print(f"\n原始获取文章的示例 (ID: {raw_data_sample[0]['id']}):")
    print(raw_data_sample[0]['article'][:500] + "...")
    print(f"\n记录中的字段数: {len(raw_data_sample[0].keys())}")
    print(f"字段: {list(raw_data_sample[0].keys())}")
else:
    print("未获取到文章。涉及文章处理的后续步骤可能会被跳过或产生无输出。")
```

---

### 步骤1.2：数据清洗与预处理
**任务：** 执行基本文本标准化。

**书籍概念：** （第3章-大数据的多样性挑战）
来自新闻文章等源的原始文本数据通常很混乱。它可能包含HTML标签、样板内容（如署名、版权声明）、特殊字符和不一致的格式。这一步对应解决大数据的"多样性"（在某种程度上是"真实性"）挑战。干净、标准化的输入对于有效的下游NLP任务至关重要，因为噪声会显著降低实体识别器和关系抽取器的性能。

**方法：**
我们将定义函数`clean_article_text`，使用正则表达式（`re`模块）来：
- 移除常见新闻样板（如"(CNN) --"、特定署名模式）。
- 移除HTML标签和URL。
- 标准化空白字符（如用单个空格替换多个空格/换行符）。
- 可选地，处理可能干扰LLM处理或JSON格式的引号或其他特殊字符。

**输出：** 字典列表，每个字典包含文章ID和其清洗后的文本。

```python
def clean_article_text(raw_text):
    """
    使用正则表达式清洗新闻文章的原始文本。
    """
    text = raw_text
    
    # 移除(CNN)样式前缀
    text = re.sub(r'^\(CNN\)\s*(--)?\s*', '', text)
    # 移除常见署名和发布/更新行（模式可能需要针对特定数据集细微差别进行调整）
    text = re.sub(r'By .*? for Dailymail\.com.*?Published:.*?Updated:.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'PUBLISHED:.*?BST,.*?UPDATED:.*?BST,.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'Last updated at.*on.*', '', text, flags=re.IGNORECASE)
    # 移除URL
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 移除邮件地址
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # 标准化空白字符：用单个空格替换换行符、制表符，然后用单个空格替换多个空格
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    # 可选：如果LLM有问题则转义引号，但对于好的模型通常不需要
    # text = text.replace('"', "\\\"").replace("'", "\\'") 
    return text

print("函数'clean_article_text'已定义。")
```

---

#### 执行数据清洗

**理论：**
这个块遍历`raw_data_sample`（在前一步中获取）。对于每篇文章，它调用`clean_article_text`函数。清洗后的文本与原始文章ID和潜在的其他有用字段（如'summary'，如果数据集中有'highlights'）一起存储在名为`cleaned_articles`的新列表中。这个新列表将是后续信息抽取阶段的主要输入。

```python
cleaned_articles = [] # 初始化为空列表

if raw_data_sample: # 仅在raw_data_sample不为空时继续
    print(f"正在清洗{len(raw_data_sample)}篇获取的文章...")
    for record in tqdm(raw_data_sample, desc="清洗文章"):
        cleaned_text_content = clean_article_text(record['article'])
        cleaned_articles.append({
            "id": record['id'],
            "original_text": record['article'], # 保留原始文本以供参考
            "cleaned_text": cleaned_text_content,
            "summary": record.get('highlights', '') # CNN/DM有'highlights'作为摘要
        })
    print(f"清洗完成。清洗后的文章总数: {len(cleaned_articles)}。")
    if cleaned_articles: # 检查处理后列表是否不为空
        print(f"\n清洗后文章的示例 (ID: {cleaned_articles[0]['id']}):")
        print(cleaned_articles[0]['cleaned_text'][:500] + "...")
else:
    print("前一步中未获取到原始文章，因此跳过清洗。")
```

---

如需继续后续内容（如实体识别、关系抽取、知识图谱构建等），请回复"继续"。

