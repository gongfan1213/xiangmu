https://github.com/FareedKhan-dev/all-rag-techniques#


# 所有RAG技术：更简单、更实用的方法 ✨

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Nebius AI](https://img.shields.io/badge/Nebius%20AI-API-brightgreen)](https://cloud.nebius.ai/services/llm-embedding) [![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/) [![Medium](https://img.shields.io/badge/Medium-Blog-black?logo=medium)](https://medium.com/@fareedkhandev/testing-every-rag-technique-to-find-the-best-094d166af27f)

本仓库采用清晰、实用的方法来处理**检索增强生成（RAG）**，将先进技术分解为直观、易懂的实现。我们不依赖`LangChain`或`FAISS`等框架，而是使用熟悉的Python库`openai`、`numpy`、`matplotlib`和其他几个库来构建一切。

目标很简单：提供可读、可修改且具有教育意义的代码。通过专注于基础知识，这个项目帮助揭开RAG的神秘面纱，让您更容易理解它的真正工作原理。

## 更新：📢
- (2025年5月12日) 新增了关于如何使用知识图谱处理大数据的笔记本。
- (2025年4月27日) 新增了一个笔记本，用于为给定查询找到最佳RAG技术（简单RAG + 重排序器 + 查询重写）。
- (2025年3月20日) 新增了关于强化学习RAG的笔记本。
- (2025年3月7日) 向仓库添加了20种RAG技术。

## 🚀 包含内容

本仓库包含一系列Jupyter笔记本，每个都专注于特定的RAG技术。每个笔记本提供：

*   技术的简明解释。
*   从头开始的逐步实现。
*   带有内联注释的清晰代码示例。
*   评估和比较以展示技术的有效性。
*   可视化结果。

以下是涵盖技术的概览：

| 笔记本                                      | 描述                                                                                                                                                         |
| :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [1. 简单RAG](1_simple_rag.ipynb)           | 基础RAG实现。一个很好的起点！                                                                                                       |
| [2. 语义分块](2_semantic_chunking.ipynb) | 基于语义相似性分割文本，创建更有意义的块。                                                                                           |
| [3. 块大小选择器](3_chunk_size_selector.ipynb) | 探索不同块大小对检索性能的影响。                                                                                    |
| [4. 上下文增强RAG](4_context_enriched_rag.ipynb) | 检索相邻块以提供更多上下文。                                                                                                     |
| [5. 上下文块标题](5_contextual_chunk_headers_rag.ipynb) | 在嵌入前为每个块添加描述性标题。                                                                                                |
| [6. 文档增强RAG](6_doc_augmentation_rag.ipynb) | 从文本块生成问题以增强检索过程。                                                                                           |
| [7. 查询转换](7_query_transform.ipynb)   | 重写、扩展或分解查询以改善检索。包括**Step-back Prompting**和**子查询分解**。                                      |
| [8. 重排序器](8_reranker.ipynb)               | 使用LLM重新排序初始检索结果以获得更好的相关性。                                                                                       |
| [9. RSE](9_rse.ipynb)                         | 相关段提取：识别并重建连续的文本段，保持上下文。                                                   |
| [10. 上下文压缩](10_contextual_compression.ipynb) | 实现上下文压缩以过滤和压缩检索的块，最大化相关信息。                                                 |
| [11. 反馈循环RAG](11_feedback_loop_rag.ipynb) | 整合用户反馈以学习和改进RAG系统。                                                                                      |
| [12. 自适应RAG](12_adaptive_rag.ipynb)     | 根据查询类型动态选择最佳检索策略。                                                                                          |
| [13. 自监督RAG](13_self_rag.ipynb)             | 实现Self-RAG，动态决定何时以及如何检索，评估相关性，并评估支持和效用。                                        |
| [14. 命题分块](14_proposition_chunking.ipynb) | 将文档分解为原子、事实性陈述以进行精确检索。                                                                                      |
| [15. 多模态RAG](15_multimodel_rag.ipynb)   | 结合文本和图像进行检索，使用LLaVA为图像生成标题。                                                                  |
| [16. 融合RAG](16_fusion_rag.ipynb)         | 结合向量搜索和基于关键词的检索（BM25）以改善结果。                                                                                |
| [17. 图RAG](17_graph_rag.ipynb)           | 将知识组织为图，实现对相关概念的遍历。                                                                                        |
| [18. 层次RAG](18_hierarchy_rag.ipynb)        | 构建层次索引（摘要 + 详细块）以实现高效检索。                                                                                   |
| [19. HyDE RAG](19_HyDE_rag.ipynb)             | 使用假设文档嵌入来改善语义匹配。                                                                                              |
| [20. CRAG](20_crag.ipynb)                     | 纠正RAG：动态评估检索质量并使用网络搜索作为后备。                                                                           |
| [21. 强化学习RAG](21_rag_with_rl.ipynb)                     | 使用强化学习最大化RAG模型的奖励。                                                                           |
| [最佳RAG查找器](best_rag_finder.ipynb)     | 使用简单RAG + 重排序器 + 查询重写为给定查询找到最佳RAG技术。                                                                        |
| [22. 知识图谱大数据](22_Big_data_with_KG.ipynb) | 使用知识图谱处理大型数据集。                                                                                                                     |

## 🗂️ 仓库结构

```
fareedkhan-dev-all-rag-techniques/
├── README.md                          <- 您在这里！
├── 1_simple_rag.ipynb
├── 2_semantic_chunking.ipynb
├── 3_chunk_size_selector.ipynb
├── 4_context_enriched_rag.ipynb
├── 5_contextual_chunk_headers_rag.ipynb
├── 6_doc_augmentation_rag.ipynb
├── 7_query_transform.ipynb
├── 8_reranker.ipynb
├── 9_rse.ipynb
├── 10_contextual_compression.ipynb
├── 11_feedback_loop_rag.ipynb
├── 12_adaptive_rag.ipynb
├── 13_self_rag.ipynb
├── 14_proposition_chunking.ipynb
├── 15_multimodel_rag.ipynb
├── 16_fusion_rag.ipynb
├── 17_graph_rag.ipynb
├── 18_hierarchy_rag.ipynb
├── 19_HyDE_rag.ipynb
├── 20_crag.ipynb
├── 21_rag_with_rl.ipynb
├── 22_big_data_with_KG.ipynb
├── best_rag_finder.ipynb
├── requirements.txt                   <- Python依赖项
└── data/
    └── val.json                       <- 示例验证数据（查询和答案）
    └── AI_Information.pdf             <- 用于测试的示例PDF文档。
    └── attention_is_all_you_need.pdf  <- 用于测试的示例PDF文档（用于多模态RAG）。
```

## 🛠️ 开始使用

1.  **克隆仓库：**

    ```bash
    git clone https://github.com/FareedKhan-dev/all-rag-techniques.git
    cd all-rag-techniques
    ```

2.  **安装依赖项：**

    ```bash
    pip install -r requirements.txt
    ```

3.  **设置您的OpenAI API密钥：**

    *   从[Nebius AI](https://studio.nebius.com/)获取API密钥。
    *   将API密钥设置为环境变量：
        ```bash
        export OPENAI_API_KEY='YOUR_NEBIUS_AI_API_KEY'
        ```
        或
        ```bash
        setx OPENAI_API_KEY "YOUR_NEBIUS_AI_API_KEY"  # Windows系统
        ```
        或者，在您的Python脚本/笔记本中：

        ```python
        import os
        os.environ["OPENAI_API_KEY"] = "YOUR_NEBIUS_AI_API_KEY"
        ```

4.  **运行笔记本：**

    使用Jupyter Notebook或JupyterLab打开任何Jupyter笔记本（`.ipynb`文件）。每个笔记本都是自包含的，可以独立运行。笔记本设计为在每个文件内按顺序执行。

    **注意：** `data/AI_Information.pdf`文件提供了用于测试的示例文档。您可以用自己的PDF替换它。`data/val.json`文件包含用于评估的示例查询和理想答案。
    `attention_is_all_you_need.pdf`用于测试多模态RAG笔记本。

## 💡 核心概念

*   **嵌入：** 捕获语义含义的文本数值表示。我们使用Nebius AI的嵌入API，在许多笔记本中也使用`BAAI/bge-en-icl`嵌入模型。

*   **向量存储：** 用于存储和搜索嵌入的简单数据库。我们使用NumPy创建自己的`SimpleVectorStore`类以进行高效的相似性计算。

*   **余弦相似性：** 两个向量之间相似性的度量。更高的值表示更大的相似性。

*   **分块：** 将文本分割成更小、可管理的片段。我们探索各种分块策略。

*   **检索：** 为给定查询找到最相关文本块的过程。

*   **生成：** 使用大型语言模型（LLM）基于检索的上下文和用户的查询创建响应。我们通过Nebius AI的API使用`meta-llama/Llama-3.2-3B-Instruct`模型。

*   **评估：** 评估RAG系统响应的质量，通常通过将其与参考答案进行比较或使用LLM对相关性进行评分。

## 🤝 贡献

欢迎贡献！如果您想添加新的RAG技术或改进现有实现，请：

1. Fork这个仓库
2. 创建一个功能分支
3. 提交您的更改
4. 打开一个Pull Request

## 📝 许可证

这个项目在MIT许可证下发布。有关详细信息，请参阅LICENSE文件。

## 🙏 致谢

感谢所有为RAG技术发展做出贡献的研究人员和开发者。这个项目旨在让这些技术更容易理解和实现。

---

**注意：** 这个项目主要用于教育和学习目的。在生产环境中使用之前，请确保适当测试和验证所有实现。 
