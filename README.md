# AI Agents/Agentic Workflow/大模型微调项目集合
# 1.多轮对话 LangGraph Agent智能客服
- 最新个人运行chatbot.ipynb
- chatbot.ipynb

Python/LangGraph/LangChain/langchain-openai/llangchaon-community/langchain-core/TavilySearchAPI/TypedDict/Annotated/PYthon.dispaly/os

**核心功能**:基础对话能力/集成外部工具/人工介入/历史对话查询

# 2.基于crewAI自动化写作应用实战

- 2.基于CrewAI自动化写作应用实战.ipynb
- 额度用完了，Azure配置有点问题，解决：https://github.com/gongfan1213/xiangmu/wiki/2.%E9%A1%B9%E7%9B%AE2AzureAPI.md

Python/CrewAI/Azure OpenAi/Agent/Task/Crew

**核心功能** ：自动化写作系统，需要两个Agent，故事内容构思者和故事内容协作者，两个Task:产出故事内容构思、根据构思编写故事，Crew组合团队并且执行、kickoff运行任务
# RAG

https://github.com/FareedKhan-dev/all-rag-techniques#
# 3.最简单的RAG的极简实现
- 3_simple_rag_副本 (1).ipynb（已经全部跑通的)
- 3_simple_rag.ipynb

# 4.基于百分位数法的语义分块
- 4_语义分块(已经跑通的).ipynb
- 4_semantic_chunking.ipynb


# 可用模型一览表
从提供的数据来看，这些模型的 `permission` 字段中，`allow_create_engine`、`allow_sampling`、`allow_logprobs` 和 `allow_view` 都被设置为 `true`，而 `allow_fine_tuning` 和 `allow_search_indices` 被设置为 `false`。这意味着这些模型可以用于生成文本、计算对数概率等任务，但不能进行微调和搜索索引操作。

以下是这些模型中可以使用的模型列表（按 `owned_by` 分类）：

### **由 `vertex-ai` 拥有的模型**
- `claude-3-5-sonnet-20240620`
- `claude-3-5-sonnet-20241022`
- `claude-3-7-sonnet-20250219`
- `claude-opus-4-20250514`
- `claude-sonnet-4-20250514`
- `gemini-2.0-flash`
- `gemini-2.5-pro-preview-03-25`

### **由 `coze` 拥有的模型**
- `deepseek-r1`
- `deepseek-v3`
- `Pro-DeepSeek-R1`
- `Pro-DeepSeek-V3`
- `qwen-max`

### **由 `custom` 拥有的模型**
- `DeepSeek-R1-Distill-Qwen-32B`
- `DeepSeek-R1-Distill-Qwen-7B`
- `Doubao-1.5-vision-pro-32k`
- `Doubao-lite-128k`
- `Doubao-lite-32k`
- `Doubao-pro-128k`
- `Doubao-pro-32k`
- `gemini-2.0-flash-lite`
- `gemini-2.5-flash-preview-04-17`
- `gemini-2.5-pro-preview-05-06`
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `kimi-latest`
- `kimi-thinking-preview`
- `moonshot-small`
- `moonshot-v1-128k-vision-preview`
- `moonshot-v1-32k-vision-preview`
- `moonshot-v1-8k-vision-preview`
- `moonshot-v1-auto`
- `o3`
- `qwen2.5-72b-instruct`

### **由 `volcengine` 拥有的模型**
- `Doubao-embedding`
- `Doubao-lite-4k`
- `Doubao-pro-4k`

### **由 `openai` 拥有的模型**
- `gpt-4`
- `gpt-4-32k`
- `gpt-4-turbo`
- `gpt-4o`
- `gpt-4o-mini`
- `o1`
- `o1-mini`
- `text-embedding-3-large`
- `text-embedding-3-small`
- `text-embedding-ada-002`

### **由 `ali` 拥有的模型**
- `qwen-plus`
- `qwen-turbo`
- `qwen3-235b-a22b`

这些模型都可以用于生成文本、计算对数概率等任务，但不能进行微调。

