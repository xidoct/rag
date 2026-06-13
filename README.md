# 🎓 毕业论文 RAG 知识问答系统

基于 RAG（检索增强生成）的毕业论文政策问答系统。上传学校官方 PDF/Word 文档后，学生用自然语言提问即可获得准确回答，所有答案严格依据文档原文。

## ✨ 功能

- 📤 **文档导入** — 支持 PDF、Word、ZIP，本地文件夹批量导入无大小限制
- 💬 **智能问答** — 自然语言提问，流式返回，标注引用来源
- 🔍 **混合检索** — 向量语义 + BM25 关键词 + Cross-Encoder 重排序
- 🗂️ **文档管理** — 查看已入库文档、统计、删除
- ⚡ **查询缓存** — 相同问题秒返回

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- Windows / macOS / Linux

### 2. 安装

```bash
# 克隆项目
git clone <your-repo-url>
cd 论文知识查询

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置 API Key

编辑 `.streamlit/secrets.toml`：

```toml
# LLM: DeepSeek（推荐，便宜好用）
LLM_PROVIDER = "deepseek"
DEEPSEEK_API_KEY = "sk-你的DeepSeek-Key"

# Embedding: 本地 BGE 模型（免费离线）
EMBEDDING_PROVIDER = "local"
```

> 💡 DeepSeek API Key 获取: [platform.deepseek.com](https://platform.deepseek.com/)  
> 💡 也支持 Claude API / OpenAI / 硅基流动，详见 `config.py`

### 4. 启动

```bash
streamlit run main.py
```

浏览器访问 `http://localhost:8501`

## 📖 使用教程

### 第一步：导入文档

1. 点击左侧 **📤 文档上传**
2. 选择「📁 本地文件夹导入」（推荐）或「🌐 浏览器上传」
3. 输入文档所在文件夹路径，如 `D:\毕业论文规范\`
4. 点击「🚀 开始导入全部文件」
5. 等待进度条完成，文档自动分块、向量化、入库

> 💡 支持 ZIP 压缩包、PDF、Word (.docx)  
> 💡 系统自动识别标题、表格，按章节智能分块

### 第二步：提问

1. 点击左侧 **💬 知识问答**
2. 输入问题，如：
   - "毕业论文查重率要求多少？"
   - "论文格式有什么具体要求？"
   - "装订要求是什么？"
   - "参考文献格式有什么规定？"
3. 系统会用文档原文回答，并标注来源文件名和页码

### 第三步：管理文档

1. 点击左侧 **📚 文档管理**
2. 查看已入库文档列表和统计
3. 支持按文件删除或清空知识库

## 🏗️ 架构

```
学生提问
    │
    ▼
┌──────────┐
│ 查询缓存  │── 命中 → 直接返回
└────┬─────┘
     未命中
      ▼
┌──────────┐
│ 查询扩展  │  LLM 改写为多角度搜索词
└────┬─────┘
      ▼
┌──────────┐
│ 双路检索  │  向量语义 (BGE) + BM25 关键词 (jieba)
└────┬─────┘
      ▼
┌──────────┐
│ RRF 融合  │  粗排 → top-20 候选
└────┬─────┘
      ▼
┌──────────┐
│ 重排序   │  Cross-Encoder 精排 → top-5
└────┬─────┘
      ▼
┌──────────┐
│ LLM 生成  │  DeepSeek 流式输出 + 来源标注
└──────────┘
```

## 📁 项目结构

```
论文知识查询/
├── main.py                 # Streamlit 入口
├── config.py               # 全局配置
├── requirements.txt        # Python 依赖
├── engine/                 # RAG 引擎层
│   ├── loader.py           #   文档解析 (PDF/Word/ZIP)
│   ├── splitter.py         #   智能分块 (标题/表格识别)
│   ├── embedder.py         #   向量化 (BGE/Voyage/OpenAI)
│   ├── store.py            #   ChromaDB 存储
│   ├── bm25_retriever.py   #   BM25 关键词检索
│   ├── retriever.py        #   混合检索 + 查询扩展
│   ├── reranker.py         #   Cross-Encoder 重排序
│   └── cache.py            #   查询缓存
├── ui/                     # Streamlit 页面
│   ├── upload.py           #   文档上传
│   ├── chat.py             #   知识问答
│   └── manage.py           #   文档管理
└── data/                   # 数据目录 (gitignore)
    ├── chroma_db/          # 向量库
    ├── uploads/            # 原始文件
    └── bm25_index.pkl      # BM25 索引
```

## ⚙️ 可选配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_PROVIDER` | `deepseek` | 也支持 `anthropic`、`siliconflow`、`openai` |
| `EMBEDDING_PROVIDER` | `local` | 也支持 `siliconflow`、`voyage`、`openai` |
| `TOP_K` | `5` | 最终返回文档片段数 |
| `CHUNK_SIZE` | `800` | 分块大小 (字符) |
| `SIMILARITY_THRESHOLD` | `0.3` | 检索相似度阈值 |

## 📄 License

MIT
