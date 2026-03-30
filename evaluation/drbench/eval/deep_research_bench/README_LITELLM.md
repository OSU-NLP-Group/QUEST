# LiteLLM 多模型支持 - 使用指南

## 🎯 改造完成

DeepResearch Benchmark 现已通过 litellm 支持多种 LLM 提供商。

## ✅ 支持的模型

| 提供商 | 模型格式 | 环境变量 |
|--------|---------|----------|
| Google Gemini | `gemini/gemini-2.5-pro-preview-06-05` | `GEMINI_API_KEY` |
| OpenAI | `gpt-4`, `gpt-3.5-turbo` | `OPENAI_API_KEY` |
| Anthropic | `claude-3-opus-20240229` | `ANTHROPIC_API_KEY` |
| DeepSeek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| Azure OpenAI | `azure/<deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| AWS Bedrock | `bedrock/anthropic.claude-3-sonnet-20240229-v1:0` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` |

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 设置 API Key
```bash
# 选择一个提供商
export GEMINI_API_KEY="your-key"        # 或
export OPENAI_API_KEY="your-key"        # 或
export ANTHROPIC_API_KEY="your-key"     # 等等
```

### 3. 运行
```bash
# 方式 A: 使用配置脚本
./run_with_models.sh

# 方式 B: 直接命令
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-4" \
    --clean_model "gpt-3.5-turbo"
```

## 📁 新增文件

### Shell 脚本
- **`run_with_models.sh`** - 简单模型配置脚本 ⭐推荐
- **`run_multiple_models.sh`** - 多模型对比运行
- **`test_all_models.sh`** - 批量测试配置
- **`test_models.py`** - 单模型测试

### 文档
- **`QUICK_START.md`** - 快速入门指南
- **`MODEL_CONFIGURATION.md`** - 详细配置说明
- **`USAGE_EXAMPLES.md`** - 使用示例
- **`LITELLM_MIGRATION.md`** - 迁移指南

## 💡 使用场景

### 场景 1: 测试配置
```bash
./test_all_models.sh
```

### 场景 2: 成本优化
```bash
python deepresearch_bench_race.py "model" \
    --eval_model "gpt-3.5-turbo" \
    --limit 10
```

### 场景 3: 生产评估
```bash
python deepresearch_bench_race.py "model" \
    --eval_model "gpt-4" \
    --clean_model "gpt-3.5-turbo"
```

### 场景 4: 模型对比
```bash
./run_multiple_models.sh
```

## 🔧 常用参数

```bash
--eval_model MODEL      # 评分模型
--clean_model MODEL     # 清洗模型
--max_workers N         # 并发数 (默认: 5)
--limit N               # 限制任务数
--only_zh               # 只处理中文
--only_en               # 只处理英文
--force                 # 强制重新评估
```

## 📚 详细文档

- 快速入门: [QUICK_START.md](QUICK_START.md)
- 详细配置: [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md)
- 使用示例: [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
- 迁移指南: [LITELLM_MIGRATION.md](LITELLM_MIGRATION.md)

## ✨ 核心特性

✅ **向后兼容** - 原有脚本无需修改
✅ **灵活切换** - 一条命令切换模型
✅ **成本优化** - 评估和清洗使用不同模型
✅ **企业支持** - Azure 和 AWS Bedrock
✅ **易于测试** - 多种测试工具
