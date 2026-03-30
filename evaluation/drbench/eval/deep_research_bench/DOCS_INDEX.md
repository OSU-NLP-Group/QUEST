# 文档索引 - LiteLLM 多模型支持

## 🎯 我应该看哪个文档？

### 我是新用户，第一次使用
👉 **[QUICK_START.md](QUICK_START.md)** - 5分钟快速上手

### 我想了解支持哪些模型
👉 **[README_LITELLM.md](README_LITELLM.md)** - 支持的模型列表和快速参考

### 我需要配置特定的模型
👉 **[MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md)** - 详细的配置指南

### 我想看实际的使用例子
👉 **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - 完整的使用示例

### 我是老用户，从 Gemini 迁移
👉 **[LITELLM_MIGRATION.md](LITELLM_MIGRATION.md)** - 迁移指南

### 我想知道改了什么
👉 **[CHANGELOG_LITELLM.md](CHANGELOG_LITELLM.md)** - 完整的改造日志

### 我想看配置示例
👉 **[config_examples.sh](config_examples.sh)** - 8种典型配置

---

## 📚 所有文档列表

### 📖 用户文档

| 文档 | 说明 | 适合人群 | 阅读时间 |
|------|------|---------|---------|
| [QUICK_START.md](QUICK_START.md) | 快速入门指南 | 新用户 | 5分钟 |
| [README_LITELLM.md](README_LITELLM.md) | 总览和快速参考 | 所有用户 | 3分钟 |
| [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) | 详细配置指南 | 需要配置的用户 | 10分钟 |
| [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) | 使用示例集合 | 所有用户 | 15分钟 |
| [LITELLM_MIGRATION.md](LITELLM_MIGRATION.md) | 迁移指南 | 老用户 | 5分钟 |

### 📋 参考文档

| 文档 | 说明 | 适合人群 | 阅读时间 |
|------|------|---------|---------|
| [CHANGELOG_LITELLM.md](CHANGELOG_LITELLM.md) | 改造完整日志 | 开发者/维护者 | 10分钟 |
| [config_examples.sh](config_examples.sh) | 配置示例代码 | 所有用户 | 浏览 |
| [DOCS_INDEX.md](DOCS_INDEX.md) | 本文档 | 所有用户 | 2分钟 |

---

## 🛠️ 脚本工具列表

### 运行脚本

| 脚本 | 功能 | 适合场景 | 难度 |
|------|------|---------|------|
| `run_with_models.sh` | 简化的模型配置运行 | 日常使用 | ⭐ 简单 |
| `run_multiple_models.sh` | 多模型对比运行 | 研究对比 | ⭐⭐ 中等 |
| `run_benchmark.sh` | 完整 RACE + FACT 评估 | 完整评估 | ⭐⭐ 中等 |

### 测试脚本

| 脚本 | 功能 | 何时使用 | 难度 |
|------|------|---------|------|
| `test_models.py` | 测试单个模型 | 验证配置 | ⭐ 简单 |
| `test_all_models.sh` | 测试所有模型 | 初次设置 | ⭐ 简单 |

---

## 🎯 按场景查找

### 场景 1: 我是第一次使用
1. 📖 [QUICK_START.md](QUICK_START.md) - 了解如何开始
2. 🧪 运行 `./test_all_models.sh` - 测试配置
3. 🚀 运行 `./run_with_models.sh` - 开始评估

### 场景 2: 我想用 GPT-4
1. 📖 [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - 查看 OpenAI 部分
2. 💡 [config_examples.sh](config_examples.sh) - 查看示例 2
3. 🚀 运行命令或修改脚本

### 场景 3: 我想用 Azure OpenAI
1. 📖 [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - 查看 Azure 部分
2. 💡 [config_examples.sh](config_examples.sh) - 查看示例 6
3. ⚙️ 设置 3 个环境变量
4. 🚀 运行

### 场景 4: 我想对比多个模型
1. 📖 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - 查看对比示例
2. 🔄 编辑 `run_multiple_models.sh`
3. 🚀 运行对比

### 场景 5: 我想节省成本
1. 📖 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - 查看成本优化部分
2. 💡 [config_examples.sh](config_examples.sh) - 查看示例 3 或 5
3. 🚀 使用 `--limit` 和便宜模型

### 场景 6: 遇到错误
1. 📖 [QUICK_START.md](QUICK_START.md) - 查看故障排除
2. 📖 [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - 查看配置说明
3. 🧪 运行 `python test_models.py --model "your-model"`

---

## 🔍 按问题查找

| 问题 | 查看文档 | 章节 |
|------|---------|------|
| 如何安装？ | QUICK_START.md | Prerequisites |
| 支持哪些模型？ | README_LITELLM.md | 支持的模型 |
| 如何设置 API Key？ | MODEL_CONFIGURATION.md | 各提供商配置 |
| API Key 找不到？ | QUICK_START.md | Troubleshooting |
| 如何节省成本？ | USAGE_EXAMPLES.md | Cost Optimization |
| 速率限制怎么办？ | USAGE_EXAMPLES.md | Troubleshooting |
| Azure 如何配置？ | MODEL_CONFIGURATION.md | Azure OpenAI |
| AWS Bedrock？ | MODEL_CONFIGURATION.md | AWS Bedrock |
| 向后兼容吗？ | LITELLM_MIGRATION.md | Backward Compatibility |
| 改了哪些文件？ | CHANGELOG_LITELLM.md | 文件清单 |

---

## 📱 快速命令参考

### 测试配置
```bash
./test_all_models.sh
python test_models.py --model "gpt-4"
```

### 简单运行
```bash
./run_with_models.sh
```

### 完整命令
```bash
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-4" \
    --clean_model "gpt-3.5-turbo"
```

### 成本优化
```bash
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-3.5-turbo" \
    --limit 10 \
    --only_en
```

### 模型对比
```bash
./run_multiple_models.sh
```

---

## 🎓 推荐阅读顺序

### 新用户 (30分钟)
1. [README_LITELLM.md](README_LITELLM.md) - 3分钟了解概况
2. [QUICK_START.md](QUICK_START.md) - 5分钟快速开始
3. [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - 10分钟学习配置
4. [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - 10分钟看示例
5. 实际操作 - 2分钟测试运行

### 老用户 (10分钟)
1. [LITELLM_MIGRATION.md](LITELLM_MIGRATION.md) - 5分钟了解变化
2. [README_LITELLM.md](README_LITELLM.md) - 3分钟快速参考
3. [config_examples.sh](config_examples.sh) - 2分钟浏览示例

### 研究人员 (45分钟)
1. [CHANGELOG_LITELLM.md](CHANGELOG_LITELLM.md) - 10分钟了解技术细节
2. [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - 15分钟详细配置
3. [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - 15分钟高级用法
4. [config_examples.sh](config_examples.sh) - 5分钟示例代码

---

## 💡 提示

- 所有 Shell 脚本都可执行，使用前运行 `chmod +x *.sh`
- 所有文档都是 Markdown 格式，可在 GitHub 或任何文本编辑器查看
- 建议先运行测试脚本验证配置，再进行完整评估
- 使用 `--limit` 参数先小规模测试，避免浪费资源

---

**需要帮助？** 按上面的场景或问题索引快速找到相关文档！
