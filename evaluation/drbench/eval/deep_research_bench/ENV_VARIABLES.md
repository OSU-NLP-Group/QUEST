# 环境变量配置参考

本文档列出了所有支持的模型提供商所需的环境变量。

## 📋 环境变量清单

### Google Gemini

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

**使用示例:**
```bash
export GEMINI_API_KEY="AIzaSy..."
EVAL_MODEL="gemini/gemini-2.5-pro-preview-06-05"
CLEAN_MODEL="gemini/gemini-2.5-flash-preview-05-20"
```

---

### OpenAI

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**使用示例:**
```bash
export OPENAI_API_KEY="sk-..."
EVAL_MODEL="gpt-4"
CLEAN_MODEL="gpt-3.5-turbo"
```

---

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**使用示例:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
EVAL_MODEL="claude-3-opus-20240229"
CLEAN_MODEL="claude-3-sonnet-20240229"
```

---

### DeepSeek

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

**使用示例:**
```bash
export DEEPSEEK_API_KEY="sk-..."
EVAL_MODEL="deepseek/deepseek-chat"
CLEAN_MODEL="deepseek/deepseek-chat"
```

---

### Azure OpenAI

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
```

**使用示例:**
```bash
export AZURE_API_KEY="abc123..."
export AZURE_API_BASE="https://myresource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
EVAL_MODEL="azure/gpt-4-deployment"
CLEAN_MODEL="azure/gpt-35-turbo-deployment"
```

**注意事项:**
- `AZURE_API_BASE` 必须以 `/` 结尾
- 模型名称使用你的**部署名称**，不是模型名称
- 部署名称在 Azure Portal 中可以找到

---

### AWS Bedrock

```bash
export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_REGION_NAME="us-east-1"
```

**使用示例:**
```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="wJalr..."
export AWS_REGION_NAME="us-east-1"
EVAL_MODEL="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
CLEAN_MODEL="bedrock/anthropic.claude-3-haiku-20240307-v1:0"
```

**注意事项:**
- 确保 IAM 用户有调用 Bedrock 的权限
- 模型需要在指定区域可用
- 常用区域: `us-east-1`, `us-west-2`

---

## 🔧 快速配置模板

### 配置单个提供商

复制以下模板到你的 shell 配置文件 (`~/.bashrc` 或 `~/.zshrc`):

```bash
# === LLM Provider Configuration ===

# Google Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# Azure OpenAI
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_REGION_NAME="us-east-1"
```

### 临时配置（仅当前会话）

```bash
# 直接在命令行设置
export GEMINI_API_KEY="your-key"

# 或者在运行脚本前设置
GEMINI_API_KEY="your-key" ./run_with_models.sh
```

### 使用 .env 文件（推荐）

创建 `.env` 文件:
```bash
# .env
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key

# Azure OpenAI
AZURE_API_KEY=your-azure-api-key
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview

# AWS Bedrock
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION_NAME=us-east-1
```

然后加载:
```bash
# 加载 .env 文件
set -a
source .env
set +a

# 运行脚本
./run_with_models.sh
```

---

## ✅ 验证配置

### 检查环境变量是否设置

```bash
# 检查单个变量
echo $GEMINI_API_KEY
echo $OPENAI_API_KEY

# 检查所有变量
env | grep -E "GEMINI|OPENAI|ANTHROPIC|DEEPSEEK|AZURE|AWS"
```

### 测试配置

```bash
# 测试单个模型
python test_models.py --model "gemini/gemini-2.5-pro-preview-06-05"

# 测试所有已配置的模型
./test_all_models.sh
```

---

## 🔒 安全建议

### 1. 不要将 API Keys 提交到 Git

添加到 `.gitignore`:
```gitignore
.env
*.key
*_credentials
```

### 2. 使用环境变量而非硬编码

❌ **错误:**
```bash
EVAL_MODEL="gpt-4"
# API key 硬编码在脚本中
```

✅ **正确:**
```bash
export OPENAI_API_KEY="your-key"  # 在外部设置
EVAL_MODEL="gpt-4"
```

### 3. 定期轮换 API Keys

- 定期更换 API keys
- 为不同项目使用不同的 keys
- 使用具有最小权限的 keys

### 4. 使用密钥管理服务

企业用户考虑使用:
- AWS Secrets Manager
- Azure Key Vault
- HashiCorp Vault

---

## 📊 环境变量优先级

litellm 按以下顺序查找配置:

1. **代码中直接传入的 API key** (最高优先级)
   ```python
   AIClient(api_key="your-key")
   ```

2. **模型特定的环境变量**
   ```bash
   AZURE_API_KEY, GEMINI_API_KEY, etc.
   ```

3. **通用环境变量** (如果支持)
   ```bash
   API_KEY
   ```

---

## 🌐 区域配置

### AWS Bedrock 可用区域

Claude 模型可用区域:
- `us-east-1` (美国东部)
- `us-west-2` (美国西部)
- `eu-west-1` (欧洲)
- `ap-southeast-1` (亚太)

### Azure OpenAI 端点格式

```bash
# 格式
https://<resource-name>.openai.azure.com/

# 示例
https://mycompany-openai.openai.azure.com/
```

---

## 🔗 获取 API Keys

### Google Gemini
- 访问: https://aistudio.google.com/apikey
- 创建新的 API key
- 复制并保存

### OpenAI
- 访问: https://platform.openai.com/api-keys
- 创建新的 secret key
- 立即保存（只显示一次）

### Anthropic Claude
- 访问: https://console.anthropic.com/
- 前往 API Keys 部分
- 创建新 key

### DeepSeek
- 访问: https://platform.deepseek.com/
- 注册并获取 API key

### Azure OpenAI
- 在 Azure Portal 中创建 OpenAI 资源
- 获取 Key 和 Endpoint
- 创建模型部署

### AWS Bedrock
- 在 AWS Console 中启用 Bedrock
- 创建 IAM 用户
- 获取 Access Key 和 Secret Key

---

## 📞 常见问题

### Q: 可以同时设置多个提供商吗？
**A:** 可以！设置所有你需要的环境变量，然后通过 `--eval_model` 参数选择使用哪个。

### Q: 环境变量在哪里设置？
**A:** 可以在:
- Shell 配置文件 (`~/.bashrc`, `~/.zshrc`)
- `.env` 文件
- 命令行临时设置
- CI/CD 环境变量

### Q: Azure 的部署名称在哪里找？
**A:** Azure Portal → Your OpenAI Resource → Model deployments

### Q: AWS 区域如何选择？
**A:** 选择延迟最低的区域，确保该区域支持你需要的 Bedrock 模型。

---

**更多信息请参考:**
- [QUICK_START.md](QUICK_START.md) - 快速入门
- [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) - 详细配置
- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - 使用示例
