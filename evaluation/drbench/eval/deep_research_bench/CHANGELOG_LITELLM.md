# LiteLLM 多模型支持改造日志

## 📅 改造日期
2026-01-24

## 🎯 改造目标
将 DeepResearch Benchmark 从仅支持 Google Gemini API 改造为通过 litellm 支持多个 LLM 提供商。

## ✅ 改造完成

### 核心代码修改 (3个文件)

#### 1. `utils/api.py`
**改动**: 完全重写 AI 客户端
- ❌ 移除: `from google import genai`
- ✅ 添加: `from litellm import completion`
- ✅ 新增: 多提供商 API 配置支持
  - Google Gemini
  - OpenAI
  - Anthropic Claude
  - DeepSeek
  - Azure OpenAI
  - AWS Bedrock
- ✅ 保持: 原有接口兼容性

#### 2. `requirements.txt`
**改动**: 更新依赖
- ❌ 移除: `google-genai`
- ✅ 添加: `litellm>=1.0.0`
- ✅ 保持: 其他依赖不变

#### 3. `deepresearch_bench_race.py`
**改动**: 添加模型选择参数
- ✅ 新增参数: `--eval_model` (评分模型)
- ✅ 新增参数: `--clean_model` (清洗模型)
- ✅ 更新: AIClient 初始化逻辑
- ✅ 保持: 所有原有功能和接口

#### 4. `run_benchmark.sh`
**改动**: 添加模型配置
- ✅ 新增: `EVAL_MODEL` 变量
- ✅ 新增: `CLEAN_MODEL` 变量
- ✅ 更新: 命令行参数传递
- ✅ 保持: 原有 RACE + FACT 评估流程

### 新增 Shell 脚本 (4个)

#### 1. `run_with_models.sh` ⭐
**功能**: 简化的模型配置运行脚本
- 8个预配置的模型提供商选项
- 注释清晰，易于切换
- 包含所有常用参数
- 适合新手使用

**使用**:
```bash
chmod +x run_with_models.sh
./run_with_models.sh
```

#### 2. `run_multiple_models.sh` 🔄
**功能**: 多模型对比评估脚本
- 自动运行多个模型配置
- 生成对比总结报告
- 适合研究和模型对比

**使用**:
```bash
chmod +x run_multiple_models.sh
./run_multiple_models.sh
```

#### 3. `test_all_models.sh` 🧪
**功能**: 批量配置测试脚本
- 测试所有已配置的模型
- 验证 API keys 有效性
- 运行前必备检查

**使用**:
```bash
chmod +x test_all_models.sh
./test_all_models.sh
```

#### 4. `test_models.py` 🔍
**功能**: 单模型测试脚本
- 测试特定模型配置
- 验证 API 连接
- 调试工具

**使用**:
```bash
python test_models.py --model "gpt-4"
```

### 新增文档 (6个)

#### 1. `QUICK_START.md` 🚀
**内容**: 5分钟快速入门指南
- 安装步骤
- 配置示例
- 常见工作流程
- 故障排除

#### 2. `MODEL_CONFIGURATION.md` ⚙️
**内容**: 详细的模型配置指南
- 所有支持的模型列表
- 每个提供商的配置方法
- 环境变量说明
- 具体使用示例

#### 3. `USAGE_EXAMPLES.md` 📚
**内容**: 实用的使用示例
- 各种场景的完整命令
- 成本优化技巧
- 高级用法
- 参数组合建议

#### 4. `LITELLM_MIGRATION.md` 🔄
**内容**: 迁移指南
- 改造前后对比
- 向后兼容性说明
- 迁移步骤
- API 对照表

#### 5. `README_LITELLM.md` 📖
**内容**: LiteLLM 支持总览
- 改造总结
- 快速参考
- 使用场景速查
- 文档索引

#### 6. `config_examples.sh` 💡
**内容**: 配置示例集合
- 8个典型配置示例
- 参数组合建议
- 成本估算参考
- 完整命令示例

## 📊 支持的模型提供商

| 提供商 | 模型数量 | 企业级 | 成本 |
|--------|---------|--------|------|
| Google Gemini | 3+ | ❌ | 中 |
| OpenAI | 5+ | ✅ (Azure) | 中-高 |
| Anthropic | 3+ | ✅ (Bedrock) | 高 |
| DeepSeek | 2+ | ❌ | 低 |
| Azure OpenAI | 多种 | ✅ | 中-高 |
| AWS Bedrock | 多种 | ✅ | 中-高 |

## 🔄 向后兼容性

✅ **100% 向后兼容**
- 所有原有脚本无需修改
- 默认行为保持不变（使用 Gemini）
- 输出格式完全一致
- API 接口保持兼容

## 🆕 新增功能

### 1. 多模型支持
```bash
--eval_model "gpt-4"
--clean_model "gpt-3.5-turbo"
```

### 2. 成本优化
```bash
--limit 10  # 限制任务数
--only_zh   # 只处理中文
--only_en   # 只处理英文
```

### 3. 配置测试
```bash
./test_all_models.sh  # 批量测试
python test_models.py --model "gpt-4"  # 单个测试
```

### 4. 模型对比
```bash
./run_multiple_models.sh  # 自动对比多个模型
```

## 📁 文件清单

### 修改的文件 (4个)
- [x] `utils/api.py` - 核心 API 客户端
- [x] `requirements.txt` - 依赖配置
- [x] `deepresearch_bench_race.py` - 主评估脚本
- [x] `run_benchmark.sh` - 运行脚本

### 新增的文件 (11个)
- [x] `run_with_models.sh` - 简化配置脚本
- [x] `run_multiple_models.sh` - 多模型对比脚本
- [x] `test_all_models.sh` - 批量测试脚本
- [x] `test_models.py` - 单模型测试脚本
- [x] `QUICK_START.md` - 快速入门
- [x] `MODEL_CONFIGURATION.md` - 配置指南
- [x] `USAGE_EXAMPLES.md` - 使用示例
- [x] `LITELLM_MIGRATION.md` - 迁移指南
- [x] `README_LITELLM.md` - 总览文档
- [x] `config_examples.sh` - 配置示例
- [x] `CHANGELOG_LITELLM.md` - 本文档

## 🎓 使用建议

### 新用户
1. 阅读 `QUICK_START.md`
2. 运行 `./test_all_models.sh`
3. 使用 `./run_with_models.sh`

### 现有用户
1. 阅读 `LITELLM_MIGRATION.md`
2. 更新依赖: `pip install -r requirements.txt`
3. 继续使用原有脚本（自动使用 Gemini）

### 研究人员
1. 阅读 `MODEL_CONFIGURATION.md`
2. 使用 `./run_multiple_models.sh` 对比模型
3. 参考 `USAGE_EXAMPLES.md` 优化配置

### 企业用户
1. 查看 Azure/Bedrock 配置
2. 阅读 `config_examples.sh` 示例 6-7
3. 使用企业级部署

## 🔧 技术细节

### API 调用流程
```
用户调用
  ↓
AIClient.generate()
  ↓
根据模型前缀选择配置
  ↓
litellm.completion()
  ↓
返回结果
```

### 模型识别逻辑
```python
if model.startswith("azure/"):      # Azure OpenAI
if model.startswith("bedrock/"):    # AWS Bedrock
if model.startswith("gemini/"):     # Google Gemini
if model.startswith("gpt-"):        # OpenAI
if model.startswith("claude-"):     # Anthropic
if model.startswith("deepseek/"):   # DeepSeek
```

### 环境变量优先级
1. 构造函数传入的 API key
2. 模型特定的环境变量
3. 通用环境变量（如果有）

## 🐛 已知问题

无已知问题。

## 📈 未来计划

- [ ] 添加更多模型提供商
- [ ] 支持自定义 API endpoint
- [ ] 添加请求重试策略配置
- [ ] 支持流式响应
- [ ] 添加成本统计功能

## 🙏 致谢

- LiteLLM 项目提供统一接口
- 原 DeepResearch Bench 团队

## 📞 支持

遇到问题？查看文档：
- 快速入门: `QUICK_START.md`
- 配置问题: `MODEL_CONFIGURATION.md`
- 使用示例: `USAGE_EXAMPLES.md`
- 迁移问题: `LITELLM_MIGRATION.md`

---

**改造完成时间**: 2026-01-24
**改造人员**: Claude Sonnet 4.5
**测试状态**: ✅ 已通过基本测试
