# Langfuse Session 轨迹导出工具

`export_langfuse_session_llm_io.py` 用于从 Langfuse 项目批量导出 OpenCode 会话轨迹，并转换为适合后训练的数据格式。

脚本只使用 Python 标准库，通过 Langfuse Public API 读取数据，不需要安装额外的 Python 依赖。

## 导出逻辑

脚本会执行以下操作：

1. 分页获取当前 Langfuse 项目中的 session；指定 `--session-id` 时只保留完全匹配的 session，并在找到后停止分页。
2. 对每个 session 获取时间最新的 trace。
3. 在该 trace 中查找最后一个名称为 `opencode.llm`、类型为 `GENERATION` 或 `SPAN` 的 observation。
4. 指定 `--user-id` 时，检查该 observation 的 `metadata.attributes["langfuse.user.id"]` 是否完全匹配。
5. 合并 observation 的 `input.messages` 和 `output` 消息，并删除两者首尾重叠的消息。
6. 将 OpenCode 消息转换为后训练轨迹格式。
7. 将每个 session 写入一个独立 JSON 文件。

没有 trace，或者最新 trace 中没有目标 observation 的 session 会被跳过。数据结构不完整、API 请求失败或文件写入失败的 session 会被计为失败。

## 环境要求

- Python 3.10 或更高版本
- 可访问目标 Langfuse 服务
- 目标项目的 Public Key 和 Secret Key

可以在 Langfuse 项目设置中创建 API Key。请勿将 Secret Key 写入代码或提交到版本库。

## 快速开始

在 Langfuse 源码仓库根目录运行：

```bash
export LANGFUSE_HOST="http://localhost:3000"
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."

python3 scripts/export_langfuse_session_llm_io.py
```

默认输出目录为：

```text
langfuse_session_exports/
```

也可以直接通过命令行传入连接信息和输出目录：

```bash
python3 scripts/export_langfuse_session_llm_io.py \
  --host "http://localhost:3000" \
  --public-key "pk-lf-..." \
  --secret-key "sk-lf-..." \
  --output-dir "langfuse_session_exports"
```

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--host` | `LANGFUSE_HOST` 或 `http://localhost:3000` | Langfuse 服务地址，也可以包含 `/api/public` 后缀 |
| `--public-key` | `LANGFUSE_PUBLIC_KEY` | Langfuse 项目 Public Key，必填 |
| `--secret-key` | `LANGFUSE_SECRET_KEY` | Langfuse 项目 Secret Key，必填 |
| `--output-dir` | `langfuse_session_exports` | JSON 输出目录 |
| `--observation-name` | `opencode.llm` | 需要导出的 observation 名称 |
| `--session-id` | 无 | 只导出 ID 完全匹配的 session |
| `--user-id` | 无 | 只导出最终目标 observation 的 `langfuse.user.id` 完全匹配的轨迹 |
| `--from-timestamp` | 无 | 只遍历在该 ISO 8601 时间点及之后创建的 session |
| `--to-timestamp` | 无 | 只遍历在该 ISO 8601 时间点之前创建的 session |
| `--environment` | 无 | 只遍历指定环境；可以重复传入多次 |
| `--page-size` | `100` | 每页获取的 session 数量，范围为 1～100 |
| `--timeout` | `30` | 单次 HTTP 请求超时时间，单位为秒 |
| `--retries` | `3` | 连接失败、HTTP 429 或 HTTP 5xx 的重试次数 |
| `--fail-fast` | 关闭 | 第一个 session 导出失败时立即停止 |

筛选示例：

```bash
python3 scripts/export_langfuse_session_llm_io.py \
  --from-timestamp "2026-07-01T00:00:00Z" \
  --to-timestamp "2026-08-01T00:00:00Z" \
  --environment "production" \
  --environment "staging"
```

按 session ID 和 user ID 筛选：

```bash
python3 scripts/export_langfuse_session_llm_io.py \
  --session-id "ses_0910bdc2affe6RoekxHS1y792v" \
  --user-id "user-123"
```

两个 ID 都采用区分大小写的精确匹配。与时间、environment 筛选同时使用时，所有条件按 AND 组合。`--user-id` 只检查最终将被导出的 `opencode.llm` observation；attribute 缺失或不匹配时，该 session 会被跳过，不会生成文件，也不会计为失败。

查看完整帮助：

```bash
python3 scripts/export_langfuse_session_llm_io.py --help
```

## 输出文件

每个成功导出的 session 对应一个文件：

```text
langfuse_session_exports/<安全化后的-session-id>.json
```

当 session ID 只包含安全的文件名字符时，文件名就是原始 session ID，例如：

```text
langfuse_session_exports/ses_0910bdc2affe6RoekxHS1y792v.json
```

如果 session ID 包含 `/` 等不适合文件名的字符，脚本会生成经过清理并附带哈希的文件名。JSON 中的 `session_id` 始终来自当前遍历的 Langfuse session，不会从文件名反推。

输出结构示例：

```json
{
  "session_id": "ses_0910bdc2affe6RoekxHS1y792v",
  "model": "glm-5",
  "provider": "alibaba-cn",
  "messages": [
    {
      "role": "user",
      "content": "分析当前项目"
    },
    {
      "role": "assistant",
      "content": "我先检查项目结构。",
      "reasoning_content": "需要先读取目录和关键配置。",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "bash",
            "arguments": "{\"command\":\"ls\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "README.md\nscripts",
      "tool_call_id": "call_123"
    }
  ],
  "tools": []
}
```

### `model` 和 `provider` 的来源

- `model`：目标 observation 顶层的 `model` 字段。
- `provider`：优先读取 `metadata.attributes["llm.provider"]`，其次读取 `metadata.attributes["llm.system"]`；同时兼容 metadata 顶层的 `llm.provider`、`llm.system` 和 `provider`。

目标 observation 缺少有效的 `model` 或 `provider` 时，该 session 会导出失败，不会生成不完整的训练数据。

### 消息转换规则

- `system`、`user`：保留 `role` 和 `content`。
- `assistant`：
  - `reasoning` 内容块合并到 `reasoning_content`。
  - `tool_use` 内容块转换为 OpenAI 风格的 `tool_calls`。
  - 工具参数统一转换为 JSON 字符串。
- `tool`：将 OpenCode 的 `toolCallId` 转换为 `tool_call_id`。
- `tools`：保留 observation input 中的工具定义；不存在时输出空数组。

文件使用 UTF-8 编码并采用原子替换方式写入，避免中途失败留下半个 JSON 文件。脚本还会清理旧版本产生的同名 session 子目录中的 `input.json`、`output.json` 和旧合并文件；只有目录为空时才会删除该目录。

## 运行结果与退出码

运行过程中会输出每个 session 的处理结果，最后打印汇总：

```text
Done: total=10, exported=8, skipped=1, failed=1
```

- `exported`：成功生成轨迹文件。
- `skipped`：没有 trace、最新 trace 中没有目标 observation，或最终目标 observation 不匹配 `--user-id`。
- `failed`：API、数据格式或文件操作失败。

全部 session 成功或仅有跳过项时退出码为 `0`；只要存在失败项，退出码为 `1`。

## 测试

在仓库根目录运行：

```bash
python3 -m unittest scripts/tests/test_export_langfuse_session_llm_io.py
```

