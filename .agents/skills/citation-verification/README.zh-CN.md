# citation-verification

约束 AI 助手在引用学术文献时必须先核验真实性与相关性，杜绝瞎编 DOI 或张冠李戴。

## 行为链

每当助手引用一篇论文支撑观点时，强制走 4 步：

1. **验真** — 确认 DOI / PMID / arXiv ID 真实可解析
2. **读摘要** — 不准只看标题就引用
3. **判相关性** — 不相关或结论相反的，直接舍弃
4. **必要时读正文** — 涉及具体数字、方法、亚组结果时

输出格式：`[作者 et al., 年份, 标题] (DOI) + 简短结论 + 原文金句 (≤ 100 字符)`。

摘要拿不到时，标 `[no accessible abstract]`，绝不编造原话凑数。

## 安装路径

| 平台 | 路径 |
|---|---|
| OpenCode（用户级）| `~/.config/opencode/skills/citation-verification/` |
| OpenCode（项目级）| `<项目>/.opencode/skills/citation-verification/` |
| Claude Code / Copilot CLI / Gemini CLI | 按各自 skill 规范放置 |

## 推荐搭配的 MCP 工具

本 SKILL 不绑定具体工具，但配上文献检索类 MCP 体验最佳：

- **academic-research** — DOI 验证、多源检索、开放获取 PDF 解析
- **semantic-scholar** — 论文元数据、全文 Markdown 转换
- **sciverse** — 语义检索、按字节偏移读正文
- **tavily** — DOI 落地页的网页搜索兜底

## License

MIT.