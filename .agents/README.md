# GVirt / xlite Agent Skills

本目录汇总本仓库提供的 Agent Skills,供各类 AI 编码代理(Claude Code、其他兼容 `.agents/` 约定的工具)发现和使用。`.claude/skills/` 是 Claude Code 专属入口,与本目录保持同步。

## Skills 列表

| Skill | 文件 | 用途 |
|---|---|---|
| debug-accuracy | [skills/debug-accuracy.md](skills/debug-accuracy.md) | 通过对比 xlite 与 torch_npu 后端输出,系统性定位 xlite 精度问题 |
| update-kernel-docs | [skills/update-kernel-docs.md](skills/update-kernel-docs.md) | 生成 / 刷新 `xlite/doc/kernels/` 下的算子实现原理文档 |

## 使用方式

- Claude Code:在 `.claude/skills/` 中以 `/debug-accuracy <model>` 形式调用(debug-accuracy);update-kernel-docs 以 `/update-kernel-docs` 调用
- 其他 agent:直接将对应 skill 的 markdown 内容作为任务指引注入上下文

## 新增 Skill 约定

1. 在 `skills/` 下新增 `<skill-name>.md`,frontmatter 必须包含 `name` 与 `description`
2. 在 `.claude/skills/` 建立**相对路径软链接**(单一来源,避免副本不同步):
   ```bash
   ln -s ../../.agents/skills/<skill-name>.md .claude/skills/<skill-name>.md
   ```
3. 在上方 Skills 列表中登记
