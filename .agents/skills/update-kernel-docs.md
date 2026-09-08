---
name: update-kernel-docs
description: Generate or refresh operator implementation-principle docs in xlite/doc/kernels/ based on current kernel source code
---

# xlite 算子文档生成与刷新 Skill

本 skill 用于为 `xlite/csrc/kernels/` 下的 NPU 算子生成或刷新实现原理文档。文档位于 `xlite/doc/kernels/`,每个算子一份 markdown,索引在 `xlite/doc/kernels/README.md`。

## 使用方法

```
/update-kernel-docs [算子名 ...]
```

- 不带参数:检查全部算子文档与当前代码的一致性,列出过时项并刷新
- 带算子名:仅处理指定算子(如 `/update-kernel-docs mla_v2 cxa`)
- 新增算子(文档不存在):为新算子生成文档并登记到 README

## 文档规范

每份算子文档(中文,文件名 `doc/kernels/<算子名>.md`)必须包含以下章节:

```markdown
# <算子名>

## 功能概述
一两句话说明数学语义与用途(含公式/伪代码,如有退化形态或模式分支需列出)。

## 输入输出参数
表格: | 参数 | 方向 | Shape | Dtype | 说明 |
- Shape 用符号表示(如 [num_tokens, n_heads, head_dim]),依据 kernel 签名、host 侧封装和测试代码推断
- 标量参数 Dtype 写实际类型(uint32_t/int32_t/float)
- 标注 workspace 类参数与原位写入(out/in-place)语义

## 支持的数据类型
列出 dtype 变体及对应实例化源文件(如 csrc/kernels/<name>_bfloat16_t.cpp)。

## 实现原理
- 数据流(GM→UB/L1→计算→GM)、UB/L1/L0 内存布局与缓冲分配
- 分块/tiling 策略、多 block/AIC/AIV 并行与任务分配方式
- 流水线同步(MTE2/MTE3/V/M pipe 的 set_flag/wait_flag、跨核 ffts_cross_core_sync/wait_flag_dev)
- 关键指令与计算步骤(vconv/vgather/TransDataTo5HD/CalMmad 等)
- 边界处理(对齐、尾部补齐、mask)

## 关键代码位置(可选)
关键函数入口的行号引用列表。
```

引用代码位置统一使用相对 `xlite/` 的路径 + 行号,如 `csrc/kernels/add.h:33`(markdown 中写成 `` `csrc/kernels/add.h:33` `` 或链接形式 `[add.h:33](../../csrc/kernels/add.h#L33)`)。

## 信息来源(按优先级)

1. **kernel 实现**:`csrc/kernels/<name>.h`(模板类/函数)或 `<name>.cpp`
2. **dtype 实例化**:`csrc/kernels/<name>_<dtype>.cpp` 文件名即支持的 dtype
3. **host 侧封装**:`csrc/op.cpp` 中 `XliteOp<Name>`(dtype 校验、参数语义、限制约束);CCL 类算子在 `csrc/ccl.cpp`
4. **Python 绑定与签名**:`xlite/_C.pyi`(函数 docstring 常含最完整的 shape/mask 语义说明)、`csrc/_C.cpp` 的 pybind11 定义
5. **测试**:`tests/kernels/<name>.py`(shape 约定、调用方式、对比基准)
6. **模型路由**:`csrc/model.cpp`(算子在整体推理流程中的位置)
7. **公共头**:`csrc/kernels/kernel_param.h`、`kernel_macro.h`、`config_macro.h` 中的常量与工具函数

辅助头文件(`<name>_aic_helper.h`、`softmax_attn_aiv.h` 等)的原理并入其宿主算子文档,不单独成文;被多个算子复用的公共组件(如 `ring_sync.h`)可单独成文并在 README 说明。

## 工作流程

### 生成新算子文档

1. 按上述信息来源依次读取代码,确认:参数列表与语义、支持的 dtype、tiling/流水线结构、host 侧约束
2. 写 `doc/kernels/<name>.md`
3. 在 `doc/kernels/README.md` 对应功能分组表中登记:`| <name> | [<name>.md](<name>.md) | <一句话说明> |`
4. 用 Write 后回读抽查引用行号是否与代码一致

### 刷新已有文档(代码变更后)

1. 用 `git diff <旧基线>..HEAD -- xlite/csrc/kernels xlite/tests/kernels xlite/csrc/op.cpp` 找出受影响算子;或直接重读算子源码与文档比对
2. 对每个受影响算子:
   - 核对功能概述、参数表、dtype 列表是否仍准确(新增/删除参数、模式、约束必须同步)
   - **逐一核对所有行号引用**(`csrc/...:<N>`):文件增删行后行号会整体偏移,用 `sed -n '<N>p' <file>` 或 Grep 关键字(函数名、特征指令)重新定位。这是最易出错的一步,不能跳过
   - 代码已删除的概念(如已合并/移除的算子路径、宏改名)需改写或删除,不能残留旧名
3. 若算子被删除,删文档并从 README 移除;若算子改名,重命名文档并全局搜旧名清理引用

### 一致性检查(收尾必做)

```bash
# 1. README 链接与实际文件一一对应
cd xlite/doc/kernels
grep -o '([a-z0-9_]*\.md)' README.md | tr -d '()' | sort -u | while read f; do [ -f "$f" ] || echo "BROKEN: $f"; done

# 2. 每份文档包含必要章节
for f in <name>.md; do grep -q "实现原理" $f || echo "missing: $f"; done

# 3. 无过时概念残留(按需替换关键词)
grep -rn "<已删除的算子/宏名>" *.md
```

## 容易出错的点

- **行号偏移**:公共头(`softmax_attn_aiv.h`、`kernel_param.h`、`kernel_macro.h`)被多份文档引用,任何增删行都会让这些引用同时失效;改动这些头文件后必须全量检查 `grep -l "kernel_param.h:" doc/kernels/*.md` 等
- **dtype 实例化文件 vs 算子**:文件名匹配 `<name>_*.cpp` 时注意 `cast_float_bfloat16_t.cpp` 这类双 dtype 文件,不要误拆成两个算子
- **shape 推断**:优先以 `_C.pyi` docstring 和测试代码为准,kernel 签名中的裸指针参数(如 `GM_ADDR q`)需结合 host 侧封装还原 shape
- **文档语言**:文档为中文,代码注释是中英混合,统一用中文表述;术语(算子名、指令名、宏名)保留英文原文
