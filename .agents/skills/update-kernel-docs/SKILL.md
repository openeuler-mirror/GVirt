---
name: update-kernel-docs
description: Generate or refresh operator implementation-principle docs in xlite/doc/developer_guide/kernels/ based on current kernel source code
---

# xlite 算子文档生成与刷新 Skill

本 skill 用于为 `xlite/csrc/kernels/` 下的 NPU 算子生成或刷新实现原理文档。文档位于 `xlite/doc/developer_guide/kernels/`,每个算子一份 markdown,索引在 `xlite/doc/developer_guide/kernels/README.md`。

## 使用方法

```
/update-kernel-docs [算子名 ...]
```

- 不带参数:检查全部算子文档与当前代码的一致性,列出过时项并刷新
- 带算子名:仅处理指定算子(如 `/update-kernel-docs mla_v2 cxa`)
- 新增算子(文档不存在):为新算子生成文档并登记到 README

## 文档规范

每份算子文档(中文,文件名 `doc/developer_guide/kernels/<算子名>.md`)必须包含以下章节:

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
- 片上缓冲分配与占用汇总小节(必备,量化各存储级占用,规范见下文「片上缓冲容量说明」)
- 分块/tiling 策略、多 block/AIC/AIV 并行与任务分配方式
- 流水线同步(MTE2/MTE3/V/M pipe 的 set_flag/wait_flag、跨核 ffts_cross_core_sync/wait_flag_dev)
- 关键指令与计算步骤(vconv/vgather/TransDataTo5HD/CalMmad 等)
- 边界处理(对齐、尾部补齐、mask)

## 关键代码位置(可选)
关键函数入口的符号定位列表(函数名/类名,**禁止行号**)。

- **禁止行号**:严禁出现任何 `文件:行号` 形式的引用(如 `add.h:33` 或 `add.h#L33`)。代码增删行后行号必然整体偏移,行号引用刷新一次即失效并误导读者,一律禁止
- **改用符号定位**:每条引用写成「相对 `xlite/` 的路径 + 函数名/类名」,如「`csrc/kernels/mla_v2.cpp` 的 `MlaV2::Compute`」
- **适用条件**:仅当算子实现源码较长(超过约 1000 行)时才需要本章节;短算子(≤1000 行)整体可读性强,不需要本章节,应省略整个章节而非留空标题
- **选取范围**:仅标记对理解实现原理最关键的函数入口(如主计算入口、tiling/分块函数、流水线同步函数、边界处理函数),总数**不超过 5 个**
- **避免堆砌**:不要把所有函数都列出,也不要列纯 boilerplate(如构造函数、Run 派发壳);每条引用应指向读者定位核心逻辑时最想跳转的那个符号
```

引用代码位置统一使用相对 `xlite/` 的路径 + 符号名(函数名/类名),**严禁使用行号**(`文件:行号`、`文件#L行号` 一律禁止)。正确写法:`csrc/kernels/add.h` 的 `Add` 类、`csrc/kernels/mla_v2.cpp` 的 `MlaV2::Compute`。

## 片上缓冲容量说明(必备小节)

每份算子文档的「实现原理」末尾必须包含 `### 片上缓冲分配与占用汇总` 小节,量化该算子的片上存储开销(范例:`doc/developer_guide/kernels/indexer_topk.md`)。

- **表格**:按算子实际使用的存储级各建一张表——AIC 算子列 L1(A1)/L0A(A2)/L0B(B2)/L0C(CO1),AIV 算子列 UB,混合算子两者都要;未使用的存储级直接省略(纯向量算子只有 UB 表)。每行一个代码中的缓冲变量,按 `Init`/`InitBuffer` 中 `bufferAddr` 的排布顺序列出
- **列**:变量、shape(每份)、dtype(模板参数名 + 各 dtype 路径的实例化映射)、乒乓个数(标注所属事件族)、单份大小(与代码一致的字节公式,如 `blockSize*headDim*sizeof(Dtype)`)、典型模型配置占用字节数(= 单份大小 × 乒乓个数,含对齐垫片时按槽位计)。两条 dtype 路径同取一组典型配置(具体值写在小节引言),使占用差异仅来自 dtype;UB 布局与 dtype 无关时单占用列即可
- **合计行**:每张表末尾加粗「合计」行,给出该存储级总占用
- **占用汇总表**:各表占用列之和 vs 硬件容量 + 占比;再以要点列出关键结论(哪级最紧、哪个缓冲是大头、断言/host 校验的容量来源)
- **硬件容量基准**(910B/dav C220,CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`):L1 512KB、L0A 64KB、L0B 64KB、L0C 128KB(`l0_c_size=131072`)、UB 192KB(`kernel_macro.h` 的 `UB_SIZE`)
- **数值自检**:各表占用列之和必须与占用汇总表逐级严格一致,建议写一次性脚本复核后再提交

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
2. 写 `doc/developer_guide/kernels/<name>.md`
3. 在 `doc/developer_guide/kernels/README.md` 对应功能分组表中登记:`| <name> | [<name>.md](<name>.md) | <一句话说明> |`
4. 用 Write 后回读抽查:文档中引用的函数名/类名在当前源码中真实存在;**全文不得出现任何 `文件:行号` 形式的引用**

### 刷新已有文档(代码变更后)

1. 用 `git diff <旧基线>..HEAD -- xlite/csrc/kernels xlite/tests/kernels xlite/csrc/op.cpp` 找出受影响算子;或直接重读算子源码与文档比对
2. 对每个受影响算子:
   - 核对功能概述、参数表、dtype 列表是否仍准确(新增/删除参数、模式、约束必须同步)
   - **逐一核对所有代码位置引用**:确认文档中提到的函数名/类名在当前源码中仍存在;符号被改名/移位时改写引用指向新符号名,**严禁改用行号定位**(行号随增删行整体偏移,刷新一次即失效)。这是最易出错的一步,不能跳过
   - 代码已删除的概念(如已合并/移除的算子路径、宏改名)需改写或删除,不能残留旧名
   - 缺「片上缓冲分配与占用汇总」小节或其数值与当前代码不符时,按「片上缓冲容量说明」补齐/更新(含合计行与占用汇总表)
3. 若算子被删除,删文档并从 README 移除;若算子改名,重命名文档并全局搜旧名清理引用

### 一致性检查(收尾必做)

```bash
# 1. README 链接与实际文件一一对应
cd xlite/doc/developer_guide/kernels
grep -o '([a-z0-9_]*\.md)' README.md | tr -d '()' | sort -u | while read f; do [ -f "$f" ] || echo "BROKEN: $f"; done

# 2. 每份文档包含必要章节
for f in <name>.md; do grep -q "实现原理" $f || echo "missing: $f"; done

# 3. 无过时概念残留(按需替换关键词)
grep -rn "<已删除的算子/宏名>" *.md

# 4. 禁止行号引用(以下应无输出;有输出即违规,必须改写为符号定位)
grep -rnE '\.(h|cpp|cc):[0-9]+|#L[0-9]+' *.md

# 5. 每份算子文档含片上缓冲容量小节(缺失的文档在下次刷新该算子时补齐)
for f in $(ls *.md | grep -v '^README.md$'); do grep -q "片上缓冲分配与占用" "$f" || echo "missing-capacity: $f"; done
```

## 容易出错的点

- **禁止行号引用**:本 skill 严禁在文档中写入任何 `文件:行号` 形式(如 `add.h:33`、`add.h#L33`)的代码位置引用。代码增删行后行号必然整体偏移,行号引用会迅速失效并误导读者,刷新成本高且易漏。统一改用「路径 + 函数名/类名」定位;刷新历史文档时若发现遗留行号引用,必须就地改写为符号定位
- **dtype 实例化文件 vs 算子**:文件名匹配 `<name>_*.cpp` 时注意 `cast_float_bfloat16_t.cpp` 这类双 dtype 文件,不要误拆成两个算子
- **shape 推断**:优先以 `_C.pyi` docstring 和测试代码为准,kernel 签名中的裸指针参数(如 `GM_ADDR q`)需结合 host 侧封装还原 shape
- **文档语言**:文档为中文,代码注释是中英混合,统一用中文表述;术语(算子名、指令名、宏名)保留英文原文
- **L0C 容量**:CANN 平台配置 `l0_c_size=131072`(16 bank × 8KB)即 L0C 总容量 **128KB**,`TPosition::CO1` 的 mmad 累加缓冲占满整个 L0C——容量说明中 L0C 的占用与占比一律按 128KB 计算(旁证:`csrc/kernels/cxa_aic_helper.h` 的 `svl0cBuf` = 128×256×4B 恰好顶满 L0C)
- **容量小节必备**:存量文档缺「片上缓冲分配与占用汇总」小节时,刷新该算子必须按「片上缓冲容量说明」补齐,不能只改过时的参数/引用
