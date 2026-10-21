# XTensor 布局与 stride 语义

XTensor(`csrc/base.h`)是 host 侧对设备内存张量的轻量描述。本文说明其 shape/stride/bytes
语义、连续性判定规则、与 torch 张量的包装策略,以及非连续布局的用途。

## 字段语义

shape/stride/numel 以元素计,bytes 为字节数:

| 字段 | 含义 |
| --- | --- |
| shape / dtype / ptr | 逻辑形状、数据类型、设备地址(ptr 指向张量首元素,而非 storage 首) |
| stride | 各维步长(元素单位);连续张量等于 `ContiguousStride(shape)`(stride[i] = prod(shape[i+1:]),末维为 1) |
| numel | 逻辑元素数 prod(shape) |
| bytes | 逻辑(紧凑)字节数 `DIV_ROUND_UP(numel × 位宽, 8)`,与布局无关 |

`IsContiguous()` 反映 stride 是否与 shape 的连续布局完全一致。

## stride 的确定规则

`XTensor::Init`/构造函数接受可选 stride(`std::optional<std::vector<size_t>>`):

1. 未提供:按连续张量处理,stride 取 `ContiguousStride(shape)`。
2. 提供:维度必须与 shape 相同,否则抛 `std::invalid_argument`。
3. 每一维都满足 stride[i] == prod(shape[i+1:])(末维为 1)→ isContiguous = true。
4. 否则 isContiguous = false,且仅允许 dim 0 偏离连续布局(行距):任一 dim ≥ 1 与连续布局
   不符时抛 `std::invalid_argument`;dim-0 步长也不得小于行长 prod(shape[1:])(行重叠/广播
   展开布局,含 stride 0),否则同样抛 `std::invalid_argument`。该行长约束仅在 shape[0] > 1
   时生效——单行(或零行)张量不存在行重叠,dim-0 步长不参与寻址。

即非连续张量被限制为"每个 dim-0 切片是 prod(shape[1:]) 个元素的连续段,pitch =
stride[0] ≥ 行长"。行距填充(padding)、按 dim 0 抽行(`x[::k]`)、2D 融合缓冲的末维 split 都属于
这一类;transpose、内层切片、expand 广播(行重叠)则不被接受。空张量(numel == 0)在包装层
(见下节)不受 stride 校验约束,一律按连续布局包装。

## 布局受限操作

`View(shape)`/`View(XDtype)`/`View(newDim, dim)`/`ResetView`/`MakeOriginal`/`Memset`
仅支持连续张量,对非连续张量抛 `std::runtime_error`。View 系列在修改 shape 后会将 stride
重置为新 shape 的连续布局;`MakeOriginal` 会同步记录 origStride。

## 非连续张量的来源与消费

内部约定:XLite 内部所有 XTensor 操作(池分配、View、算子输出)只产生连续张量;非连续
(dim-0 行距)XTensor 只能来自 `InitXTensor` 包装的外部 torch 张量,且仅限两类消费方:

1. **KV cache 相关张量**(MHA/GQA/MLA/DSA/CXA 等):cache 以分块/行距布局存放,正在对齐
   上游 vllm-ascend 让 kernel 按 stride[0] 寻址(后续提交);其 wrapper 以 `allowPitched`
   包装 cache 数据张量。混合注意力模型中线性层的 conv/ssm 状态张量虽经 kvCache 传入,但按
   连续布局消费,不参与该豁免。其余 meta 张量(blockTables/slotMapping/lens 等)仍要求连续。
2. **调试输出**:`Print`/`CheckNanInf` 按实际内存跨度((shape[0]-1)×stride[0] + 行长)拷贝
   设备内存并按 stride 定位元素,因此对 dim-0 行距张量输出正确的逻辑值;`Save` 通过
   `at::from_blob` 携带 stride 包装,再由 torch 自己的 `.cpu().contiguous()` 适配布局落盘,
   无需自定义设备侧拷贝;`ToStr` 在非连续时追加 `stride=[...]`。

其余算子与 kernel 一律按 ptr + shape 的连续布局寻址,不做 stride 适配;对这类入口,
`InitXTensor` 在包装期直接拒绝非连续 torch 张量(见下节),避免静默读错数据。

## torch 张量包装(csrc/xlite.h `InitXTensor`)

`InitXTensor` 以 torch 的 `is_contiguous()` 作为布局依据先行分流,再按调用方声明的策略处理:

- torch 判定为连续的张量(含 size-1 维步长偏差——size-1 维的 stride 不参与元素寻址,内存
  布局仍与行主序逐字节一致)以及 numel == 0 的张量,直接按连续布局包装,不转发 stride。
- 其余非连续张量在默认(`allowPitched = false`)下抛 `std::invalid_argument`,报错信息同时
  给出实际与期望 stride,补救为 `.contiguous()`(此处 torch 已判定非连续,`.contiguous()`
  必然重新物化)。
- `allowPitched = true`(cache 包装与调试输出):将 torch stride 转发给 `XTensor::Init` 校验,
  接受合法的 dim-0 行距(不重叠、高维连续),其余布局仍抛错。负 stride/负 size 一律抛错。

## 测试参考

- [tests/funcs/stride.py](../../tests/funcs/stride.py):Print 的 stride 语义、size-1 维张量的
  连续包装、包装期拒绝规则(transpose/内层切片/expand),以及非 cache 入口的连续性守卫的
  功能测试。
