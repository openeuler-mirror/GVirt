### xlite与vllm ascend的特性交互矩阵

符号说明
- ✅ = Full compatibility
- 🟠 = Partial compatibility
- ❌ = No compatibility
- ❔ = Unknown or TBD


| 特性 | xlite full mode | xlite decode only mode |
|---|---|---|
| ACLGraph Full_Decode_Only | N/A | ✅ |
| ACLGraph Piecewise | N/A | ✅ |
| Async Scheduling | ✅ | ✅ |
| Automatic Prefix Caching | ✅ | ✅ |
| Chunked Prefill | ✅ | ✅ |
| Context Parallel | ❌ | ❌ |
| Cpu Binding | ✅ | ✅ |
| Data Parallel | ✅ | ✅ |
| Disaggregated Prefill | ✅ | ✅ |
| Speculative Decoding(MTP) | 🟠 | 🟠 |
| Speculative Decoding(Eager3) | ❌ | ❌ |
| EPLB | ❌ | ❌ |
| EP | ✅ | ✅ |
| Flashcomm1 | N/A | ✅ |
| KV Cache Pool | ✅ | ✅ |
| Lmhead TP | ❔ | ❔ |
| MLAPO | N/A | ✅ |
| Multimodal Inputs | 🟠 | ✅ |
| Multistream Moe | N/A | ✅ |
| Shared Expert DP | ❔ | ✅ |
| Quantization W4A8 | ❌ | ❌ |
| Quantization W8A8 | ✅ | ✅ |
| Tensor Parallel | ✅ | ✅ |
| Weight NZ | 🟠 | 🟠 |


备注说明：
* KV Cache Pool/Disaggregated Prefill：kv connector仅支持非layerwise方式
* speculative：仅支持线性投机，暂不支持树形投机
* N/A: 由于xlite full mode已经接管了全部forward流程，对应特性在xlite中有相关实现，因此无需再手动开启vllm ascend的特性