# 代码转换实现与注意事项

本页只处理CUDA→HIP/HCU代码层的header、条件宏、库/API/数据类型和句柄语义。转换器选择、CMake/Make/Extension接入放在[编译构建适配](build-adaptation.md)；内嵌汇编、builtin和warp的具体实现放在[指令处理](instruction-porting.md)，本页不复制另一套规则。

转换时先保留目标API的解析链，最小补丁隔离真实HCU差异；原CUDA分支、公共API/ABI/layout、原测试和容差保持。转换命中或编译通过不证明运行语义等价。

拟修改接口的理由和验证入口记入[分析模板](../docs/pre-build-analysis.md)，区分编译兼容、行为修复与可选整理；不以“适配需要”代替实际缺口。

## 1. 头文件与API映射

头文件与API映射应查实际转换器加载的 `cuda_to_hip_mappings.py`、额外映射及其合成顺序，不能假定构建使用的是当前安装Torch自带的表。

涉及 `c10::cuda`、`at::cuda`、allocator/stream/guard 或 `MasqueradingAsCUDA` 时，先查 [PyTorch HIPIFY v1/v2 与目标接口匹配](pytorch-hipify.md)。分别核对框架符号、header路径和runtime API；不要把v1生成名带入v2目标，也不要将“v2保留框架名”推广成全部停止转码。

以下是映射**候选**，须核对目标DTK头、wrapper、导出符号与函数参数，不能批量替换库名：

| CUDA面                    | HIP/DTK候选                           | 关键区别                                                   |
| ------------------------- | ------------------------------------- | ---------------------------------------------------------- |
| `cuda_runtime.h`        | `hip/hip_runtime.h`                 | runtime/Driver API范围、异步错误与生命周期                 |
| `cublas_v2.h` / cuBLAS  | `hipblas/hipblas.h` / hipBLAS       | enum、math/compute模式、stream、workspace与底层owner       |
| `cublasLt.h` / cuBLASLt | `hipblaslt/hipblaslt.h` / hipBLASLt | matmul描述符、epilogue、数据/累加/scale类型和可实现组合    |
| `cudnn.h` / cuDNN       | DTK hipDNN（或工程已选的其他backend） | 不是直接链接MIOpen便等价；核对descriptor、layout和算法范围 |
| CUB / Thrust              | hipCUB / rocThrust兼容面              | 模板版本、逻辑warp和TempStorage                            |
| cuRAND                    | HIP wrapper或目标实际提供者           | 不将HIP wrapper与rocRAND底层ABI混用                        |

其他接口映射：

```
    "CUdevprop_st": "hipDeviceProp_t_v2",
    "CUdevprop": "hipDeviceProp_t_v2",
    "cudaDeviceProp": "hipDeviceProp_t_v2",
    "cuDeviceGetProperties": "hipGetDeviceProperties_v2",
    "cudaGetDeviceProperties": "hipGetDeviceProperties_v2"
```

## 2. 编译期条件宏

| 条件                                                               | 要证明的事                                                                         |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `USE_ROCM` / 平台宏                                              | 工程/框架何处定义、是否覆盖当前编译单元；不使用裸else代表所有HCU                   |
| `CUDA_VERSION` / `CUDA_ARCH` / `__CUDA_ARCH__`               | 区分项目自定义宏、SDK版本与设备编译条件，不作名字对名字替换                        |
| `ATen/dtk_macros.h` / `FPT_DTK_VERSION` / `__FPT_DTK_ARCH__` | 目标头和HIPIFY是否注入、各条件和值；兼容常量不等于真实DTK或gfx，不改大常量开启接口 |
| `ROCM_VERSION` / host/device条件                                 | 未定义、空定义、0不同；分别核实host/device宏，架构兼容宏可能两侧都存在             |

最小平台隔离示意（`PROJECT_USE_HIP`必须由当前构建实际设置，不能照抄成设备能力证明）：

```cpp
#if defined(PROJECT_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
```

## 3. hipBLAS与hipBLASLt精度类型、句柄兼容

| 参数                                        | 处理                                                                                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 数据类型                                    | 某些版本hipBLAS使用`hipblasDatatype_t`，Lt使用`hipDataType`；即使表示FP32也可能是不同枚举族。新版本别名也需看原型，逐值转换，不整数强转 |
| compute / scale type                        | 与输入输出dtype分开。由具体函数/descriptor原型选类型，核实FP16/BF16/FP8存储、累加、缩放组合                                                 |
| `hipblasHandle_t` / `hipblasLtHandle_t` | 分别create/destroy，不共享或reinterpret_cast；保持device/context/lifetime合同                                                               |
| stream与workspace                           | 按各API设置/传入，确认所有权和异步释放条件；BLAS可调用不证明Lt对应组合可用                                                                  |

原CUDA代码依赖某版本typedef兼容时，迁移后也要重新核实；不能把CUDA侧可编译推成HIP枚举/句柄可互换。


## 4. 依赖库支持与替代边界

先核对库 owner、版本和实际重载，再决定是否需要 wrapper。`cuda::atomic_ref`/`minimum` 在 HCU 的 `libhipcxx`、以及 hipCUB reduce 只能作为版本限定候选；若目标已有 `g.ballot` 等可直接复用的原语，应优先复用，不能仅凭库名或 namespace 判断不可用。

| 具体候选 | 决策前的证据 | 合理结果 |
| --- | --- | --- |
| 手写字节/word CAS | 当前HCU libhipcxx的`cuda::atomic_ref<T>`是否支持所需类型、对齐、memory order/scope；实际编译参数与并发值验证 | 满足时复用库原子接口；packed布局保持，不凭类型声明就删除原实现 |
| 手写min归约 | hipCUB `WarpReduce<T,G>`及函数对象的版本、返回lane约定、每组TempStorage和广播 | 库负责归约，必要适配仅保留存储/结果广播；不能共享不同wave正在使用的TempStorage |
| 将`g.ballot`换成自写wrapper | 实际cooperative groups头中是否已有对应分组接口及足够宽的返回值 | 可用时保留调用，只修不够宽的接收/消费类型 |

只在候选接口能力确有疑问时，用当前生产编译参数做有界探针；保留首个失败声明/重载/feature gate，区分未提供与include配置错误。独立库探针通过后，仍验证生产调用链和受影响原测试。

| 范围              | 支持边界与处理                                                                                                                           |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| CUTLASS/hytlass   | 历史HYTLASS0.1.0说明基于CUTLASS3.5.0，不能外推CUTLASS4.x/CuTe DSL/Sm90或Sm100/WGMMA/TMA。固定实际release/完整commit及工具链              |
| CUDA/OpenGL互操作 | 本HCU适配范围按不支持处理，不能拿普通HIP构建或上游OpenGL文档证明DTK该功能可用；可选图形功能需批准排除，必需功能则记录阻塞/替代方案       |
| FP8               | gfx936不按原生FP8计算规划；gfx938硬件支持仍需compiler→backend→格式/算子→运行闭包。量化/存储、CK/Triton/FlyDSL、GEMM/attention分别核验 |
| TensorRT等依赖    | 先查是否有满足当前版本/功能的HCU实现；必需功能不能直接禁用，替代影响需明确                                                               |

hytlass逐kernel核查`can_implement`、lane-fragment、tile、alignment/stride、workspace、尾部、stages/barrier、epilogue/alpha-beta和Split-K归约。无等价实现时形成明确改写方案，不能靠Sm→Gfx改名。

依赖源码/包检索入口统一见[相关资源路径](resources.md)。固定匹配版本和实际消费路径，CI安装/optional import不证明该backend可省略。hytlass的构建接入与准备工具见[编译构建适配](build-adaptation.md)。

## 5. 扫描卡片索引

<a id="s-01"></a>

### S-01 版本与平台宏

见上面的宏表；保持条件作用域，不伪造版本/能力。

<a id="s-02"></a>

### S-02 内嵌汇编

按[指令指南](instruction-porting.md)核对约束、lane布局、内存序和流水线；文本转换不证明等价。

<a id="s-03"></a>

### S-03 builtin

按[指令指南](instruction-porting.md)逐个查声明/feature gate/lowering，不按前缀替换。

<a id="s-04"></a>

### S-04 库

查本页header/库/类型表，保留参数、生命周期与链接owner。

<a id="s-05"></a>

### S-05 runtime、Driver 与纹理

逐API核对JIT、资源寿命、同步、pitch/对齐；头文件存在不等于实现和行为可用。

<a id="s-06"></a>

### S-06 矩阵核心、CUTLASS 与低精度

核查上面的版本子集及format×operation×backend，不能用类型声明或一个kernel代证全部实例。

<a id="s-07"></a>

### S-07 架构参数

SM/compute与完整gfx、专用数字参数分开；从生成命令和实际产物确认，不FORCE覆盖用户目标。

<a id="s-08"></a>

### S-08 工具链标记

HIP/HCU字段和路径只作候选，按[构建指南](build-adaptation.md)查真实入口、宏与加载产物。

<a id="s-09"></a>

### S-09 路线限制

静态runtime、混用、设备侧launch按所选路线和版本核查；rdc也可用于分离编译，不单独判动态并行。非转码专属要求在该路线指南。

<a id="s-10"></a>

### S-10 非包源码依赖

先查已适配源码，固定版本、调用API和构建消费关系；外部include默认只读。

<a id="s-11"></a>

### S-11 后端依赖

追schema/注册、host/device声明、模板实例、dispatch/metadata和实际运行。optional import、架构筛选和CI安装不独立证明可用或可排除。
