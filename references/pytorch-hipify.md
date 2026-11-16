# PyTorch HIPIFY v1/v2：框架接口、头文件与 runtime 分开核对

遇到 `c10::cuda::CUDAStream` 被转成 `c10::hip::HIPStreamMasqueradingAsCUDA`，或类似的 stream/guard/allocator 类型未声明时，先核对**实际转换器的代际与目标 Torch 的接口是否匹配**。这通常是转换规则或消费路径问题的线索，不能仅凭报错就给项目新增兼容类。

## 1. v2 保留的是框架名字，不是停止转码

[PyTorch 2.11.0 发布说明的 ROCm 部分](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)介绍了 hipify v2：取消旧 Caffe2/PyTorch 两套命名策略的特殊处理，保留框架的 CUDA 类名，继续转换 runtime API，文件仍可重命名。该 tag 的 [hipify 版本](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/hipify/version.py)为 `2.0.0`。

这是一个可查证的版本样例，**不是所有发行版统一以 Torch 2.11 为分界**。HCU fork、backport、vendor 工具及安装包补丁可能不同；以实际入口、最终映射、目标头和库为准。

| 层次 | v2 样例 | 判断依据 |
| --- | --- | --- |
| PyTorch 框架类型/namespace/函数 | 保留 `c10::cuda::CUDAStream`、`c10::cuda::getCurrentCUDAStream()` | 目标 HIP 头中的真实声明/owner，不由路径名推断 namespace |
| include 路径 | `c10/cuda/CUDAStream.h` → `c10/hip/HIPStream.h` | 该版本的路径映射与安装包布局；类名保留不意味着头路径保留 |
| runtime 类型和函数 | `cudaStream_t` → `hipStream_t`，`cudaStreamSynchronize` → `hipStreamSynchronize` | 当前 HIP/DTK 原型、参数、错误处理和 stream 语义 |

v1 曾用 Masquerading 类衔接 Caffe2 的 HIP device 与 PyTorch 的 CUDA device 约定。v2 不再要求生成这些旧类名，但兼容头可能保留；例如 v2.11.0 的 [HIPStreamMasqueradingAsCUDA.h](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h)仍定义继承 `c10::cuda::CUDAStream` 的薄壳。**某编译单元报“未声明”不等于整个包已删除该类型**，还要查 include 链和条件宏。

## 2. 核对实际使用的那一份 HIPIFY

在构建实际使用的解释器和已初始化环境中，按需执行以下只读排查。它会 import Torch，只用于调查；不改变现有静态扫描工具不导入目标包的约定，也不要在不可信项目目录下运行。

```python
import sys
import torch
from torch.utils import hipify
from torch.utils.hipify import hipify_python

print("python:", sys.executable)
print("torch:", torch.__version__, torch.__file__)
print("backend:", torch.version.hip, torch.version.cuda)
print("hipify:", getattr(hipify, "__version__", "unknown"), hipify.__file__)
print("converter:", hipify_python.__file__)
```

这只证明该解释器导入的 Torch 自带工具。外部 `hipify-torch`、项目 vendor 副本、FastPT 转换器仍需分别查实际模块路径、commit/包版本、映射文件及额外规则。缺版本字段或 import 失败记为 unknown，不自行认定 v1。

一个具体误选案例是 [ROCm/hipify_torch 的固定版本 CLI](https://github.com/ROCm/hipify_torch/blob/63b6a7b541fa7f08f8475ca7d74054db36ff2691/hipify_cli.py)：

- 此仓库的包版本为 `1.1.0`，但同时含 `hipify_torch.hipify_python`（v1）和 `hipify_torch.v2.hipify_python`（v2）。包版本不是所选转换代际。
- CLI 通过 `--v2`，或检测 installed `torch.utils.hipify.__version__ >= 2.0.0`，选择 v2；检测失败会打印信息并回退 v1。该行为仅针对已核查的实现，不推广到所有工具。
- 构建环境缺 Torch/依赖、CMake 调用另一 Python、或直接导入根目录的 v1 模块，都可能让新 Torch 搭配旧规则。
- `is_pytorch_extension=True` 选择扩展处理方式，**不是 v2 开关**；C++ 的 `-DHIPIFY_V2` 也不能替 Python 转换器选版本。`--v2` 是上述外部 CLI 的参数，不要传给不支持它的 Torch Python API。

处理时优先沿项目已指定、与目标匹配的 provider；修实际入口、解释器绑定或该实现支持的选项。不要只升级 Torch、添加宏便认定转换链已更新，也不静默换 provider。读合成后的有效映射：上述 v1 在 extension 模式下由后面的 PyTorch 表覆盖前面的 C10 表，只查一条 `C10_MAPPINGS` 会得出错误结果。

## 3. 具体代码：头路径变了，C10 名字不变

以下是扩展模式下的普通 C++ 片段，假定目标采用上述 v2 规则；它用于观察转换结果，不是已通过编译的项目或 GPU 测试。

原始 CUDA 源：

```cpp
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void wait_current(c10::DeviceIndex device) {
  c10::cuda::CUDAGuard guard(device);
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(device);
  cudaStream_t raw = stream.stream();
  C10_CUDA_CHECK(cudaStreamSynchronize(raw));
}
```

v2 对应的代码形态（省略工具生成标记和发行版注入头）：

```cpp
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPException.h>

void wait_current(c10::DeviceIndex device) {
  c10::cuda::CUDAGuard guard(device);
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(device);
  hipStream_t raw = stream.stream();
  C10_CUDA_CHECK(hipStreamSynchronize(raw));
}
```

不要因 `c10/hip/HIPStream.h` 的路径含 HIP 就把类改为 `c10::hip::HIPStream`；也不要为了保留 `CUDAStream` 而屏蔽 `cudaStream_t`、runtime 调用和头文件的必要转换。

其他接口也需要按同样原则处理。下表的 v1/v2 对照绑定上述外部固定 commit 的 **PyTorch extension 映射**，不是所有模式的通用替换表：

| 原始源码 | v1 映射结果 | v2 映射结果 |
| --- | --- | --- |
| `c10::cuda::CUDAStream` | `c10::hip::HIPStreamMasqueradingAsCUDA` | 保留原名 |
| `c10::cuda::getCurrentCUDAStream()` | `c10::hip::getCurrentHIPStreamMasqueradingAsCUDA()` | 保留原名 |
| `c10::cuda::CUDAGuard` | `c10::hip::HIPGuardMasqueradingAsCUDA` | 保留原名 |
| `c10::cuda::CUDACachingAllocator::get()` | `c10::hip::HIPCachingAllocatorMasqueradingAsCUDA::get()` | 保留原名及 allocator owner |
| `C10_CUDA_CHECK(expr)` | `C10_HIP_CHECK(expr)` | 保留宏名；其内部 runtime 表达式仍转换 |
| `c10/cuda/CUDAStream.h` | `ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h` | `c10/hip/HIPStream.h` |
| `c10/cuda/CUDAGuard.h` | `ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h` | `c10/hip/HIPGuard.h` |
| `c10/cuda/CUDACachingAllocator.h` | `ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h` | `c10/hip/HIPCachingAllocator.h` |

对 `at::cuda`、event、allocator 的其他成员、ATen 宏等同类接口，继续逐项查目标定义，不能据上述几项扩大成“所有 PyTorch API 都无需转换”。框架 stream/allocator/guard 必须复用目标 Torch 的状态，不用强转掩盖类型或 ABI 不匹配。

## 4. 按证据区分原因，再修生成入口

| 观察 | 判断与动作 |
| --- | --- |
| 新目标保留 CUDA 框架名，生成文件却出现旧 Masquerading 名 | 查实际选中模块、v2 检测日志、表合成和 custom mapping；证明代际错配后修入口/版本化规则 |
| 声称已选 v2，但输出仍旧 | 检查是否直接 import v1、额外规则先改写、二次转码或编译仍消费旧 generated；核对生成映射与真实编译 argv |
| 旧兼容头存在，当前 TU 中旧类仍未声明 | 查真正 include 的文件、namespace 和条件宏；属于可见性问题的不能报告成整个包缺 API；也不以随手补旧头替代代际核对 |
| 转换函数退出 0，却没有预期结果 | 查返回的 `hipified_path`、status、过滤/忽略列表和真实文件；不要固定读取原 `.cu` 或猜生成后缀。只有不需要任何改写的输入才可合理复用原文件 |
| 正确框架名字仍编译/链接失败 | 追目标头/库、实际 include 顺序、符号 defined/UND/missing、宏和类型 ABI；不能把所有错误都归于 hipify 代际 |

修复落在原始源码、实际构建入口或可复现的项目转换规则中；生成文件只作检查结果，不只手改 build 副本。已确认 v2 目标时使用匹配实现；旧目标仍可能需要 v1，不强制所有项目切 v2。存在用户指定 provider 时，依其支持的入口处理，不无声替换成别的工具。

只在能力/归因不明时做这种小文件对照。记录构建 Python、Torch/hipify 路径与版本、外部 commit、参数/额外映射、原始→生成结果、目标声明及实际消费命令；复用已有同身份证据，避免每个工程重复全部调查。

规则或工具改变后，在新的任务生成目录从原始源码再生成，检查关键接口，再以原工程编译参数复编受影响 target、核验链接/加载并运行原 GPU 测试。映射正确、完整文件转换、编译链接、运行正确分别报告。怀疑 FastPT/其他 overlay 时先记录实际头/库来源；本节不授权清理、重装共享 Torch 或伪造本地 shim。

## 5. 源码依据

- [PyTorch v2.10.0 旧映射](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/hipify/cuda_to_hip_mappings.py)与 [v2.11.0 新映射](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/hipify/cuda_to_hip_mappings.py)：表项、路径与有效规则需结合各版本处理器查看。
- 外部固定 commit 的 [v1 映射](https://github.com/ROCm/hipify_torch/blob/63b6a7b541fa7f08f8475ca7d74054db36ff2691/hipify_torch/cuda_to_hip_mappings.py)、[v2 映射](https://github.com/ROCm/hipify_torch/blob/63b6a7b541fa7f08f8475ca7d74054db36ff2691/hipify_torch/v2/cuda_to_hip_mappings.py)与 [CLI 选择逻辑](https://github.com/ROCm/hipify_torch/blob/63b6a7b541fa7f08f8475ca7d74054db36ff2691/hipify_cli.py)。

这些版本用于解释差异；现场的 fork、安装头和真实调用链才决定具体工程应采用的映射。
