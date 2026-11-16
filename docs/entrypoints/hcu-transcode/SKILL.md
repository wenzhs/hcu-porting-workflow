---
name: hcu-transcode
description: 将 CUDA 应用源码转换为 HCU/DCU（DTK）可构建的 HIP 工程。用于明确要求 hipify/转码或 CUDA-only 的默认适配路线；核验转换器能力，维护源码映射、构建入口与原测试合同。
---

# CUDA 源码转码适配

本入口固定 `transcode`，遵循[执行约定](../../../docs/contract.md)。只分析时保持只读；已授权适配时沿以下五步完成。指定 provider 失败不自动改用其他 provider 或路线。

| 步骤 | 工作与产物 |
| --- | --- |
| 1. 固定源码 | 核对 ref/HEAD、分支和已有修改；无指定版本沿当前基线新建适配分支，恢复沿用确认分支 |
| 2. 分析 | 查实际构建与依赖、原始/生成根、转换映射和预计写集；按[转码指南](../../../references/code-conversion.md)核验 provider，写[编译前分析](../../../docs/pre-build-analysis.md) |
| 3. 编译 | 运行已核实的有界转换；修复原始源码或版本化规则，同步[构建入口](../../../references/build-adaptation.md)；CUTLASS/hytlass 子集单独核验 |
| 4. 测试 | 核对真实编译和加载的新产物，运行原 GPU/数值与约定测试；[排障](../../../references/common-errors-and-solutions.md)后落实现修复并重测 |
| 5. 总结 | 仓根交付[适配总结](../../../docs/adaptation-summary.md)，保留映射、改动、结果与剩余项，冻结仓内文件后做树外最终检查 |

```bash
python -B -I "<skill-root>/scripts/route_preflight.py" --repo "/path/to/project" --route transcode
```

provider 需要能力证据：工具存在不等于能转换该工程，unknown 不当作不可用或成功。预检不执行转换/构建。输入及用户改动保留，预计写集审计不是转换器沙箱；工具接口见[说明](../../../docs/tools.md)。

宏、PTX/builtin、warp/mask 与同步见[指令指南](../../../references/instruction-porting.md)。不机械替换库/前缀，不放宽测试容差，不因依赖缺失自行排除功能；编译成功只证明编译阶段。
