---
name: hcu-native-adaptation
description: 通过工程已有的 HIP/ROCm/HCU 构建路径适配并验证 DTK 应用，包括 CMake、Make 和框架扩展。用于明确选择原生 HIP 的任务；按实际构建分支核验源码、依赖和原测试。
---

# 原生 HIP/HCU 适配

本入口固定 `native-hip`，遵循[执行约定](../../../docs/contract.md)。只分析时保持只读；已授权适配时沿以下五步完成。失败保留原生路线并定位，不静默切换。

| 步骤 | 工作与产物 |
| --- | --- |
| 1. 固定源码 | 核对目标 ref/当前 HEAD、分支和已有修改；未指定用当前基线建适配分支，恢复任务沿用确认分支 |
| 2. 分析 | 查实际 HIP/HCU 入口和依赖，核实 `.cu`/`CUDAExtension` 背后的框架分支；写[编译前分析](../../../docs/pre-build-analysis.md) |
| 3. 编译 | 按[构建指南](../../../references/build-adaptation.md)复用仓库流程和已初始化环境，项目原有 HIPIFY 保留，不重复转换；最小补丁隔离 HCU 差异 |
| 4. 测试 | 核对本次产物及加载路径，执行原 GPU/数值测试和约定范围；异常按[排障](../../../references/common-errors-and-solutions.md)定位、实现修复并重测 |
| 5. 总结 | 仓根交付[适配总结](../../../docs/adaptation-summary.md)，记录环境、改动、结果和未验证范围；冻结仓内文件后完成树外最终检查 |

```bash
python -B -I "<skill-root>/scripts/route_preflight.py" --repo "/path/to/project" --route native-hip
```

预检只输出计划，不执行探针、转换或构建。脚本按需使用，见[工具接口](../../../docs/tools.md)。遇到 warp/PTX/builtin 看[指令指南](../../../references/instruction-porting.md)，类型、库或现有 HIP 宏分支问题查[源码语义卡片](../../../references/code-conversion.md)。这些是核查原生实现的规则，不是要求额外转码。

核验目标 DTK 头文件/库/示例及对应版本官方资料，不按 ROCm 上游版本推断当前 DTK 支持。保留原测试、容差、公共布局及非 HCU 分支；只 import、CPU fallback、全 GPU skipped 都不构成功能完成。
