---
name: fastpt-non-transcode
description: 使用匹配的 FastPT/GPUFusion 在 HCU/DCU（DTK）直接编译原 CUDA 源码并测试。仅用于用户明确指定不转码、直接编译原 CUDA 源码或指定 FastPT 不转码路线的任务。
---

# FastPT 不转码适配

本入口固定 `non-transcode`，遵循[执行约定](../../../docs/contract.md)。只有环境和当前 target 的能力证据匹配，才能进入实施；失败不静默转成 HIPIFY。

| 步骤 | 工作与产物 |
| --- | --- |
| 1. 固定源码 | 核对 ref/HEAD、分支、已有修改；未指定用当前基线建适配分支，恢复任务沿用确认分支 |
| 2. 分析 | 查原 CUDA 构建入口及依赖，按[不转码指南](../../../references/fastpt-non-transcode.md)核验 FastPT/框架/GPUFusion/DTK，写[编译前分析](../../../docs/pre-build-analysis.md) |
| 3. 编译 | 在授权隔离环境按现场帮助核实并激活，保留 CUDA-facing 源码/编译链；准备器只准备，沿仓库原命令构建 |
| 4. 测试 | 核对产物/加载路径，执行原 GPU/数值与约定测试；按[排障](../../../references/common-errors-and-solutions.md)实现修复并重测 |
| 5. 总结 | 仓根写[适配总结](../../../docs/adaptation-summary.md)，说明激活/产物/验证/遗留；冻结仓内文件后做树外最终检查 |

```bash
python -B -I "<skill-root>/scripts/route_preflight.py" --repo "/path/to/project" --route non-transcode
```

通用 probe 不收集或激活 FastPT；本路线按需取实际入口/help、包和编译链证据。预检不 import、激活或构建，见[工具接口](../../../docs/tools.md)。FastPT 模式可能改写安装包，新 shell 不能撤销文件改写；先核实授权范围和恢复方式。

HCU 特化保持原接口、布局、容差和非 HCU 分支；warp/PTX/builtin 看[指令指南](../../../references/instruction-porting.md)。激活成功、编译成功、加载成功、GPU/数值通过分别记录；无 GPU 命中或全 SKIP 不能完成。
