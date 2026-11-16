# FastPT 不转码路线

由独立 `fastpt-non-transcode` 或总入口使用。先读[共同契约](../docs/contract.md)；本页集中保存 FastPT/GPUFusion 细节，不要求其他路线先失败。沿入口五步推进；本页只提供不转码专属处理。

## 识别工程、版本与必需能力

按 target 固定 CUDA-facing 源码、原构建入口、必需接口族/库和原数值合同，保留显式路线与 provider 约束。`capabilities.fastpt` 分开记录 installed、version_compatible、invocable、feature_probed、hardware_validated；未知为 null，明确失败为 false。先查实际 API/模板实例，不机械凑若干假设；既往 hardware_validated 不认证新目标。

计划入口为 `route_preflight.py --route non-transcode --request ...`，不导入 Torch/FastPT，不运行帮助命令、source、编译或测试。通用 `probe_environment.py` 只提供 HIP/框架事实，**不收集 FastPT 版本或调用能力**。FastPT 安装、launcher、所需 Torch/DTK 与 GPUFusion 版本需本路线另外核验。

FastPT 与 Torch 主次版本强绑定，但主次一致仍不足以证明全部能力。版本不匹配可能在包初始化时抛出 `ValueError … torch version X does not match the required version Y`，`--help` 就失败，并非只有 `ImportError`。各 CLI 可能经过同一初始化检查；未实测入口保持 unknown，不因另一个命令存在就写可用。明确版本失败返回 blocked，不删检查、不降级共享 Torch，也不用 `-T` 绕过；native 不受此路线前提影响。

只在已获授权时显式调用经过核实的 launcher；例如记录帮助入口的实际结果：

```bash
python -B -I "<skill-root>/scripts/record_command.py" --cwd "/path/to/project" --out-dir "/path/to/new-help-evidence" --timeout 30 -- /verified/path/fastpt --help
```

这不是激活命令，也不是目标功能证明。`-C/-E` 可能替换 Torch 安装目录文件，新 shell 不能撤销；只有真实授权的独占隔离安装才可激活。

### GPUFusion 特性清单

以下表格来自现有摘要中标注为 DTK《CUDA程序适配使用手册》DTK26.10 的章节/条目；未包含可逐条复核的手册原文，作为待核查清单。执行时须以目标同版本手册、头文件与实测为准，不得无版本外推。扫描用 `source_risk_scan.py --route non-transcode`，查看 `groups`、`summary.by_category` 和 `findings` 的 category/subtype，不能把分类命中直接当支持结论。

| 能力或限制 | 摘要标注章节 | 最小判别与处理 |
| --- | --- | --- |
| 默认 warp64，`GPUFUSION_ENABLE_WARPSIZE32` 为实验性软兼容，有性能成本，不适合 warp 间数据依赖 | §4.2.9 | 查 `warp-size` / `warp-sync`；核实当前 wave 与参与集合，不默认注入开关 |
| 不支持 Dynamic Parallel | §4.3 | `dynamic-parallelism:device-side-launch` / `device-launch-api` 需确认；`rdc-flag` 可能只是分离编译，不能单独判阻塞 |
| cublasLt、cufftw 等非基础库未支持或效果不足 | §4.3 | 查 `cuda-library`，按目标版本的必需 API、dtype/layout 实测；不足则报 blocker |
| 已删除静态库支持 | §4.3 | 查 `static-runtime`；改为动态链接前确认交付合同，不静默改要求 |
| 不支持 CUDA/HIP 混用 | §4.3 | 查 `mixed-hip-cuda` 的实际 include 链和互斥分支；名字存在不证明混用 |
| 内嵌 PTX 需显式 `-finline-asm-ptx` | §4.3 | `inline-asm:ptx` 仍逐指令核对编译、结果和同步，开关不保证等价 |
| CUPTI profiler 需要特定 `HSA_TOOLS_LIB` | profiler 一节 | 按实际版本配置及授权核查，不因工具可运行就证明 GPU 正确性 |

这些限制不推广到 native 或转码，原生 HIP 库能力也不能替代 GPUFusion 的能力。可处理项记方案，不能满足的写 blocked；重构、换库、改路线或改变交付范围由用户决定。

## 准备与授权后构建

静态准备无需 FastPT，也不修改输入：

```bash
python -B -I "<skill-root>/scripts/fastpt_non_transcode_prepare.py" --repo "/path/to/project"
```

实际写入须明确 `--write`，并提供核实的原构建 argv、launcher、GPUFusion 路径与能力证据。以下只展示接口，不能复制示例 true 值来代替调查：

```bash
python -B -I "<skill-root>/scripts/fastpt_non_transcode_prepare.py" --repo "/path/to/project" --fastpt "/verified/path/fastpt" --gpufusion-cuda-home "/verified/gpufusion/cuda" --arch gfx936 --build-argv '["cmake","-S","{ROOT}","-B","{BUILD_DIR}"]' --capabilities-json '{"fastpt":{"installed":true,"version_compatible":true,"invocable":true}}' --write
```

- `--build-argv` 是 JSON argv；`--build-entry` 仅检查 target 相对文件存在，不替代原命令。保留 CUDA 语言，不改成 HIP 或加 HIP 兼容头。
- helper 支持 `{ROOT}`、`{BUILD_DIR}`、`{CUDA_HOME}`、`{ARCH}`、`{TORCH_PREFIX}`；Torch prefix 须显式提供，不导入框架猜值。一次执行一个原项目命令，configure 成功不等于 build 完成。
- Git 工程在所选适配分支处理并记录 ref/HEAD/改动；无 Git 才建立可见新副本并报告路径。保留原始字节与输入，不按 `.hip`、build/dist 等名称删除；`--out` 不迁移 Git 工程，`--force` 不授权覆盖旧目录。
- 固定输出名碰撞、越界、链接逃逸和副本重叠均拒绝。静态 runtime 调整另需 `--allow-dynamic-runtime` 且先确认交付合同；不是激活授权。
- 结果只是 **prepared-not-built**，包含计划、报告、身份记录与 helper，不执行生成脚本。

执行前核实隔离安装、资源和真实授权。helper 要求 `FASTPT_ACTIVATION_APPROVED=1`、`FASTPT_ISOLATED_INSTALL=1`，变量本身不授予权限，也不创建隔离安装。`-C` 构建与 `-E` 运行分开，不能混入 `-T` 或原生 HIP 环境。

用 `record_command.py` 记录已授权 helper/原项目 argv，可保留实际子进程 rc 与完整日志；任意脚本仍非沙箱。若自行使用 tee，调用者启用 pipefail 并即时保存真实 rc。详细接口见[工具](../docs/tools.md)与[执行约定](../docs/contract.md)。

## 最小验证、验收与失败

核对本次产物及实际加载路径后做最小 GPU/数值验证，再按约定范围完成测试。保留原 oracle、容差、ABI/layout/输入副作用；按完整 ID 绑定账本 covered，额外阶段用 `--requirements` 声明。主项、subtest、collection error、skip 分账；全 GPU skipped、CPU fallback、旧 so、仅 import 或仅准备均不能通过。

HCU 优先等价的 `tile=warpSize` 实现，64 仅限实测 wave64；联动任务映射、mask/shuffle、存储与 block/grid，保留公共合同和格式 L。无法等价时用原逻辑 tile 的 phase/串行回退。不要盲改 32→64、默认关 cache 或默认启用实验 warp32；性能需真实对照。

出现症状按[排障指南](common-errors-and-solutions.md)定位；构建细节看[构建指南](build-adaptation.md)，指令/warp看[指令指南](instruction-porting.md)。

本包携带的分类器只给首错建议，仍需人工复核，不执行修复：

```bash
python -B -I "<skill-root>/scripts/classify_fastpt_failure.py" "/path/to/build.log" --route non-transcode --phase build --json
```

保留本路线和失败阶段，不自动卸载包、清源码或切转码。历史研究不认证当前候选；具体轮次是否完成由其真实日志/回执说明，不沿用历史结果。
