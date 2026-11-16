# 调查与执行记录工具

所有脚本仅依赖 Python 标准库，以 `python -B -I` 运行。**职责不同，不都是只读工具**：上下文检查和风险扫描不执行项目；环境探测显式运行版本/设备查询，可选编译探针；账本写记录；命令记录器执行调用者给定 argv。preflight 不自动触发任何探针或命令，也不 import 目标包。

默认只调用当前阶段需要的工具，不强制每个工程重复生成全套文件。先读[共同契约](contract.md)，有症状再读[排障指南](../references/common-errors-and-solutions.md)。构建细节按需看 [CMake](../references/build-adaptation.md) / [Torch 扩展](../references/build-adaptation.md)。

## 1. 已有编译上下文与转换写集

`inspect_build_context.py` 只分析调用者显式提供的证据，不执行编译数据库、不 import 项目、不启动编译器。示例：

```bash
python -B -I "<skill-root>/scripts/inspect_build_context.py" --repo "/path/to/project" --compile-commands "/path/to/compile_commands.json" --file "src/op.cpp" --requirement-header "/verified/framework/header.h" --host-macros "/path/to/host.macros" --device-macros "/path/to/device.macros"
```

各证据参数按需要提供，不必每次齐全；`--file` 可重复，使用项目相对路径，且须提供包含这些单元的数据库。`--out` 写全新 JSON 文件，父目录应已存在；stdout 仍输出报告。

- 数据库优先取 `arguments`；`command` 仅按 POSIX 规则分词，不求值 shell、wrapper 或 response file。记录 compiler_entry、argv/cwd、标准、宏、include 和架构；未知/冲突须查真实展开命令，不推断默认标准。
- 框架头的 `#error` 标准诊断只是候选下限，未判断周围预处理条件；“显式标准满足候选”不等于工程兼容。
- `--host-macros` / `--device-macros` 须来自各自真实编译上下文并附生成命令。工具解析空宏并分别展示，不验证文件来源，不把混合 dump 伪装成 host/device 对照。架构宏不等于 device-pass 标识。
- 转换审计可单独调用：

```bash
python -B -I "<skill-root>/scripts/inspect_build_context.py" --repo "/path/to/original" --transform-map "/path/to/map.json" --generated-root "/path/to/generated"
```

映射为非空 JSON 对象，如 `{"src/op.cu":"src/op.hip"}`，两侧相对于原始根和生成根。拒绝输入越界/缺失、链接逃逸、输出碰撞、原始/生成身份混用及计划覆盖。生成根不能等于或包含原始根；合法独立生成子目录仍按路径保护核验。外部只读 include 不会自动成为输入。生成后加 `--outputs-exist` 只核验输出存在，不证明转换等价。

这是**声明的预计写集审计，不是转换器副作用沙箱**，不会自行转换/创建生成文件。更多判别见[编译上下文](../references/build-adaptation.md)与[执行证据](contract.md)。

<a id="changed-build-coverage"></a>

### 改动文件到实际构建目标的核对

在已有 configure/构建记录后，给同一个工具提供改动清单。`--changed-files` 是JSON文件，内容为相对原始repo的非空字符串数组，例如`["benchmark/cache.cu", "include/locks.cuh"]`。从本轮指定基线的Git diff、暂存差异及新增文件整理清单；工具不会自行运行Git，也不能证明清单完整。已删除路径会单列`missing_source`。

```bash
python -B -I "<skill-root>/scripts/inspect_build_context.py" --repo "/path/to/source" --changed-files "/path/to/changes.json" --compile-commands "/path/to/build/compile_commands.json" --cmake-index "/path/to/build/.cmake/api/v1/reply/index-example.json" --configuration Release --dependency-manifest "/path/to/dependencies.json" --scope-ledger "/path/to/ledger.json"
```

只提供已有且相关的证据，后面四个参数均可省略；`--configuration`仅在有File API时使用，多配置必须明确选择。无File API仍可检查编译单元，目标归属保持未知。工具不创建查询、不configure、不编译。

| 输入 | 格式与边界 |
| --- | --- |
| `--cmake-index` | 真实[File API index](https://cmake.org/cmake/help/latest/manual/cmake-file-api.7.html)，沿`objects`读取codemodel v2、选定configuration和target JSON；不靠扫描CMake文本猜foreach展开结果。记录目标源/构建目录、source及compile-group语言；缺记录或错误配置不按默认配置补齐 |
| `--transform-map` / `--generated-root` / `--outputs-exist` | 复用上节原始→生成相对路径映射，关联已存在生成文件；覆盖核查不能用尚未生成的计划产物 |
| `--dependency-manifest` | JSON数组，例如`[{"file":"/path/to/source/a.cu","depfile":"CMakeFiles/a.dir/a.cu.o.d"}]`；file必须绝对路径，depfile相对该编译单元的compile DB cwd。多个cwd时加`directory`明确选择。只接受命名了该TU的真实Make依赖记录；支持续行、转义空格、盘符和空phony规则，不求值Make变量/命令 |
| `--scope-ledger` | 读取账本中的目标状态/回执引用，并核对repo及源码身份；按target名字关联只算候选，重名/手工feature仍人工核对完整ID。不会读回执后代验、修改账本或自动升级covered |

输出`coverage.files`逐文件包含`status`、`generated_file`、`compile_units`、`included_by`、`targets/target_details`、`gaps`和`ledger_entries`。`evidence`绑定实际读取的文件哈希。主要结果如下：

| 具体case | 预期结果/后续动作 |
| --- | --- |
| 修改`benchmark/cache.cu`，但`BUILD_BENCHMARKS=OFF`，该配置没有对应编译单元 | `missing_compile_unit`；补实际构建/验证，或按范围合同记录未验。不会因默认all通过忽略该文件 |
| `foreach`展开a/b两个目标，共用`include/locks.cuh`，两个TU的.d均记录该头 | 关联两个编译单元和目标，保留各自目录/语言；配置关系不证明二者都已编译成功 |
| 修改`src/op.cu`，编译实际消费`generated/src/op.hip` | 按映射关联生成文件与编译单元；没有映射且无法关联时保留缺口，不按basename猜 |
| 修改头文件，但没有.d，或.d包含无法解析的变量/不属于声明的TU | `unknown_header_dependencies`或`partial_dependency_evidence`；不能因头文件不直接出现在compile DB就判断未构建，也不能猜它已覆盖 |
| compile DB有该文件，但没有build/测试证据 | 至多`configured`；`compiled/executed=false`，`build_evidence/runtime_evidence=not_verified` |
| 账本声明covered但源码已变 | `source_identity_match=false`；声明仅回显，仍按账本check及当前原始记录核验 |

`configured`只表示找到了编译单元。File API与compile DB按source路径关联，不保证同源多目标的每条flags/对象记录一一匹配；.d只证明记录中的include关系，不证明当前宏分支、模板实例或GPU执行。缺少任何一层证据保持未知。CLI返回0仅表示报告生成，必须查看每行gaps；原`--file`精确选择的缺项仍报错。

## 2. 环境事实与可选编译探针

在已初始化的目标 shell/容器中，用工程实际解释器显式调用：

```bash
python -B -I "<skill-root>/scripts/probe_environment.py" --evidence-dir "/path/to/new-probe" --out "/path/to/new-probe/probe.json"
```

默认查询工具路径、hipcc/hipconfig 版本、rocminfo 设备、包元数据和 Torch 安装文件/HIPIFY 线索，不导入 Torch。它**不收集 FastPT 元数据、不调用 FastPT 帮助或激活模式**。输出的 `capabilities` 只有通用 HIP/native_backend 证据，backend 仍需按 target 核查；不能补成版本匹配或功能已验证。

- `--evidence-dir` 可选，必须是全新目录；保存各命令的完整 stdout/stderr 文本日志、argv、rc、超时信息，以及编译时的源和对象身份。显式要求编译探针或证据目录时报告 `read_only=false`，不再称为纯静态调查。仅 stdout JSON 中的截取文本不能代替完整日志。
- 显式 `--compile-probe [--arch gfx936]` 才编译内置的极小 shuffle/ballot kernel 并导出宏；需 hipcc 与单一已核实 gfx，`--timeout` 控制单条命令。它不构建目标工程，也不运行 GPU kernel。
- 单个 compile probe 只有 **rc=0 且预期对象存在、非空** 才 passed。CLI 退出 0 只表示报告生成，可能包含 failed/skipped/unknown；逐项看 `compile_probes`。按实际 dtype、接口族和模板另做最小判别，不把内置浮点探针当全部 API 支持。
- 未设 evidence 目录时编译临时源/对象会被清理，摘要不能提供持久产物路径。宏 dump 标为 compiler-default，可能混合多个 pass，不能用来比较 host/device。
- 找不到工具保持 unknown，不证明没安装；先核实解释器和正常启动链，不擅自换版本。rocminfo 的 `launch_limits` 是硬件上限，`kernel_launch_limits` 未测；真实 kernel 元数据、block/grid/共享内存仍须另查。

## 3. 限定范围扫描与摘要

```bash
python -B -I "<skill-root>/scripts/source_risk_scan.py" --repo "/path/to/project" --route native-hip --files-from "/path/to/files.json" --summary --json-out "/path/outside/project/scan.json"
```

`--files-from` 是**JSON 数组**，如 `["src/op.cu","setup.py"]`，不是逐行文本；文件相对于 repo，需实际存在且在可选 `--target` 内。它与 `--transform-map` 互斥。映射扫描用非空原始→生成对象，`--map-side source|generated` 选择 key/value；`--repo` 必须相应指向该侧的根。扫描器只验证选择侧，不替代上一节的双侧转换审计。

- `--summary` 在 stdout 省略 findings/skipped，显示 `groups`、`summary.by_category` / `by_card` 等；`--json-out` 保存完整报告结构，且必须在源码树外、不可覆盖已有文件。
- `groups` 按 category/subtype/card 聚合，最多给少量样例，状态仍为 unreviewed。`--max-findings` 默认 5000，只截断明细，不改变总计；`truncated=true` 时扩大限额或按范围分批取得全部位置，不能将截断 JSON 当完整明细。
- 检查 `skipped`、显式 `--exclude-dir` 和默认依赖目录排除；必要时用 `--include-dependencies`。显式清单按给定文件扫描，仍受文件类型、大小与排除条件约束。源码身份始终绑定 repo 整树，不因扫描变窄就成为局部身份模型。
- 命中是候选、无命中不是通过；没有宏展开、完整构建依赖图或自动生成源码追踪。分组只减少重复阅读，不替任何命中关闭风险。

按接口族进入对应规则：宏/平台/架构 → [编译上下文](../references/build-adaptation.md)；warp-sync/warp-size/lane-mask/barrier/cooperative-groups → [warp/wave 卡](../references/instruction-porting.md)；asm/builtin/库/runtime → [源码语义](../references/code-conversion.md)；Python/Triton 低精度与 backend-dependency → [依赖与后端](../references/code-conversion.md)。source-dependency 先查 [HCU 源码依赖](../references/code-conversion.md)。路线专属限制留在对应指南，不推广成通用 HIP 限制。

## 4. 范围账本与完整 ID

`adaptation_ledger.py` 写进度，不是只读预检。账本放源码树外，保留静态枚举加人工补充的原约定范围；动态 target/其他构建系统不由枚举自动识别。

```bash
python -B -I "<skill-root>/scripts/adaptation_ledger.py" init --repo "/path/to/project" --out "/path/outside/project/ledger.json" --feature "前向" --requirements "/path/to/requirements.json"
```

此例的 requirements 文件可以是 `{"feature:前向":["integration"]}`，将额外阶段绑定该完整 ID。基础 build/load/gpu/correctness 不会被覆盖或降低；把功能名直接命名成某个阶段时，该阶段也会成为该项要求。requirements 中未知 ID/阶段被拒绝。先读取账本 `entries` 的实际 ID；重名 target 带文件/行号等消歧后缀，不能凭简称猜 ID。

覆盖回执须有 `covered_entries`，例如字段片段 `"covered_entries": ["feature:前向"]`，这不是一份完整回执。完整回执还需当前源码、实际产物/加载、GPU 计数、原合同和阶段证据：

```bash
python -B -I "<skill-root>/scripts/adaptation_ledger.py" set --ledger "/path/outside/project/ledger.json" --id "feature:前向" --status covered --route native-hip --receipt "/path/to/result.json" --target-features "feature:前向"
python -B -I "<skill-root>/scripts/adaptation_ledger.py" check --ledger "/path/outside/project/ledger.json"
```

`--target-features` 是可选**完整 ID 列表**，只缩小回执的 `covered_entries`，不能扩大或弥补缺 scope；旧的标签字段不提供覆盖。set/check 都验证 scope、route、阶段、当前源码与回执/产物身份。源码变动后旧覆盖 stale，旧账本不能新记 covered；保留旧范围和历史，为当前固定源码重建账本并重验，不编辑哈希绕过。环境/树外参数变化仍需人工核验。

未完成用 owed，明确阻塞用 blocked 并记理由；scoped_out 需要理由和**真实用户批准记录**，不因缺依赖、失败或 skip 自动排除。`check` 返回 0 表示账面全部有交代（可包含批准的 scoped_out），3 表示仍欠工作或空账本，2 表示无效/过期；不等于整个工程所有 GPU 功能都通过。

## 5. 显式命令记录器

`record_command.py` 仅执行调用者已经授权并审查过的 argv，在指定 cwd **继承当前已初始化环境**。不会自己 source 环境、选版本、登录 SSH、安装、重试或启动下一阶段；但调用者传入的任意脚本本身仍可能有副作用，记录器不会阻止它。不要将记录器或 JSON 证据当成授权与沙箱。

接口示例中的构建目录、target 与预期输出须替换为本项目实际值：

```bash
python -B -I "<skill-root>/scripts/record_command.py" --cwd "/path/to/project" --out-dir "/path/to/new-build-evidence" --timeout 900 --expected-artifact "build/example" -- cmake --build build --target example
```

- `--` 后是 executable 和逐项参数，不默认解释 shell 字符串、管道或 `source`。确需 shell helper 时传已审查的脚本 argv，不能因此绕过激活、安装或环境变更授权。
- `--out-dir` 必须全新；保存 `command.json`、原始 `stdout.log`/`stderr.log`，正常封存后记录日志哈希；中止且不能确认后代退出时日志仍可能追加，详见下文。记录 argv/cwd、选定环境键、起止/耗时、真实 child `exit_code`、超时和错误。子进程继承全部当前环境，但 JSON 不是完整环境快照；日志和记录对外共享前检查敏感信息。
- `--expected-artifact` 可重复，可省略；指定时必须是**执行前不存在**的新文件，路径不重复且不能与记录器证据目录重叠。不能把其自产的 stdout/stderr 日志当作构建产物。执行后检查非空与哈希，既有产物会拒绝启动，不据 rc0 认定旧产物新鲜。
- `artifact_check_requested` 表明是否要求产物检查；未指定产物时为 false，`expected_artifacts_present=null`，不是“已验证有产物”。`command_succeeded` 单独描述命令执行结果，不推断产物/加载。CLI 超时返回 124、中断返回 130、启动错误 127、缺预期产物/参数错误 2；子进程非零另行透传或映射信号，JSON 保留真实 child rc，不用外层码覆盖它。
- 有限 `--timeout` 默认 120 秒。超时/中断时 POSIX 只清理本次启动的进程组，Windows 只终止直接子进程，不保证其所有后代退出。因此 Windows 中止或清理失败时 `evidence_finalized=false`，日志仍可能增长，不生成日志稳定哈希/大小或产物结论，相关字段为 null。先确认本任务后代进程状态再使用证据或重试，不能盲重放或杀其他任务。
- 只包装会等待其工作完成的前台 argv；主动脱离/另建进程组的后台任务由调用者另行核查。`evidence_finalized` 描述记录器可完成的封存，不是进程存活或外部写入的安全证明；记录器不是共享资源调度或通用进程清理工具。

命令记录不推断测试计数、GPU backend、数值结果或完整账本覆盖。需要执行回执时人工汇总到 `adaptation-result`，保留原统计口径和失败状态；详见[执行证据](contract.md)与[测试与数值](../references/common-errors-and-solutions.md)。
