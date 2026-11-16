# 执行约定（schema_version 1.0）

四个入口共用：固定源码 → 编译前分析 → 适配编译 → 原测试 → 总结交付。入口控制路线，处理器不重新选路。详细构建、转换和运行机制只放在相应参考指南。

## 范围与工作区

| 情况 | 处理 |
| --- | --- |
| 用户固定路线/转换器 | 保留选择，失败记录 blocker；不静默切换。`cutlass-hytlass` 是 transcode 专项别名 |
| 未固定路线 | 当前 target 可用原生路径优先；CUDA-only 优先转码。原生未知先补证据；纯 HIP 无可用原生路径且无 CUDA 输入时不能凭空转码 |
| 只分析 | 只读调查、返回计划；preflight 不执行探针、import、激活、转换或构建 |
| 已授权适配/修复 | 在该源码/环境/功能范围内推进；JSON 授权字段或确认变量不能扩大用户授权和宿主权限 |
| 有 Git | 核对指定 ref/HEAD/dirty 后在当前源码目录建适配分支；无指定 ref 用当前 HEAD，恢复任务沿用确认分支。不 reset/stash 用户修改，不自动切版本 |
| 无 Git | 建全新可见副本，报告绝对路径，不含 VCS 元数据；输入/输出不相同、不父子重叠。`--out` 仅用于无 Git 副本 |

原输入字节、行尾和已有改动必须保留；不按 build/dist/`.hip` 文件名清理。写入、chmod、固定报告名和补丁均检查碰撞、链接/junction、越界，`--force` 不授权覆盖未知目录。修复原始源码/构建/版本化转换规则；外部只读 include 不成为转码输入。映射审计只检查声明的预计写集，不是任意转换器或构建命令的副作用沙箱。

## 能力与实现边界

能力分开记录 `installed`、`version_compatible`、`invocable`、`feature_probed`、`hardware_validated`；未知用 null/unknown，false 只用于已确认失败。附环境、版本、路径与证据来源，用户声明不冒充实测。

沿用已初始化环境，不凭空选 DTK/Conda 版本或改启动文件；新 shell 不能撤销已发生的包文件改写。核实真实 argv/cwd、框架要求、host/device 宏、完整 gfx；不按框架版本猜 C++ 标准或把架构常量当 device-pass。builtin、库、低精度和 backend 按版本/操作核查，见[构建](../references/build-adaptation.md)、[转码](../references/code-conversion.md)、[指令](../references/instruction-porting.md)。

保持非 HCU 分支、公共 API、ABI/layout、输入副作用和原数值合同。wave64 下整 wave 实现必须联动任务映射、mask/shuffle、共享存储和 block/grid，保持格式宽度；不能等价转换时用原逻辑 tile 的 phase/串行回退。不盲改 32→64，不以默认关闭 cache 代替修复。

## 结果与证据

| 记录 | 含义与必要边界 |
| --- | --- |
| `adaptation-plan` | `plan_status=planned/blocked/needs-information`，`executed=false`，执行阶段均 not-run |
| `adaptation-result` | 分别记录 prepare/transform/build/load/gpu/correctness/integration/performance/original-platform，状态不合并成一个“通过” |
| `command-record` | 实际 argv/cwd、子进程 rc、日志和可选新产物；不推断 GPU、测试计数或功能回执 |
| 编译探针 | rc0 且有非空对象只证明该探针编译；不证明 GPU 运行或其他 API |

执行结果绑定源码、配置、环境、产物/实际加载路径；需要时保存 oracle、shape/dtype/layout、设备与进程模型。只转换、只 import、旧 so、CPU fallback、全 GPU skipped 不算完成。构建、加载、GPU、数值阶段分别验收；性能/集成/原平台回归按约定运行，未做写 not-run，walltime 不是性能结论。

- 原测试保留断言与容差。主项、subtest、collection error 分账；FAIL/SKIP/崩溃/超时/未运行明确区分，复测不重复计为新用例。
- rc 必须来自实际子进程；使用管道时即时保存 rc。整套失败、原样单项通过都保留，不能用单项覆盖整套失败，见[排障](../references/common-errors-and-solutions.md)。
- 源码/环境/配置/依赖变化后重新核验，哈希只证明身份和一致性，不证明执行真实性。诊断有界，保留首错、输入、dispatch、尝试和 blocker，不自动禁用功能或降精度。
- 证据是必需信息，**不强制重复生成一套文件**；已有完整日志、编译数据库、JUnit 等可直接引用，按问题使用[工具](tools.md)。

## 分析、总结与范围账本

编译前保存 [ADAPTATION_ANALYSIS.md](pre-build-analysis.md)；交付保存 [ADAPTATION_SUMMARY.md](adaptation-summary.md)，两者位于实际编译仓根。只读咨询/仅预检不强制落盘；同任务增量维护，不覆盖无关同名文件。总结保留稳定证据索引，历史结果绑定其真实版本，不改标签冒充本次验证。

范围账本位于源码树外。每个 target/必需功能须有 covered、scoped_out、owed 或 blocked 的交代：

| 状态/参数 | 检查 |
| --- | --- |
| covered | 有效回执的 `covered_entries` 显式列完整 ID（含消歧后缀），至少 build/load/gpu/correctness passed；简称/无 scope 不提供覆盖 |
| `--target-features` | 只缩小回执的完整 ID 集合，不能扩大或替回执补 scope；set/check 同规则 |
| `init --requirements` | 完整 ID → 额外阶段列表，可增加 integration/performance 等，不降低基础要求 |
| scoped_out | 有具体理由和真实用户批准；缺依赖、FAIL/SKIP 不自动成为排除 |
| owed / blocked | 未完成 / 已确认阻塞，保留原范围及原因 |
| stale | 整树变化后旧覆盖失效，也不能在旧账本追加 covered；保留历史和原范围，冻结新源码并重新验证 |

静态枚举不是完整构建图，动态目标/其他构建系统用 `--feature` 补足。空账本不代表完成，check=0 只表示账面有交代，不保证所有条目通过。树外环境/配置变化不能全由工具自动检测，仍须核验。

涉及可选目标、动态目标或共享头文件的修改，用实际构建配置核对“改动文件→编译单元→target→原验证入口”。可复用[改动覆盖检查](tools.md#changed-build-coverage)关联已有证据；缺少依赖或配置时保留 unknown。进入编译数据库只表示已配置，工具关联和账本声明都不能替代真实构建、当前有效回执与运行结果。

**收尾顺序：先完成仓内分析、总结和其他文件，再冻结最终快照，在树外生成最终身份、回执和原范围账本，最后 check。** 仓内总结参与整树身份，最后 check 后再改一行也会使覆盖过期。总结不写包含自身的整树哈希；引用稳定树外证据路径，最终检查结果写树外交付记录。后续命令改变仓内文件时更新快照并重验，不忽略报告或改哈希绕过。

## 恢复、评测与交接

恢复同一任务：核对当前源码、配置、环境、产物及用户修改，使用本任务证据；远端先确认会话有效，状态不明的写入/构建先查结果，不盲重放。

独立冷启动：本轮原始输入、冻结 skill、全新任务目录和专用缓存；不借上一轮组件补丁、调查答案、生成树、产物或验收回执，不清共享缓存。同轮不改 skill，需改先结束该轮、新建 run_id。维护者提炼的方法可以更新到下一轮，历史组件答案不提供给执行者。普通恢复不算独立评测。

具体算子交接包含源码哈希、文件/函数、路线、gfx/wave/DTK/编译器、原 argv/cwd、失败输入、oracle/原容差、原始样本和禁止项。补丁落原始源码，接收方重验构建/加载/GPU/数值；下游口头“通过”不能替代证据。

## 工具与导出约定

辅助脚本仅依赖标准库，不要求安装目标框架。路线提供 `plan(request: dict) -> dict`，request 含 repo、可选 target、selection_mode、arch、build_command argv、provider/specialization、required_features、capabilities。计划 commands 的 phase/argv/cwd 与 `executed=false` 保留；命令不接受隐式 shell 字符串。准备器默认 dry-run，显式 `--write` 只准备，不运行生成脚本。

独立包自包含，不读取兄弟 skill 或开发机路径。CLI 推荐 `python -B -I`，导入包内模块前执行同源完整性检查，缺资源、版本/哈希不一致、字节码缓存均拒绝；重新导出干净目录，不删除 manifest 或改哈希绕过。完整性不是签名、环境隔离或构建沙箱。离线回归不等于真实 HCU 应用验证，具体轮次结果见维护记录。
