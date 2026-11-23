# 常见适配、测试问题以及解决方法

按“现象→证据→原因→代码/配置修复→原测试复验”处理。经验给调查方向，不以关键词代替根因；构建做法见[编译构建适配](build-adaptation.md)，指令细节见[指令处理](instruction-porting.md)。

## 1. 编译与加载故障

| 问题 | 先定位什么 | 处理与复验 |
| --- | --- | --- |
| 找不到CMake包/库 | 当前config、导出target、实际prefix、条件开关 | 修实际包/路径/target；重新configure，确认GPU目标未静默关闭 |
| missing header | 真实编译argv、include graph、生成头/转换集合 | 修源列表/解析链/生成规则；不直接新增无依据shim |
| 转码后PyTorch HIP/Masquerading类型未声明 | 实际hipify v1/v2、最终映射、目标Torch的header/namespace及真实消费的生成文件 | 按[PyTorch HIPIFY案例](pytorch-hipify.md)区分代际错配、include缺失、额外规则和旧产物；先修转换入口/规则，不先造兼容类 |
| 类型或重载不匹配 | 目标函数原型、enum族、host/device类型及实际分支 | 做最小显式映射/条件隔离；不同handle不强转 |
| 宏/API/builtin不匹配 | 宏定义来源、host/device预处理、gfx/version gate、当前DTK声明 | 选当前版本真实可用实现；不改大版本宏或机械替换前缀 |
| undefined symbol | 缺失符号、caller、owner source、最终库defined/UND/missing | 补正确source/export/link闭包；caller保留而owner缺失时先补owner，不先加wrapper |
| rc0却无产物 | 是否只configure/prepare、目标是否关闭、预期文件时间/身份 | 执行真实build/package，确认新产物；不能据工具rc0写通过 |
| import到旧so | 解释器、安装位置、RPATH/搜索路径、实际maps和hash | 加载本次正确产物后重新运行GPU测试 |
| OOM/超时/无输出 | 进程状态、日志增长、资源和编译器活动 | 先区分仍运行、中断和失败；处理已确认的资源问题后再继续，不盲重放/杀其他任务 |

## 2. HCU卡死：先区分执行进度与内存可见性

HCU上的代码不能依赖CUDA独立线程调度（ITS）假设。逻辑tile、同wave锁依赖、跨block或跨kernel等待都可能影响进度；一次无输出/GPU利用率低不能单独证明死锁。

长测试（例如 300 秒仍未结束）也不能单独证明死锁。记录阶段进展、初始化和析构位置、日志增长、进程与设备状态，再定位首个无进展点；涉及 kernel hang 时沿用 `HSA_DISABLE_CACHE=1` 单变量对照规则，不把诊断开关写成根因修复。

| 现象/结构 | 风险机制 | 候选修复 | 必须验证 |
| --- | --- | --- | --- |
| 同一物理wave内自旋锁，另一些lane负责释放 | 等待者占住执行路径，释放者无法前进 | 重映射为完整wave承接独立任务，或明确phase/串行顺序；不要只加fence | 所有参与者最终可运行，锁协议、mask、原子序和原业务结果 |
| tile小于warpSize，例如wave64中两个tile32 | 逻辑组独立执行/同步假设与物理wave不符 | 能等价时tile=warpSize并重算任务/grid；否则原tile逐phase执行 | 上下半组、lane31/32/63、尾部、存储和输出不串扰 |
| 两个kernel互相barrier/轮询 | stream顺序可能让被等待者未启动；并发/驻留和内存可见性也可能不足 | 若算法允许，改host事件/分阶段launch或有明确进度保证的协议 | launch/stream依赖、resident数量、发布/获取作用域和完成条件 |
| 部分lane提前return后仍有block barrier | barrier参与集合不一致 | 让所需lane继续参与同步，只屏蔽其无效数据操作 | 所有返回路径和同步点，包括尾块 |

### 案例：两个tile32不能直接合并成一行

原wave64里tile0写row0、tile1写row1，tile间又通过锁等待。将32改64可能消除部分分歧，却会合并两行和改变输出合同。可选方案：

1. **整wave方案**：每个wave只处理一行，重新分配row/grid、mask、shuffle与shared/temp；输出行数和格式保持原样。
2. **原tile的phase方案**：按原任务顺序逐phase处理row0/row1，并证明每phase等待的发布者已经完成；若仍互相等待，单纯串行也会死锁，必须重构依赖。

两者都需原测试与边界输入，不能凭一个primitive探针通过就证明整个锁算法可前进。性能在正确性完成后另做同条件对照。

若 `io_by_cpu`、`mapped`、`fast fallback` 等 API 条件会进入不同 kernel，必须逐条核对 API → condition → kernel → lock/release → launch 的同构路径，不能只验证其中一个分支。

## 3. HSA_DISABLE_CACHE=1：单变量诊断与最终修复

1. **在两个新进程中做对照。** 固定构建产物、设备、输入、原测试用例和超时。先用 `env -u HSA_DISABLE_CACHE <原命令>` 复现阻塞，再用 `env HSA_DISABLE_CACHE=1 <原命令>` 复测。若只有后者恢复，优先排查自旋等待和跨 kernel 可见性。若两次都卡住，回到第 2 节，定位最后有进展的 kernel 和等待点。
2. **沿轮询地址找到写方。** 从实际执行的循环或原子操作找到读方等待的锁字、计数器或完成标志，对上写方更新的设备地址及预期值。结合 kernel 的执行轨迹或最小复现核实写方能否运行、是否已经发布。若读方占住同一 stream，写方排在它之后，应按算法调整 launch 顺序或用事件表达依赖。同 wave 内负责释放锁的 lane 无法前进时，按第 2 节处理。
3. **对同步对象做分配实验。** 若写方已经发布、读方仍在同一地址轮询，核对发布/获取操作的原子序和作用域，再从地址和偏移追到实际分配函数或内存池。在默认 cache 下保持 kernel 和输入相同，仅改变承载同步状态的显存分配方式。例如写方发布 `ready`、读方轮询 `ready`，可在原生 HIP 路线核实目标 DTK 支持后，单独用 `hipExtMallocWithFlags(..., hipDeviceMallocUncached)` 分配这个标志。沿用原设备归属和初始化顺序，按对应接口释放；业务数据继续走原分配路径。CUDA 兼容层或框架内存池须核实当前构建路径实际支持的接口，并保留原有异步使用和释放时序。
4. **在默认 cache 下验收。** 重编并核对实际加载的新产物，在未设置 `HSA_DISABLE_CACHE` 的新进程中重复运行原阻塞用例，核对完成状态和原数值断言，再运行受影响的原测试集。记录同步对象、修改的分配位置、复测次数和结果。仍需关闭 cache 才能通过时，结论记为临时规避，默认环境下的故障继续保留。

**Warp 示例。** 一个 float16 atomic 原用例在关闭 cache 后恢复。此时可以记录“阻塞对 cache 设置敏感”。下一步沿实际 kernel 找到等待地址和写方，判断该地址属于独立锁字、完成标志还是被更新的数据本身。确定分配位置后才选择局部修复并按第 4 步验收。

## 4. 数值、低精度和测试覆盖问题

| 问题 | 处理 |
| --- | --- |
| FP8/FP4/BF16类型存在但算子失败 | 逐format×operation×backend确认声明、实例、注册、dispatch和实际设备支持；量化/存储不代证GEMM |
| 量化后数值不符 | oracle使用实际量化输入，核对舍入、scale、累加和布局；保留原容差，不拿量化前数据直接归因kernel |
| 怀疑原测试oracle错误 | 用独立精确参考/对应CPU行为交叉定位，保留原失败与证据；不自动改期望、容差或输出特例 |
| collection error/模块abort | 记录未收集或未执行的节点；不能说没有测试或把缺项算通过 |
| 全GPU skipped或CPU fallback | 不满足GPU验收；查依赖/数据/门禁和实际dispatch，不删除skip装作通过 |
| 整套失败、原样单项通过 | 保留整套统计、精确selector、环境、输入、二进制和两次rc；继续查共享状态/顺序/资源，单项不覆盖整套失败 |
| 缺OpenGL/PNG等可选环境组件 | 按项目配方判断需求；已批准排除注明范围，不能推广为其他任务的自动排除授权 |

## 5. 每个问题的最小闭环

1. 绑定失败命令、首错/trace、源码/生成物、实际加载binary与失败输入。
2. 区分已确认原因与待验证假设；必要时用同条件最小复现和单变量对照。
3. 修改原始代码、构建或版本化转换规则；保留其他平台和原测试合同。
4. 重编、验证新产物/加载路径，重跑原失败用例及受影响范围。
5. 总结“现象、原因、修复文件、复验结果、未解决项”；详细CLI/证据字段按需查[工具接口](../docs/tools.md)。
