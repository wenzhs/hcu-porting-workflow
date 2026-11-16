---
name: hcu-porting-workflow
description: 将 CUDA、HIP 或 ROCm 应用与组件移植到 HCU/DCU（DTK）并完成验证，覆盖分支确认、可行性分析、代码与依赖分析、编译构建和原测试。支持原生 HIP、CUDA 源码转码和用户指定的 FastPT 不转码路线；也可只分析可行性，不负责 FastPT 自身跨版本升级。
---

# HCU 软件移植工作流：分析、编译构建与测试

按照下面五步执行到用户要求的阶段。用户只要求分析时保持只读；要求适配编译时，完成代码、构建与测试，不停在预检计划。`scripts/`是辅助工具，不能代替项目实际构建命令。

遇到逻辑组宽、mask、锁或多分支 dispatch 时，按参考文档核对 W/G/L、真实 grid/stride、owner/版本重载和完整 API→kernel→同步路径；具体案例不替代项目证据。

## 一、适配要求

| 要求 | 实施方式 |
| --- | --- |
| 最小改动 | 优先复用已有HIP/ROCm/HCU实现，只修目标功能所需差异；HCU专有实现用项目已有平台宏或经核实的新条件隔离 |
| 保留源码与行为 | 保留用户修改、非HCU分支、公共API/ABI、layout、输入副作用、原断言和容差；修复写入原始源码/构建/版本化转换规则 |
| 使用实际环境 | 沿用已初始化的DTK、解释器和依赖；容器即可，Conda不是前提；不凭空切工具链版本或修改启动文件 |
| 保持功能范围 | 必需依赖不能直接屏蔽。可选功能需说明影响并取得排除依据；编译成功、GPU运行、数值正确分别验收 |
| 按证据选路 | unknown不等于不可用；指定路线失败先定位，不静默换路、不用全SKIP或CPU fallback完成GPU需求 |

## 二、已知的重要问题

| 问题 | 处理方向 |
| --- | --- |
| gfx936 / gfx938 FP8 | gfx936不按原生FP8计算目标规划，gfx938有FP8硬件支持；具体DTK、backend、format和算子仍逐项核实，类型/量化可用不证明GEMM可用 |
| CUTLASS | hytlass/已适配实现只支持特定版本与接口子集；先固定版本、架构和实例，不能机械改名后宣称支持 |
| PyTorch接口与HIPIFY代际 | 核对实际转换器与目标Torch；v2保留框架CUDA类名不等于停止runtime或header转换，见[版本差异与案例](references/pytorch-hipify.md) |
| warp同步与逻辑tile | 不依赖NVIDIA ITS语义；核实物理wave后选择整wave等价映射或原tile的phase/串行回退，联动mask、存储与grid |

## 三、适配流程

### 步骤1：明确目标代码仓、分支和版本

1. 核对源码实际路径、仓库来源、branch、HEAD/tag、dirty修改，以及要支持的功能/架构。
2. 用户指定ref时核实该ref；未指定时用**当前HEAD和工作树**作基线，新建适配分支，不自行切main/latest或历史成功版本。恢复任务沿用已确认的适配分支。
3. 保留用户改动；无Git时建立全新、不重叠的可见副本，报告编译实际使用的路径。

### 步骤2：代码分析，并形成编译前分析文档

| 分析项 | 要找到的具体内容 | 参考文档 |
| --- | --- | --- |
| 构建入口 | README/构建文档、CMake/Make/setup/pyproject/项目脚本，真正使用的target、源码列表、编译和安装命令 | [编译构建适配](references/build-adaptation.md) |
| 已有平台路径 | 当前target是否已有HIP/ROCm/HCU实现；核对框架`torch.version.hip/hcu`、真实`cpp_extension.py`和原有HIPIFY链 | [编译构建适配](references/build-adaptation.md) |
| GPU依赖 | CUDA库、CUTLASS/TensorRT、源码依赖、所需API/版本；查HCU已适配代码或wheel，说明替代/保留/阻塞/批准排除 | [代码转换](references/code-conversion.md)、[资源路径](references/resources.md) |
| 代码差异 | header、USE_ROCM/CUDA_VERSION/CUDA_ARCH、PTX/builtin、warp/mask、同步、精度/布局 | [代码转换](references/code-conversion.md)、[指令处理](references/instruction-porting.md) |
| 测试方法 | 原仓/官方测试命令、输入、设备条件、GPU覆盖、原断言、完整约定范围 | [常见问题与解决](references/common-errors-and-solutions.md) |

选路：**显式选择优先 → 复用当前target可用原生路径 → 只有CUDA路径时优先转码**。原生能力未知先核实；用户明确保留CUDA源码、指定FastPT/GPUFusion不转码时走不转码。混合工程逐target判断，项目原有HIPIFY不重复执行。

构建前在实际编译仓根写`ADAPTATION_ANALYSIS.md`，保存基线、上述结论、路线、依赖方案、准备/转换/configure/build/安装/测试的实际命令和未决项。用[分析模板](docs/pre-build-analysis.md)，现有完整记录可以复用。纯分析请求直接返回这些内容即可。

按需使用调查工具：

```bash
python -B -I "<skill-root>/scripts/source_risk_scan.py" --repo "/path/to/project" --summary
python -B -I "<skill-root>/scripts/route_preflight.py" --repo "/path/to/project" --route auto
```

预检仅输出计划，不执行转换/构建，`planned`不是完成。

### 步骤3：代码编译适配与构建

**按3.1→3.6实际推进；只执行选定路线和当前工程的构建方式。** 开始前应从步骤2得到源码根、生成根（如需要）、构建目录、目标gfx、依赖根和原仓命令；以下示例中的变量/项目选项必须由这些事实填入。

#### 3.1 准备构建环境与依赖

- 进入步骤1确认的适配源码目录；检查实际编译器、DTK、解释器/框架和目标gfx。
- 按项目文档和当前DTK版本初始化一次所需环境。原生HIP使用HIP环境；只有明确的不转码路线才进入相应CUDA-facing环境。
- 准备步骤2选定的HCU依赖和实际include/lib/CMake路径；固定版本，不让安装依赖解析替换已固定的框架包。
- 关键工具链或转换配置改变时采用新的任务构建目录；合法同轮增量构建保留，不能误用旧产物。

#### 3.2 按路线处理源码与构建配置

| 路线 | 构建前必须完成的动作 | 完成后编译什么 |
| --- | --- | --- |
| 原生HIP | 沿用原仓HIP开关/框架HIPIFY；修DTK路径、runtime包/导出target、架构和必要宏分支，不再额外转换 | 原仓原生target及其原有生成源 |
| 源码转码 | 选并核实hipify-torch/hipify-perl/FastPT转换接口；执行有界转换并检查映射/diff；修改CMake/Make/Extension的源路径、语言、库映射和flags | 已确认的生成HIP源；剔除同一target重复的原CUDA源和不适用NVCC参数 |
| FastPT不转码 | 核实FastPT/Torch/DTK/GPUFusion匹配和隔离安装；按现场入口完成编译模式激活，确认CUDA-facing扩展链不进入HIPIFY | 原CUDA源和该路线实际编译链，不改成HIP语言 |

源码/构建做法见[编译构建适配](references/build-adaptation.md)，语义修改见[代码转换](references/code-conversion.md)；FastPT激活与模式要求见[不转码适配](references/fastpt-non-transcode.md)。prepare/helper只准备，不能据此跳过下面真实构建。

#### 3.3 配置并编译目标

优先执行仓库文档已经确认的命令。下面给出执行顺序，不能仅运行预检或configure就记录“编译通过”：

| 工程入口 | 配置/准备 | 实际编译 |
| --- | --- | --- |
| CMake | `cmake -S "$SOURCE_ROOT" -B "$BUILD_DIR" <项目确认的路线/依赖/架构选项>` | `cmake --build "$BUILD_DIR" --target <实际目标> -j <并行数>`；验证全项目时再构建约定的其余target |
| Make | 按原Makefile设置已核实的CC/CXX/HIPCC/NVCC、include/lib和架构参数 | `make -C "$SOURCE_ROOT" <原仓目标与已确认变量> -j <并行数>` |
| Torch扩展 | 先确认解释器、框架实际路径和HIP/CUDA-facing分支；按项目配方准备构建依赖 | 沿原`setup.py`/PEP517/项目脚本构建；例如支持wheel构建的项目可用`python -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$WHEEL_DIR"` |
| 项目专用脚本 | 阅读脚本及其配置，核对它最终消费的源码/工具链 | 执行原脚本，记录底层真实configure/compile/link命令，不能套用另一工程的入口 |

CMake原生HIP、嵌入/外部转换、Make和Extension的具体代码例子集中在[编译构建适配](references/build-adaptation.md)。不预设所有工程支持同名HIP开关或需要统一C++标准。

#### 3.4 打包、安装并核对产物

- 按仓库配方生成库/可执行文件/wheel。安装或package overlay只在任务授权的环境内执行，记录实际安装位置。
- 检查本次预期产物存在、非空、架构及身份；核对链接依赖和运行时实际加载路径。
- 防止旧`.so`、另一解释器安装、错误RPATH/库搜索路径掩盖本次构建；import通过后仍进入GPU测试。

#### 3.5 编译失败时定位、修复、重编译

| 首个稳定错误 | 首先核对 | 修复后动作 |
| --- | --- | --- |
| configure/package缺失 | 实际config文件、包/target名称、依赖根和条件开关 | 修依赖/构建配置，重新configure再build |
| header/类型/宏/指令错误 | 实际编译单元、argv/include链、host/device宏、接口声明及转换结果 | 最小补丁进入原始源码/转换规则，重新生成并重编该target |
| 链接/undefined symbol | caller、符号owner、源码是否入库、真实链接库/类型ABI | 补正确source/export/link闭包，重链接并验证加载 |
| 构建中断/OOM/超时 | 实际进程与日志进度、资源和并行度 | 明确进程结果后调整已证明的资源问题，不盲重放或把中断算代码错误 |

先复编失败target验证首错消失，再继续约定构建范围。保留失败与修复后日志；不能为脚本返回0删除目标、关闭必需功能或把CUDA路径偷偷换成CPU。详见[常见适配和测试问题](references/common-errors-and-solutions.md)。

#### 3.6 进入测试的条件

必须有本次构建/打包真实结果，以及对应产物身份和加载路径。没有GPU运行条件时说明阻塞和未运行范围，不能把“成功生成文件”当整个适配完成。

### 步骤4：执行原测试并实现修复

1. 按仓库已有或对应版本官方文档的方法，先做加载/最小GPU正确性，再完成约定测试集合。
2. 记录主项、subtest、collection error、PASS/FAIL/SKIP、崩溃、超时、未运行及实际GPU命中，保留原输入、oracle和容差。
3. 异常先查[常见问题与解决](references/common-errors-and-solutions.md)；无匹配证据时定位实际dispatch/kernel/输入，形成最小复现。
4. 修复落在代码/构建或转换规则，重构建新产物、重跑失败用例和受影响原测试；整套失败与单项复测结果分别保留，不用单项通过覆盖整套失败。

### 步骤5：整理适配结果

实际编译仓根保存简洁的`ADAPTATION_SUMMARY.md`：仓库/路径/分支/HEAD、系统/DTK/编译器/gfx、改动、真实构建和测试命令/结果、问题原因与解决、未验证/遗留范围。采用[总结模板](docs/adaptation-summary.md)，不放临时处理文件链接。

需要范围账本/最终回执时，先完成仓内文件，再冻结源码，在树外形成最终证据；其余细则见[执行约定](docs/contract.md)。

## 四、按问题读取参考文档

| 文档 | 范围 |
| --- | --- |
| [常见适配、测试问题以及解决方法](references/common-errors-and-solutions.md) | 编译/加载/运行/测试故障，锁、跨kernel barrier、HSA诊断和tile进度 |
| [编译构建适配参考](references/build-adaptation.md) | CMake/Make/Extension，原生/转码接入，ROCm与DTK差异，构建代码案例 |
| [代码转换实现与注意事项](references/code-conversion.md) | header、条件宏、库/API/类型/句柄映射，依赖支持边界 |
| [指令处理与warp适配](references/instruction-porting.md) | ROCm分支限制、PTX/builtin、矩阵指令、shuffle/ballot/reduction |
| [FastPT不转码适配](references/fastpt-non-transcode.md) | 匹配环境、激活模式、CUDA-facing编译、GPUFusion限制与验证 |
| [相关资源路径](references/resources.md) | 目标DTK本地查证、光源文档、代码仓、工具和wheel下载入口 |

工具真实CLI按需查[脚本接口](docs/tools.md)，不要求每轮运行全部工具。维护者导出/评测说明在`docs/`，不作为应用编译前必读材料。
