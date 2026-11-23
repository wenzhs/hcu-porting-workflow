# 编译构建适配参考：CMake、Make、Torch Extension

本页回答“怎样配置、编译、链接、打包当前工程”。源码header/宏/API映射见[代码转换](code-conversion.md)，PTX/builtin/warp见[指令处理](instruction-porting.md)。FastPT激活由不转码路线的专属指南说明，原生路线无需读取。执行前从项目文档确认实际构建入口，不要求项目改成本文示例。

## 1. 先确定构建路线和输入

| 路线 | 编译输入 | 构建系统应做什么 | 不应做什么 |
| --- | --- | --- | --- |
| 原生HIP/ROCm/HCU | 原仓已有HIP实现/其框架原生生成源 | 沿用HIP开关、target和框架HIPIFY，修当前DTK差异 | 再叠加外部hipify、因CUDAExtension名字改判CUDA-only |
| 源码转码（CUDA 输入的默认路线） | 映射后的 HIP 源 | 优先复用已有 HIPIFY 生成步骤；否则接入已核实的转换器，更新源路径、语言、库和 flags | 重复转换，或把原 CUDA 源和生成源同时编入同一 target |
| FastPT不转码 | 原CUDA源 | 保留CUDA-facing入口和GPUFusion工具链 | 设置LANGUAGE HIP、混入HIP兼容头或执行HIPIFY |

记录源码根、生成根、构建根、编译器、标准、gfx和实际configure/build/package命令。Conda不是前提，沿用容器/解释器的已有初始化；同一target不混用三条链。

## 2. ROCm与DTK路径、runtime、包的差异

表中是调查位置和常见名称，最终值来自目标安装，不能凭目录名编造HCU package/target。

| 项目 | ROCm常见形式 | DTK/HCU处理 | 核对结果 |
| --- | --- | --- | --- |
| 安装根 | `/opt/rocm`、ROCM_PATH/ROCM_HOME | 可能为`/opt/dtk`或版本目录，沿当前shell/发行版配置 | 工具realpath、实际include/lib根 |
| 编译驱动 | hipcc包装器、底层clang++ | 可能有发行版驱动/包装器，不能将hipcc/hycc/clang++任意互换 | 编译器版本、子工具、真实argv |
| runtime | HIP runtime开发包、`libamdhip64.so`等 | 包名/布局随DTK版本，SONAME也可能保留兼容名字 | 安装清单、链接owner、DT_NEEDED/加载路径 |
| CMake包 | `hip-config.cmake`、`hip-lang-config.cmake`及库config | 查现场实际config与导出target；不臆造`hcu::...` | `CMAKE_PREFIX_PATH`、所用config文件和target |
| 架构 | 完整gfx、如`--offload-arch=gfx...` | 使用当前驱动/项目真正支持的入口 | 生成argv、实际代码对象架构 |
| 框架 | ROCm torch常以`torch.version.hip`标记 | HCU发行版还可能有`torch.version.hcu` | 字段、安装源码分支与生成编译命令一起核验 |

不要把CUDA-compatible环境的CUDA_HOME/include/lib注入原生HIP验证。`find_package(... QUIET)`可能关闭GPU target，configure成功后还要查实际目标和源码消费。

## 3. CMake：原生HIP构建

1. 先沿仓库已有USE_HIP/BUILD_HIP等实际开关，确认该target的HIP源码和库。
2. 设置当前安装的依赖prefix和完整gfx；标准来自项目/框架头实际要求，分别作用于C++/HIP target，不统一加C++20。
3. 核实当前CMake/DTK组合支持的HIP language/compiler；有些CMake实现不接受hipcc包装器作为CMAKE_HIP_COMPILER。
4. configure后检查真实compile/link命令，再build目标。下面是原生HIP最小结构示意，变量值须来自已确认配置：

```cmake
cmake_minimum_required(VERSION 3.21)
project(hcu_example LANGUAGES CXX HIP)
add_library(op SHARED src/op.hip)
set_target_properties(op PROPERTIES
  HIP_ARCHITECTURES "${TARGET_GFX}"
  HIP_STANDARD "${PROJECT_GPU_CXX_STANDARD}")
target_link_libraries(op PRIVATE ${VERIFIED_HIP_LIB_TARGETS})
```

```text
cmake -S <source> -B <build> -DTARGET_GFX=<实际gfx> -DPROJECT_GPU_CXX_STANDARD=<项目要求> <实际依赖选项>
cmake --build <build> --target op -j <并行数>
```

不是所有工程都需要新增HIP language；采用框架Extension或现有编译helper时沿其设计。`HYTLASS_HIPCC_ARCHS=936`等专用数字参数不能填进标准gfx变量。

## 4. CMake：主动转码的两种接入方式

### 4.1 选择转换器

| 工具 | 常见适用面 | 必须核实 |
| --- | --- | --- |
| hipify-torch / Torch HIPIFY | 框架源码和CUDA扩展 | 当前入口/API、HIPIFY代际、映射文件、extra_files/headers覆盖 |
| hipify-perl | 文本/API转换 | 真实脚本/help、输入输出约定、宏/模板/PTX残留 |
| FastPT转换接口 | 当前版本明确提供的源码转换 | 初始化、版本匹配、转换操作与所需功能；不把CUDA-facing编译当转换 |

指定转换器不可用时保留阻塞，不擅自改provider。命令按现场help/API确定，不给所有转换器套相同参数。

PyTorch扩展还要记录构建实际Python、转换模块路径/代际、入口参数与额外映射；外部hipify-torch、Torch自带HIPIFY及项目vendor副本可能不同。`is_pytorch_extension=True`不选择v2，某些外部CLI的版本检测失败会回退v1；按[版本核对和生成代码案例](pytorch-hipify.md)验证，不能只升级Torch便认定外部转换器同步更新。

### 4.2 内嵌转换与外部转换对照

| 方式 | 执行顺序 | CMake关注点 |
| --- | --- | --- |
| CMake构建图内嵌 | 输入变化→转换命令→生成HIP→compile/link | 明确OUTPUT/DEPENDS、生成目录与完整argv；不能仅在configure阶段生成而漏增量依赖 |
| 外部转换 | 固定输入→独立运行转换→检查映射/diff→配置生成后的工程 | 生成文件已存在；源列表消费新路径，原CUDA输入不重复编译 |

构建图内嵌示意：`VERIFIED_HIPIFY_ARGV`是按本机转换器验证过的CMake参数列表，包含输入/输出参数，不是任意shell字符串。

```cmake
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/op.hip"
  COMMAND ${VERIFIED_HIPIFY_ARGV}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/op.cu"
  VERBATIM)
target_sources(op PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/op.hip")
```

外部转换的target前后对照：

| 原CUDA构建 | 转换后处理 |
| --- | --- |
| CUDA language、`src/op.cu` | HIP language、真实`generated/op.hip`；若保留`.cu`，对该文件显式LANGUAGE HIP |
| `CUDA::cudart`、cuBLAS/cuDNN等 | 按[库映射](code-conversion.md)找到现场HIP runtime/数学库config和实际导出target |
| `-gencode/-arch/--gpu-architecture` | 删除不适用的NVCC专用项，按当前HIP驱动传完整gfx；不删除其他平台分支 |
| CUDA include、生成头路径 | 更新为目标解析链和真实生成根，不全局添加影子头 |
| CUDA标准属性/分离编译 | 核查相应HIP标准与链接需要，不机械映射所有CUDA属性 |

库的API/类型语义在[代码转换](code-conversion.md)，此处只处理源文件和链接配置。

### 4.3 现有CMake准备工具的范围

`scripts/cmake_fastpt_convert.py`由总包/转码包提供，不是CUDA源码转换器。

| 接口/限制 | 实际含义 |
| --- | --- |
| `--source-map` | 值须是当前输入工作区内已经存在的文件；不同于原始根→生成根的计划映射 |
| `.cu→.cu` | 合法identity映射，但会对消费方设置LANGUAGE HIP |
| 支持的改写 | 字面量project/enable_language、source list、已知库和有限flags；保留package条件及子目录作用域 |
| 复杂动态构造 | 宏/函数/循环/生成器表达式、未知库/flags或消费者作用域不明时阻塞，改为明确人工补丁 |
| `--write` | 只准备，不构建；Git在适配分支原地处理，无Git才建新副本；不覆盖未知旧目录 |
| `--emit-build-script` | 显式请求才生成helper，执行前仍核实环境和命令；生成不等于运行 |

转换预计写集可用`inspect_build_context.py --transform-map`审计，生成后用`--outputs-exist`核对；两侧映射不是转换器副作用沙箱。真实参数见[工具接口](../docs/tools.md)。

### 4.4 CUTLASS/hytlass依赖如何接入

| 情况 | 构建处理 |
| --- | --- |
| 头文件消费 | 固定本地hytlass根、release/完整commit，仅向实际消费target增加正确include；不因此运行库根CMake |
| 已有hytlass库target | 复用工程已声明的真实target，不猜名字，不额外FetchContent/add_subdirectory |
| 使用准备器 | `cutlass_to_hytlass_prepare.py`只在总包/转码包提供；先扫原文WGMMA/TMA/SM风险，`headers`与`existing-target`各有前提 |
| 参数含义 | 该CLI的`--target`是CMake目标名，区别于preflight的调查子目录；不自动下载、不改gitlink、不指向main |
| 输出状态 | `prepared-pending-semantic-review`仅表示有限准备，仍需接口/布局/同步修复、实际构建和原GPU正确性 |

对应库支持的版本与代码语义见[代码转换](code-conversion.md)，不能靠Sm→Gfx改名完成适配。

## 5. Make：编译和链接都要迁移

| 位置 | 检查/修改 |
| --- | --- |
| CC/CXX/HIPCC/NVCC | 使用选定路线的实际驱动；不要只替换一个nvcc字符串 |
| CPPFLAGS/CXXFLAGS | include、标准、宏和host/device flags分开；移除本路线不适用的NVCC项 |
| LDFLAGS/LDLIBS | runtime/数学库/RPATH/ABI及静态动态链接合同 |
| 源列表与依赖 | 生成HIP的先后关系、对象名碰撞、header变化是否触发重编 |

已验证驱动支持`--offload-arch`时的HIP示意，HIPCC/ARCH由当前任务传入：

```makefile
op.o: generated/op.hip
	$(HIPCC) --offload-arch=$(ARCH) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
app: main.o op.o
	$(HIPCC) --offload-arch=$(ARCH) $^ $(LDFLAGS) $(LDLIBS) -o $@
```

递归Make还要追子Makefile、生成规则和最终链接，不用单个对象成功替代完整目标。

## 6. Torch Extension：CppExtension/CUDAExtension与HCU字段

先读取实际解释器/安装元数据和`torch/utils/cpp_extension.py`，必要时在授权目标环境显式import确认：

```python
import torch
print(torch.__version__, torch.__file__)
print({k: getattr(torch.version, k, None) for k in ("hcu", "hip", "cuda")})
hip_candidate = bool(getattr(torch.version, "hcu", None)
                     or getattr(torch.version, "hip", None))
```

`hip_candidate`仅是候选，不应直接写成支持通过。`CUDAExtension`可能在HIP/HCU框架内进入HIPIFY/hipcc，不能因其名字选择外部转码；反之某工程只判断`torch.version.hip`也可能漏掉仅有hcu字段的发行版，应修最小检测并验证实际分支。

| 环节 | 必查 |
| --- | --- |
| 路线门禁 | IS_HIP_EXTENSION/ROCM_HOME、HIP/HCU字段和项目target；FORCE_CUDA等环境变量不能伪造硬件支持 |
| 转换集合 | sources、include_dirs、extra_files和生成名分别核实；可读include目录不自动被HIPIFY |
| 工具链 | 实际编译器、标准、ABI flags、宏和完整gfx；不要按Torch版本号猜C++标准 |
| 原生HIP | 沿框架原有HIPIFY，不能额外混入CUDA兼容工具链 |
| 不转码 | 确认CUDA-facing分支且不执行HIPIFY，模式操作见专属指南 |
| 打包/安装 | 沿原setup/PEP517配方；`--no-build-isolation`控制构建隔离，`--no-deps`控制依赖安装，二者不等价 |

## 7. 构建后必须检查

默认 `all` 只覆盖当前配置启用的目标。修改可选子目录或 `foreach` 生成的目标后，核对该配置的 CMake File API codemodel、编译数据库和实际 build 日志；根目录的 `BUILDSYSTEM_TARGETS` 不递归代表整个工程，源文件 `LANGUAGE` 属性也有目录作用域。一个源可属于多个 target，头文件通过实际依赖记录关联使用它的编译单元。改动未进入配置、缺少依赖证据或没有运行结果分别记录，不能从默认测试通过推导可选目标通过。已有记录可用[改动覆盖检查](../docs/tools.md#changed-build-coverage)整理；工具不执行 configure/build，也不证明模板分支被实例化。

1. 保存configure/compile/link/package真实命令、cwd、环境、子进程rc和完整日志。
2. 确认预期库/可执行文件/wheel为本次新产物，检查架构、非空和身份。
3. 核对实际安装与加载路径、依赖owner，排除旧`.so`和错误解释器。
4. 进入原GPU/数值和约定测试；编译、加载、运行、正确性分别记录。失败处理统一见[常见问题与解决](common-errors-and-solutions.md)。
