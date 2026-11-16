# 相关资源路径

用于查询DTK接口、寻找HCU已适配依赖和下载匹配组件。**先查目标安装中的代码和二进制，再查对应版本资料**，入口可访问不等于API/包已支持。

## 1. 目标DTK安装目录

| 要找的内容 | 查证方法 | 留下的结果 |
| --- | --- | --- |
| 实际工具链与根路径 | 当前shell的`command -v hipcc`、编译器版本、项目真实argv及路径解析 | DTK/编译器版本、实际绝对路径；不强制改成某个`/opt`目录 |
| header/API/builtin | 在已确认的DTK include、编译器定义和示例范围内用`rg -n '<符号>' <目录>` | 声明、类型、条件宏、feature gate与版本；builtin还需lowering和目标实例验证 |
| runtime/数学库与CMake包 | 查实际安装清单、config、导出target、动态库符号及`DT_NEEDED` | 包名、target、SONAME、owner及实际加载路径 |
| 框架HIP/HCU链 | 已安装框架的`version.py`、`cpp_extension.py`、HIPIFY实现与头文件 | `torch.version.hip/hcu`、分支和实际消费链，不由版本号推实现 |

## 2. 文档、代码与下载入口

| 类别 | 入口 | 使用方式 |
| --- | --- | --- |
| 光源文档总入口 | https://developer.sourcefind.cn/document | 找当前DTK/库版本的编程、移植与API资料 |
| DTK相关文档/下载 | https://download.sourcefind.cn:65024/1/main/ | 固定对应发行版手册/组件，不从最新目录倒推现场版本 |
| HYGON-AI代码仓 | https://github.com/HYGON-AI | 查同名HCU适配仓、tag/commit、构建方法和支持范围 |
| DAS代码仓 | https://developer.sourcefind.cn/codes/ | 查社区适配源码、维护线和依赖版本 |
| 工具和库下载 | https://developer.sourcefind.cn/tool | 查匹配DTK/系统/架构的工具和库 |
| DAS PyPI | https://pypi.sourcefind.cn/ | 查匹配Python/框架/DTK的wheel及安装说明；不擅自升级共享依赖 |

## 3. 复用已适配组件时的记录

| 必查项 | 处理 |
| --- | --- |
| 用户固定版本/功能 | 先判断适配仓是否覆盖；不能因同名便换成main |
| 源码来源 | 仓库URL、tag/完整commit、patch、实际include/链接路径 |
| 二进制来源 | wheel/库版本、平台标签、解释器/框架ABI、DTK依赖和实际加载身份 |
| 支持证据 | 对应API、dtype/layout、算子和测试；README支持架构不代替当前实例验证 |
| 找不到资料或入口不可访问 | 明确待核实项，继续查现场源码和最小证据，不编造接口/下载地址/支持结论 |

维护者应记录材料来源与版本边界；历史组件答案不作为新任务已验证结果。
