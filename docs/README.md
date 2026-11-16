# 维护与导出

执行入口是仓库根 `SKILL.md`。本目录保存公共执行约定、工具接口、输出模板和维护评测资料；`references/` 按问题领域组织。

公共执行文档：

- [contract.md](contract.md)：公共边界和证据规则
- [tools.md](tools.md)：真实 CLI、schema 与限制
- [pre-build-analysis.md](pre-build-analysis.md)：编译前记录模板
- [adaptation-summary.md](adaptation-summary.md)：总结模板

入口源码：[原生 HIP](entrypoints/hcu-native-adaptation/SKILL.md)、[源码转码](entrypoints/hcu-transcode/SKILL.md)、[FastPT 不转码](entrypoints/fastpt-non-transcode/SKILL.md)。

`docs/evals/` 仅供维护者进行独立冷启动评测，包含协议和 JSON 空白输入；不属于应用执行依赖，也不进入四个导出包。

## 四个自包含导出包

| 包名 | 路线 | 专属脚本 |
| --- | --- | --- |
| hcu-porting-workflow | 自动或显式 | 三路线处理器 |
| hcu-native-adaptation | native-hip | native 处理器；不含不转码 prepare/classifier |
| hcu-transcode | transcode | transcode、CMake 转换、hytlass 准备、分类器 |
| fastpt-non-transcode | non-transcode | non-transcode、FastPT 准备、分类器 |

`docs/entrypoints/*/SKILL.md` 是源码定义，不应单独复制安装。导出器将入口、运行脚本和引用闭包放进各自包根；不携带维护者评测、源码测试或维护记录。通用 S 卡片供各路线核查已有源码，因此公共包包含 `code-conversion.md`；不转码激活与 GPUFusion 细节仅进总包/不转码包。

在源码根执行：

```bash
python -B -I scripts/build_skill_packages.py --out /path/to/new-package-directory
python -B -I -m unittest discover -s scripts/tests -v
```

导出目录必须在源码树外且不存在。导出不安装、不更新宿主配置；需要安装时复制完整包，并按宿主规则刷新会话。

导出包推荐 `python -B -I`，运行前校验版本、资源哈希和路线锁，拒绝缺失/篡改/额外字节码缓存。该校验不等于签名认证或构建沙箱，详见[执行约定](contract.md)。

## 修改与验证方法

先锁定当前工作树并保留未提交修改，再改最小规则和工具；新行为用合成工程回归，四包分别在无兄弟目录的含空格路径验证。检查真实 CLI 参数、Markdown 引用与 W/S 卡片定位；避免测试只逐字绑定正文句式。

离线测试只覆盖路由、文件保护、证据处理、工具行为与导出闭包，不证明应用在 HCU 上构建、GPU 数值通过或性能提升。真实效果评估另按 [evals/README.md](evals/README.md)冻结输入/skill并独立运行。历史应用日志只供维护者提炼一般方法，不把组件答案交给被评测执行者。

开发日期、访问状态、个人路径和单次验证结果保存在任务记录中，不写入通用指南。技术来源、版本号、固定提交和适用边界放在对应主题文档，便于核查规则依据。
