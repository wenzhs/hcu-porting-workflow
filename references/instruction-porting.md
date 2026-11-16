# 指令与 warp 适配

扫描 W-xx 是待核对位置，不是缺陷结论。先查实际编译命令和目标 DTK，再判断宏分支、指令和调用方。工具见[接口](../docs/tools.md)，运行异常见[排障](common-errors-and-solutions.md)。

## ROCm 分支、PTX 与 builtin

| 已有实现 | 先检查 | 处理 |
| --- | --- | --- |
| ROCm/HIP 分支 | gfx/ROCM_VERSION 限制、实际宏值、当前 DTK 声明与所用接口 | 不因命中 ROCm 分支便认定可用；有真实差异再加最小 HCU 特化 |
| 内嵌 PTX/asm | 输入输出约束、寄存器宽度、lane 布局、内存序、收敛与返回值 | 优先查等价 HIP API/当前 builtin；保留原平台实现，逐指令验证 |
| `__builtin_amdgcn_*` / `__builtin_hcu_*` | 编译器定义、重载和 feature gate、gfx lowering | 当前原接口不可用时才调查 HCU 对应接口；禁止只替换前缀或发明函数名 |
| Tensor Core / MMA（DTK26.04+ 调查重点） | 对应发行版矩阵指令/builtin、输入累加类型、shape、fragment、lane 分工 | 声明→宏选支→实例编译→ISA/lowering→GPU数值与同步；不能由版本号宣称所有 CUDA MMA/WGMMA 可用 |

用目标安装中的头文件/编译器源码查完整符号，再做最小实例。`__has_builtin`（若可用）只能筛编译器声明，不证明目标 feature、参数组合或数值语义。`mma/ldmatrix/cp.async/mbarrier` 要联同流水线和同步检查，不独立改名。

## 分清 W、G、L

| 量 | 含义 | wave32→wave64 时 |
| --- | --- | --- |
| W | 设备实际物理 wave 宽度 | 由设备、编译目标、执行模式确认；本页64案例仅用于已测wave64 |
| G | 一个算法任务的协作组宽度 | 可等价改成W，也可保留原G分phase执行；独立任务不能直接合并 |
| L | 序列化字段、公共layout或固定格式宽度 | 保持原合同，不能随W扩大 |

HCU 优先评估 `tile=warpSize`：一个独立任务由完整 wave 承接，联动 caller grid、lane映射、mask、shuffle、shared/temp storage 和尾部覆盖。不能等价承接时，保留原G并用phase/串行回退。该偏好不证明性能提升，不全仓32→64；`warpSize`能否作模板常量需当前编译器验证。

### 案例：整 wave 广播误用 width32

前提是已确认 W=64，一个请求由完整64个lane协作，全部成员到达调用点；仅物理lane0产生结果。将旧调用机械套入默认width32的wrapper，会让上半组读取物理lane32：

```cpp
int value = (lane == 0) ? 7 : -1;
int wrong = __shfl(value, 0, 32); // lane32..63读物理lane32，得到-1
int right = __shfl(value, 0, 64); // 同一任务的64个lane均读物理lane0，得到7
```

原算法的同步和内存序仍单独保留。检查lane31/32/63，并通过原入口覆盖DIM31/32/33/63/64/65及尾部请求；只测DIM32可能没有验证高半组输出。若实际是两个独立tile32任务，则上例的width64不适用。

### 案例：隔离 tile 后漏改 host grid

把一个原tile隔离到一个wave后，每任务实际占用W个线程；host仍按`ceil(n*G/block)`发射会改变分批。例：W=64、G=4、n=4、block=128，旧公式只发1个block、2个wave，后两个key进入后续stride；新映射需要核对`ceil(n*W/block)`，此例为2个block。淘汰算法可能在resident key更新前处理新key，必须同时验证原更新/淘汰结果。

这不提供全局线程执行顺序保证。核对device索引/stride、host grid、乘法溢出、n=0和尾块；正确性依赖全局先后时还需明确同步/阶段协议。数值错误消失与性能提升分别验收。

## 接口差异与实例

| CUDA/原代码 | HIP/HCU 核查 | 必要修改/验证 |
| --- | --- | --- |
| `__shfl_sync(mask,v,src,width)` | 当前DTK头/编译器与编译开关是否启用_sync变体；原mask和逻辑分组 | 可适配为`__shfl(v,src,width)`，去掉mask形参并显式保留分组宽度；确认全部所需lane实际参与，见下例 |
| `__shfl_down_sync` / `__shfl_down` | width与参与组、目标lane越界返回规则 | 归约offset按实际G；所有约定成员均满足调用前提 |
| `__ballot_sync` / `__ballot` | 返回位宽、同步参与集合和位编号 | wave64链路用足够宽的类型；不把无mask版本当同步等价物 |
| `__activemask` | 调用瞬间活跃集合与分歧 | 不把瞬时mask当算法固定成员或同步证明 |
| `__syncwarp` / block barrier | 版本声明、参与者、执行同步与内存可见性 | 同wave不等于不需内存序；不能无条件宏屏蔽 |

<a id="shuffle-sync-fallback"></a>

### `__shfl_sync`可以适配为`__shfl`

`_sync`接口是否可用受工具链版本、头文件及编译开关影响。适配当前DTK时，若该变体没有提供或支持开关未启用，可以采用已有的非`_sync`接口，不必为了保留原拼写强行启用另一版本的选项。

```cpp
// CUDA原调用
auto out = __shfl_sync(member_mask, value, src_lane, group_width);
```

```cpp
// HIP/HCU适配：去掉mask参数，保留value、src_lane和显式width
auto out = __shfl(value, src_lane, group_width);
```

| 原调用条件 | 转换时怎么处理 |
| --- | --- |
| 一个完整、收敛的逻辑组参与，源lane有效 | 可以按上述形式替换，检查组内源lane编号和边界返回值 |
| 原代码省略width | 先查原逻辑组宽度，再显式填写。原CUDA默认32不能无意识变成目标wave64默认宽度 |
| mask选择部分/稀疏lane，或调用处存在分歧 | `__shfl`没有mask参数，不会自动重现原成员约束；先组织合法参与集合/数据处理，或保留已验证且支持的_sync实现 |
| 原算法还依赖共享内存可见性或其他同步 | 分别保留所需同步/内存序；shuffle接口替换不代替这些操作 |

同族`__shfl_down_sync`/`__shfl_up_sync`/`__shfl_xor_sync`也可按当前DTK接口调查相应非_sync形式，分别核对delta/laneMask、width、参与者及越界语义，不做全文件无条件字符串替换。

编译开关要查现场定义：例如[HIP 6.4.3接口文档](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.3/how-to/hip_cpp_language_extensions.html)记录过`HIP_ENABLE_WARP_SYNC_BUILTINS`启用方式，而[ROCm 7.0.2变更记录](https://rocm.docs.amd.com/en/docs-7.0.2/release/changelog.html)记录默认启用及`HIP_DISABLE_WARP_SYNC_BUILTINS`关闭方式。它们是上游版本例子，不能据此认定当前DTK采用相同开关/默认值。

例如**同一个求和任务由整wave承接**，且所有lane都有效参与：

| G | shuffle down offsets | 归约步数 |
| --- | --- | --- |
| 32 | 16,8,4,2,1 | 5 |
| 64 | 32,16,8,4,2,1 | 6 |

尾部lane须按原运算使用正确单位元，保留原浮点容差；树形顺序变化需数值验证。若原wave64中是两个独立tile32任务，直接增加offset=32会串组，须重新分配每个任务或保留phase方案。

```cpp
// 位操作示意：已确认物理lane在[0,63]；不构造 1ULL << 64。
unsigned long long lane_bit = 1ULL << physical_lane;
// ballot结果沿参数、存储、返回值都保持64位；原格式L不因此改变。
```

### 案例：改了 mask 变量，漏改移位和消费者

```cpp
// 原32槽代码；slot=40时，1U的移位已超出32位左操作数范围。
unsigned int retry_mask = 0;
retry_mask |= 1U << slot;
int next = __ffs(retry_mask) - 1;
```

```cpp
// 已确认64槽且slot在[0,63]的目标分支。
unsigned long long retry_mask = 0;
retry_mask |= 1ULL << slot;
int next = __ffsll(retry_mask) - 1;
```

验证仅bit40置位时返回40，并覆盖空mask（ffs返回0，减一为-1，不能用作索引）、bit31/32/63、多个置位和清位循环。使用平台条件类型`LaneMask`时，常量应为`LaneMask{1}`，ffs/popcount也匹配位宽；仅扩大变量无法修复原`1U`表达式。参数、返回值和持久存储沿链检查，公共数据格式L不随内部mask扩大。

与[HIP上游语言说明](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)对照接口语义时，仍以目标DTK版本头文件和实测为准；上游新增API不能证明DTK已同步。

## W 卡片

<a id="w-01"></a>

### W-01 宽度来源

核实设备W、host分派及模板/constexpr要求；常量折叠不等于常量表达式。未知目标不默认32或64。

<a id="w-02"></a>

### W-02 lane 与组编号

2D/3D先算 `x + Dx*(y + Dy*z)`，再按W/G求lane和任务编号；重算grid与输出覆盖，验证lane31/32/63、尾部及多任务。

<a id="w-03"></a>

### W-03 物理 mask 位宽

检查ballot到参数/返回值的完整链；wave64避免32位截断及 `1u<<lane`，匹配64位popcount/ffs接口并确认空mask约定。

<a id="w-04"></a>

### W-04 逻辑 tile mask

原G回退时区分物理lane位段和组内位，上半tile不能沿用低半mask；检查对齐、合法移位、收敛点及每phase不同谓词。

<a id="w-05"></a>

### W-05 shuffle

当前DTK编译链未提供或未启用_sync变体时，可将`__shfl_sync(mask,value,src,width)`适配为`__shfl(value,src,width)`。显式保留src/width，并检查去掉mask形参后的实际参与集合；完整wave重映射与原tile回退分别测。具体转换见[接口替换示例](#shuffle-sync-fallback)。

<a id="w-06"></a>

### W-06 vote/reduce/scan

记录归约/扫描算子、单位元、inclusive/exclusive和返回lane范围；G变化联动offset/步数，保持原精度与负向测试。

<a id="w-07"></a>

### W-07 部分参与与 activemask

活动mask只是采样；检查全部指定成员满足调用前提，分歧、稀疏/空参与和尾部不能由满wave探针代证。

<a id="w-08"></a>

### W-08 执行同步与内存可见性

分清成员收敛、shared/global可见性、原子序/作用域和进度。不因同wave删同步；以当前头/编译器/运行证据决定等价实现。

<a id="w-09"></a>

### W-09 block barrier 与提前返回

追踪kernel所有返回路径和barrier参与者；必要时保留无效lane参与同步、仅屏蔽数据操作，不让部分线程跳过所需barrier。

<a id="w-10"></a>

### W-10 派生存储

检查每wave/group的shared/temp数量、索引、寿命、对齐及组间复用；G/grid变化不能只改shuffle。

<a id="w-11"></a>

### W-11 CUB/hipCUB

按版本核查Warp/Block原语模板、逻辑组大小、TempStorage及调用合同；可编译不保证跨组存储安全。

<a id="w-12"></a>

### W-12 序列化与 ABI

格式L、外部布局/位段、shape和输入副作用保持原样；内部wave宽度通过显式映射衔接，不扩大公开结构。

<a id="w-13"></a>

### W-13 cooperative groups

查partition支持尺寸、参与集合与group.sync；先评估整wave等价方案，否则原G按phase处理；锁释放者与caller geometry须有进度保证。
