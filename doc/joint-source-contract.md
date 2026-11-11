# 联合源合同：v1 基线与 v2 B 层修订

本合同的源码基线是 FAST fork `989872ac2ea3051e21febe1254023efda7246830`。
命名空间为 `fastlio::joint_source`。依据是
[身份与账本](../include/frontend_source_contract.hpp)、
[数值接口](../include/frontend_correlated_update.hpp) 和
[数值实现](../src/frontend_correlated_update.cpp)。

标注适用于其后的整条规则、表格行或接口块：

- **[v1 规范／已实现]**：第 1 部分记录上述提交的实际行为。
- **[v2 规范／待 B1b]**：第 2 部分规定后续实现的要求，当前源码尚未提供。
- **[数据／待 B0b/B2]**：已有清点或成本推导，等待真实祖先、寿命和执行数据回填。
- **[裁定 Q1–Q3]**：独立合同审阅对原设计冲突的修订；只约束 v2 和后续验收，
  不代表源码已实现，也不改变 v1 的默认可接受集合。

v2 采用新增内核的方式；v1 稠密内核继续供 A 层使用，也作为小规模独立参照。
本文没有实现、编译、数学测试或回放验收的含义。

## 1. v1：K1/K2 已合入合同

### 1.1 变量、坐标与知识

| 规则 | 合同 | 状态 |
| --- | --- | --- |
| V1-01 | `Source` 的 `key` 是物理变量的规范身份；`kind`、`model`、`lifetime` 是调用方声明的非空有界字符串，不是内核推断的物理模型。`aliases` 只给同一变量命名。`rawAncestry` 声明原始观测祖先。 | [v1 规范／已实现] |
| V1-02 | `Coordinates` 含有序 `components`、对应的 `units`、`frame`、`tangent`、`epoch`。分量名唯一；所有字符串合法非空。源的坐标与数值块顺序一致。坐标相同不表示 raw 身份相同。 | [v1 规范／已实现] |
| V1-03 | `JointMoments` 唯一保存联合均值和协方差：active 在前，随后为有序源块。记其协方差为 `Σ = [[P, K], [Kᵀ, C]]`；交叉可有正负号，不得忽略。 | [v1 规范／已实现] |
| V1-04 | `MomentKnowledge` 只有 `kUnknown`、`kKnown`，默认未知。已知的零协方差表示确定性支撑；空源列表表示源自由度为零；未知不能补成零。初始化和过程噪声要求 `kKnown` 及合法 `evidence`。 | [v1 规范／已实现] |
| V1-05 | 初始均值必须有限，协方差必须逐元素严格对称。带主元 LDLT 的负主元一律拒绝，不裁剪；重构误差阈值是 `64 × dimension × ε × maxAbs(covariance)`，只作算术误差预算，不作方差膨胀。舍入后的奇异或病态输入可能被拒绝。 | [v1 规范／已实现] |
| V1-06 | 入图是关联／谱系元数据，本身不证明一个新的独立随机源。物理祖先、独立性和幅度的证据由调用方负责；内核只检查所声明模型、身份和矩的合法性。 | [v1 规范／已实现] |

### 1.2 线性因子与身份

**[v1 规范／已实现]** `LinearFactor` 的域必须为 `kLinearFixedModel`，模型为
`y = H x + A ζ + r`。它自带 active/source 坐标快照、完整 `H/A`、观测、余项均值、
逐行单位及方差。余项要求 `kKnown`、`independentRemainder = true`，且每一行的
标量方差严格大于零。缺失、非有限、维数不符或不支持的噪声结构均拒绝。
潜平面、TLS/EIV、选择和随机 H 的转换属于外部适配层。

**[v1 规范／已实现]** 新因子必须按当前全部源的顺序列出 `sourceKeys`、
`sourceCoordinates` 和 A 的列块；零载荷列也保留。别名归一为规范 key。
active 的完整命名映射允许最多 23 维，包括速度、两类 IMU bias 和重力的两个切向分量。

**[v1 规范／已实现]** `FactorIdentity.key` 标识完整似然，和 raw 身份分开。
内核在调用方 `payload` 后附加模型、线性化身份、坐标、源规范 key、行单位及全部数值
`H/A/y/remainderMean/remainderVariance`，以精确字节比较去重。
这是进程本地编码，不是跨平台线格式，也不依赖散列无碰撞假设。
规范化后的整个 payload 仍受 256 KiB 限制。

**[v1 规范／已实现]** `FactorLedger::consume` 在副本上逐个校验，成功才换入：

- 同 key 且完整规范身份相等返回 `kNoOp`；同 key 有任何差异返回 `kIdentityConflict`。
- 不同 key 不能复用同一个 `remainderKey`；重叠 raw 必须在两个因子中都有相同的
  `shared {raw, source}`，否则返回 `kUnsupportedNoiseStructure`。
- 批内后续拒绝不发布前面的消费；满账本不回收，容量拒绝也不发布。

**[v1 规范／已实现]** 数值内核的 `canonicalizeSources` 另要求 `shared` 的 source
解析到携带该 raw 的源，且其 A 列块非零；独立的 `FactorLedger` 没有源字典或 A，
不会代做这项检查。改变几何函数不能隐藏共同噪声；新的独立余项仍须单独论证。
相关余项须显式建成共同源，否则不支持。

### 1.3 活源、退役源与实际检查顺序

**[v1 规范／已实现]** `Snapshot::validateFreshFactors` 对已消费的相同 key 先比较
规范身份；精确重发跳过当前绑定和新鲜祖先检查。因此精确重发可跨 reset、增广或退役
成为 no-op，但仍须通过前置的因子格式和规范化校验。

**[v1 规范／已实现]** 对尚未消费的新因子，`validateAncestry` 的实际规则是：

1. 任一退役源携带该 raw，返回 `kSourceHistoryUnavailable`。
2. 没有活源携带该 raw 时，这一项祖先检查通过；消费账本仍检查跨因子的噪声复用。
3. 有活源携带时，**至少一个**携带源须有规范 `shared` 项并经非零 A 块覆盖。
   源码使用 `std::any_of`，没有要求所有活的携带源都覆盖，也没有角色字段。

**[v1 规范／已实现]** `validateBatch/validateFactor/canonicalizeSources` 在上述检查
之前执行。特别是 `shared` 指向退役源但 A 块为零时，989872ac 会先返回
`kUnsupportedNoiseStructure`。不能把“退役历史不可用”的概括写成所有输入下的状态码
优先级保证。K3 设计提到的 R1 修复不属于本基线，也不在本文声称已实现的范围内。

### 1.4 生命周期、事务与数值运算

| 接口／规则 | 合同 | 状态 |
| --- | --- | --- |
| `initialize` | 接受空源列表但 active 非空；已初始化实例再次初始化返回 `kIdentityConflict`。未初始化的矩／账本视图为空，数值操作不能把未知状态当零先验。 | [v1 规范／已实现] |
| `propose` | 只读同一不可变先验，复制私有 Snapshot，账本筛选后只更新新因子。每个独立标量行通过正交三角化更新联合均值和 root，不求 P 的逆。 | [v1 规范／已实现] |
| `commit` | 比较 proposal 的先验 Snapshot 身份后发布均值、root/协方差、账本和退役表。旧 proposal 或其他实例的 proposal 返回 `kStaleProposal`。一次有变更的提交使同先验 proposal 失效；无变更 proposal 可在同一快照上重复返回 `kNoOp`。 | [v1 规范／已实现] |
| `propagate` | transform 作用于**整个**联合向量。均值为 `T m + offset + noise.mean`；`ProcessNoise` 必须给出完整噪声矩与 old-joint/noise 交叉。交叉严格为零时复用先验 root；非零时重分解完整联合矩，可能拒绝舍入后的奇异输入。 | [v1 规范／已实现] |
| `augment` | 新源为 `generator × oldJoint + offset + noise`，允许一般联合 generator 和相关噪声。须保留完整交叉；不得复活退役 key、alias 或 raw 祖先，也不能给已消费但无活源保留的 raw 换 key 复活。 | [v1 规范／已实现] |
| `reset` | 用已知零过程噪声调用一般联合合同变换并更新坐标描述，不限于 active 的坐标运输。 | [v1 规范／已实现] |
| `marginalize` | 只选取保留变量的精确边缘，不做条件化。非空请求须 `willNeverReuse = true` 和合法 evidence；被删源进入 tombstone，账本保留。空请求为 `kNoOp`。别名解析到同一源的重复请求拒绝。 | [v1 规范／已实现] |
| root 与协方差 | root 是数值支撑；`finish` 从 root 的乘积取同一三角填两侧，保持后续输入所要求的精确对称，不裁剪负主元。 | [v1 规范／已实现] |
| 原子性与所有权 | 所有可达拒绝都不发布状态。类为 thread-confined，调用方串行访问；`moments()/ledger()` 是下次变更前有效的借用视图，proposal 自持不可变快照。 | [v1 规范／已实现] |

**[v1 规范／已实现]** `appliedGain(factors, activeGain)` 是观测式辅助接口，要求整个
批次均为新鲜独立因子；空批、包含已消费因子的批或批内重复不满足该域。
它在私有账本上校验但不消费、不提交，只返回 active 的均值和协方差。
若 `L = [H, A]`，其中心化模型为：

```text
E = Cov(active, L × joint)       S = L Σ Lᵀ + D
mean_G = mean_active + G (y − L mean_joint − remainderMean)
P_G = P − G Eᵀ − E Gᵀ + G S Gᵀ
```

**[v1 规范／已实现]** 实现用 error-root/Joseph 形式计算 P_G，不把最优增益的后验
当成任意实际增益的协方差。`GainResult` 没有提交后的 K/C 或知识类别字段。

### 1.5 固定容量、状态与边界

| 项目 | 限制 | 状态 |
| --- | --- | --- |
| active / 源总 DOF / 批次总行数 | 23 / 32 / 64；因子个数也不超过 64，重复因子同样计入前置行数检查。 | [v1 规范／已实现] |
| 消费账本 / 退役表 | 各最多 128 条，不自动压缩。 | [v1 规范／已实现] |
| 活的规范名和别名总数 | 128。 | [v1 规范／已实现] |
| 每个 source 的 raw/alias 列表、factor 的 raw/shared 列表 | 各最多 128 项；source/factor 的 raw 列表非空。 | [v1 规范／已实现] |
| 身份字符串 / 规范 factor payload | 每串最多 256 字节 / 每因子最多 256 KiB。 | [v1 规范／已实现] |

**[v1 规范／已实现]** `Status` 依次为 `kOk`、`kNoOp`、`kUnknownMoment`、
`kInvalidIdentity`、`kIdentityConflict`、`kDimensionMismatch`、`kNonFinite`、
`kInvalidCovariance`、`kUnsupportedNoiseStructure`、`kUnsupportedModel`、
`kModelCapacityExceeded`、`kSourceHistoryUnavailable`、`kStaleProposal`、`kNumericalFailure`。
容量检查在发布前完成；这些边界不构成实时容量证明。

**[v1 规范／已实现]** [CMake 定义](../CMakeLists.txt) 建立独立的静态数学库，公开
include 路径和 Eigen 依赖，测试链接它；该基线没有安装／导出该库及两份公共头文件的
规则，生产 estimator 也未链接该库。完整生产谱系、重入恢复、一般奇异观测似然、
物理幅度、非高斯尾部和运行性能均不由 v1 合同证明。
现有行为夹具见 [source 测试](../test/test_frontend_source_contract.cpp) 和
[correlated-update 测试](../test/test_frontend_correlated_update.cpp)；本文没有运行它们。

## 2. v2：B 层修订

### 2.1 容量与第一版预算依据（delta §1）

**[v2 规范／待 B1b]** 不调大 `JointSourceFilter` 的固定容量。新类共享身份类型，
采用初始化时固定的 `Budget{sourceDof, liveSources, sourcesPerBatch, rowsPerBatch,
recentLedger, memoryBytes}`；运行中不可放大。任何预算超出须在改状态之前返回
`kModelCapacityExceeded`。`kMaxActiveDof = 23`、旧类的 32/64/128、
`kMaxIdentityBytes` 和 `kMaxFactorPayloadBytes` 不变；带类型整数身份不占字符串预算。
大 s 下 A 层 `23 + s + r_g` 行压缩不能保证压到 64 行，新类必须支持预算给定的更大批。

**[数据／待 B0b/B2]** B0a 的 `counts.json` 和 `results.md` 提供以下**请求级上界**。
计数从首次已记录 Build 到 C5 捕获闭窗结束；A/B 子窗仍是半开区间。

| 项目 | g2-D | g1-D | 状态 |
| --- | ---: | ---: | --- |
| 首次 Build 的 legacy ns | 1793866152399998903 | 1793864808499996900 | [数据／待 B0b/B2] |
| 捕获闭窗（传感器秒） | [1793866207.1, 1793866329.1] | [1793864845.5, 1793864967.5] | [数据／待 B0b/B2] |
| 非空插入请求来源帧上界 | **1714** | **1540** | [数据／待 B0b/B2] |
| 非空 offer 块 / offer 点（非 admission） | 2973 / 2116674 | 2883 / 2651678 | [数据／待 B0b/B2] |
| 激光通道调用 / valid | 1715 / 1714 | 1541 / 1540 | [数据／待 B0b/B2] |
| 帧级 propagation 通道行 | 1714 | 1540 | [数据／待 B0b/B2] |
| 激光 reported iterations | 5625 | 5143 | [数据／待 B0b/B2] |
| 捕获窗内 H 求值块 | 4005 | 4080 | [数据／待 B0b/B2] |
| 重力 valid 通道 / 轮速通道 / 平面通道 | 1118 / 0 / 0 | 1375 / 0 / 0 | [数据／待 B0b/B2] |
| 已记录 Build 请求 | 1 | 1 | [数据／待 B0b/B2] |
| 捕获附近去重累计区间数 | 1180 | 1173 | [数据／待 B0b/B2] |
| 上述区间 noise_steps / external covariance_writes | 25471 / 1027 | 25441 / 1092 | [数据／待 B0b/B2] |

**[数据／待 B0b/B2]** 1714/1540 不是实际同时存活帧源数 m。它们受旧 C5 offer
身份、admission/lifetime 未公开及 mixed Build 元素级 local/prior 身份缺失的限制。
完整 m、`|J|`（每帧实际引用的不同帧源数）和 `r_a`（点的总使用帧数）仍为
`NOT_IDENTIFIABLE`，须等 B0b 的实际祖先和寿命数据。
offer 条件代理的任意迭代 `|J|` 最大值是 354/372，支持复用帧数最大值是 192/321，
它们既不是完整分布，也不是可登记的真实 `r_a` 预算。

**[数据／待 B0b/B2]** propagation 行不是逐 IMU predict；H 求值块和 reported
iterations 不是逐 boxplus 账本。25471/25441 与 1027/1092 只覆盖已记录累计区间，
不是从地图起点的完整 Φ/Q 或按调用点拆分的写者计数。hold、速度清零、pin 尝试、
take-back、guard、外部注入及各 clear 调用的完整次数仍未知；surviving=0、节流日志
或缺通道不能补成零。g2-O 没有插入 trace，不能据相同通道计数补出地图谱系。

**[数据／待 B0b/B2]** cost-model 的推导采用帧源 d=17、s=d m；这是固定外参场景的
6 维更新后 pose 加 11 维更新前 deskew 状态。在线估计外参时须重新评估 23 维帧源。
完整稠密 C 占 `8 s²` 字节，K 占 `8 × 23 × s` 字节，主乘法约为
`2 × 23 × d|J| × s` FLOP/帧；P 更新另有约 `2 × 23 × (23 + d|J|)²` 项。

| 假设 m / \|J\| | s | 稠密 C 投影 | 主乘法投影/帧 | 状态 |
| --- | ---: | ---: | ---: | --- |
| 64 / 30 | 1088 | 9.5 MB | 0.026 G FLOP | [数据／待 B0b/B2] |
| 128 / 50 | 2176 | 38 MB | 0.085 G FLOP | [数据／待 B0b/B2] |
| 600 / 300 | 10200 | 0.83 GB | 2.4 G FLOP | [数据／待 B0b/B2] |
| 2000 / 300 | 34000 | 9.2 GB | 8.0 G FLOP | [数据／待 B0b/B2] |
| 6000 / 300 | 102000 | 83 GB | 24 G FLOP | [数据／待 B0b/B2] |

**[数据／待 B0b/B2]** 将所有请求帧都假设保留为 17 维源时，
g2-D 的 `8 × (17 × 1714)² ≈ 6.79 GB`，g1-D 的
`8 × (17 × 1540)² ≈ 5.48 GB`（十进制）。这两个数字仅说明**不能承诺在线**，
不能作为 `memoryBytes`、实际内存或可接受容量。前向递推的峰值还包括 K、
候选／工作区、账本、身份和灵敏度；对称存储能减半 C 也不等于总内存减半。

| Budget 字段 | 首版记录／后续回填 | 状态 |
| --- | --- | --- |
| `sourceDof` / `liveSources` | 记录 1714/1540 请求上界；实际峰值和两阶段克隆、锚点／外生源额外容量待填。 | [数据／待 B0b/B2] |
| `sourcesPerBatch` / `rowsPerBatch` | 完整 `\|J\|`、实际批行数和压缩行数待填，不用 offer 代理代替。 | [数据／待 B0b/B2] |
| `recentLedger` | 构造参数；可保留近窗与所需重发跨度待填，不宣称 128 足够。 | [数据／待 B0b/B2] |
| `memoryBytes` | 用实际峰值、分配和离线资源回填；本合同不把 C 投影定成预算。 | [数据／待 B0b/B2] |

**[v2 规范／待 B1b]** 用户已定首版只做离线：机上仅有默认关闭的捕获，不接生产，
不回写生产协方差。cost-model 的约 70–80 KB/帧捕获、0.8 MB/s、2.7 GB/h，
以及离线伴随每活点约 144 B 等均为推导，CPU、RSS、吞吐和在线余量均未实测。

### 2.2 状态码与知识类别（delta §2）

| 新增项 | 合同 | 状态 |
| --- | --- | --- |
| `Status::kLedgerCompacted` | 新类 `proposeAppliedGain` 遇到已压缩水位线以下的因子键，无法判别重发；拒绝且不改状态。 | [v2 规范／待 B1b] |
| `Status::kReuseBudgetExceeded` | raw 子单元消费数超过登记总数；拒绝且不改状态。 | [v2 规范／待 B1b] |
| `MomentKnowledge::kUpperBound` | 联合协方差在 Loewner 序下覆盖声明模型的真实联合协方差，并附证明依据；不是“未知”的替代值。 | [v2 规范／待 B1b] |

**[v2 规范／待 B1b]** 上述枚举只在末尾追加，旧值顺序不变。只有新类的实际增益
递推接受并产生 `kUpperBound`。在同一捕获增益和固定线性模型下，联合合同变换加
半正定项保持 Loewner 序；新加噪声须对所比较的系统相同，或另有联合上界依据。
从上界重新计算增益会换掉被评估的估计器，所得后验不能代替原捕获增益的误差矩。
这不否认固定线性似然下最优后验的单调性，也不否认固定该新增益后可为新估计器求界。
`JointSourceFilter::initialize/propose/appliedGain` 必须隔离新类别并返回
`kUnknownMoment`，不能把它投入 v1 的最优更新路径。现有输入通道分别是
`JointMoments.knowledge` 和 `LinearFactor.remainderKnowledge`；没有新增 prior 参数。

**[v2 规范／待 B1b]** 膨胀独立副本的依据为：`ω_i > 0`、`Σω_i ≤ 1` 时，

```text
(Σ_i L_i) C (Σ_i L_i)ᵀ ⪯ Σ_i (1/ω_i) L_i C L_iᵀ
```

**[v2 规范／待 B1b]** 离线在首次使用之前登记总使用帧数 r，**包括点自己作 query
的那一帧**及以后所有支持帧；从首次使用起取 `ω_i = 1/r`，每次使用的副本方差
乘 r。同帧多行／多个因子／迭代中的同一点仍是一份共同副本，须帧内精确联合处理，
不能按行或因子重复消费、独立化或重复膨胀。因子带复用预算条目即输出
`kUpperBound`。一旦某一步使用上界，后续全部读数保持该类别，不能因 reset、
退役或取得某个块而降回 `kKnown`。幅度未知的标定／外生项单列灵敏度，不能借上界
类别把未知幅度补成零，也不能声称整个物理误差已有上界。
按键读出的交叉块 K、C_ab 或 P_ab 继承的是联合界的类别；不能把某个非对角块
单独解释为 Loewner 上界。对具体线性读数须从完整联合界作合同变换。

**[v2 规范／待 B1b]** 证明从原始噪声各次出现到完整 `[e; ζ]` 的线性映射成立，
包括进入克隆源的首次 query 贡献；不能把尚与历史相关的下一次原始噪声直接假定独立。
若先按未膨胀 query 推进、到支持使用才登记 r，上界不成立：单位方差噪声同向使用
两次时真实方差为 4，迟登记只得 `1 + 2 = 3`。离线须用已登记预算重建受影响前缀，
不能只修下一步；缺失此前消费或其上界证明时读数为 `B_UNAVAILABLE(reason)`。

### 2.3 账本、角色与复用预算（delta §3）

**[v2 规范／待 B1b]** 字符串身份继续存在，新增
`RawId{stream: u32, ordinal: u64, sub: u32}`。stream 区分传感器流和模型版本，
ordinal 是单调扫描序号，sub 标识帧内点或整帧单元。调用方声明单调性，内核检查：

| 规则 | 合同 | 状态 |
| --- | --- | --- |
| L1 | 新鲜 raw 的 ordinal 必须严格大于该 stream 已消费的最大值；旧 raw 不能伪装成新鲜。合法复用须走活源路由或登记的复用预算。 | [v2 规范／待 B1b] |
| L2 | 每 stream 水位线加最近 `recentLedger` 条完整条目；近窗内仍逐字节返回 `kNoOp` 或 `kIdentityConflict`。 | [v2 规范／待 B1b] |
| L3 | 因子键落入已压缩区，返回 `kLedgerCompacted`，不能接受为新因子，也不能声称 no-op。拒绝不改变数值、账本、复用计数或水位线。 | [v2 规范／待 B1b] |
| L4 | 非单调身份不压缩，沿用有界字符串账本／tombstone；现有 128 条上限不靠丢记录绕过。 | [v2 规范／待 B1b] |
| 帧源退役 | 用高水位线加活集合；新帧源 ordinal 必须超过高水位线。旧且不在活集合中的源不能再次 `augment`，绑定它返回 `kSourceHistoryUnavailable`。 | [v2 规范／待 B1b] |
| 复用预算 | `RawId`（sub 为点序号）映射到不可改登记总数 r 和已消费整数 c；每使用一帧消费一份，c 超 r 返回 `kReuseBudgetExceeded`。条目随其路由承载帧源退役而删除。 | [v2 规范／待 B1b] |

**[v2 规范／待 B1b]** 压缩仅删除比较 no-op/冲突所需的旧载荷，不删除拒绝旧身份
重入的能力。因此重复不增信息；退役序号也不能重新增广。在线权重
`1/(i(i+1))` 只是有证明的未来变体，不纳入首版离线授权。

**[v2 规范／待 B1b]** 每个 source 对每个 raw 声明角色：

- 路由承载（designated）：因子对该 raw 的一部分依赖经该源列进入。
- 派生依赖（derived）：源值传递依赖该 raw，但该因子的 raw 依赖不经该源列表达。
- 未声明角色默认路由承载，只有显式声明派生依赖才放开其退役限制；内核不能从数值
  自动证明真实载荷为零。

**[v2 规范／待 B1b]** 设计要求 fresh factor 中非新鲜 raw 的**全部**路由承载源
都活着，并都由 `shared` 加非零载荷覆盖；任一个已退役则
`kSourceHistoryUnavailable`。派生依赖的退役不参与此判断。
`shared` 只覆盖经该源路由的部分，不能代表地图点自己的回波噪声也已覆盖；
B 层地图点因子还须带独立的复用预算条目。

**[裁定 Q1]** 按类划清兼容边界：v1 `JointSourceFilter` 保留 1.3 的 `any_of`
和既有退役检查；上述“全部覆盖、未声明默认路由承载”只属于新类。
撤回“新类未声明角色时与 v1 的所有输入逐项等价”的承诺，不能为落实新规则而改旧类。
两个活源都声明 raw q、A 的第二块为零、`shared` 仅指第一源时，v1 仍可接受，
新类默认拒绝 `kUnsupportedNoiseStructure`；新类将第二源显式声明为 derived 后，
在其余输入合法时可接受。新类缺少任一路由承载历史时仍拒绝
`kSourceHistoryUnavailable`。X8 同时锁住这两套默认行为，不把差异解释成回归。

### 2.4 新类接口形状与读数（delta §4）

**[v2 规范／待 B1b]** 以下是接口形状，名称、具体类型、offset/evidence 等完整参数
由 B1b 实现；不是当前可编译 API，也不从省略号推导隐式独立性或默认矩：

```text
class ConsiderSourceFilter {
  initialize(active: JointMoments, budget: Budget) -> Status  // 源为空
  propagate(phi: Matrix, noise: ProcessNoise) -> Status      // 仅 active
  reset(transport: Matrix) -> Status                        // 仅 active
  augment(source: Source, generator: Matrix, ...) -> Status  // 仅克隆 active
  retire(keys, willNeverReuse, evidence) -> Status
  proposeAppliedGain(batch: [LinearFactor], activeGain: Matrix) const -> Candidate
  commit(candidate: Candidate&&) -> Status
  activeCovariance() const -> Matrix                        // 23×23
  cross(key, out) const -> Status                           // 23×d
  sourceCovariance(a, b, out) const -> Status                // d×d
  knowledge() const -> MomentKnowledge
}
```

**[v2 规范／待 B1b]** 新类不提供最优增益 `propose`，也不返回整个联合阵。
Candidate 的新数值只有 P（23×23）和 K（23×s），不拷贝 C。
提交前实例不变，commit 换入；数值与消费、近窗、水位线和复用预算的发布仍须原子。
source 只能由 active 克隆，不能由旧源的一般线性组合生成新源。

**[v2 规范／待 B1b]** 在声明的固定支持、一阶、独立新鲜余项模型内，以捕获的
实际增益 G、`T = I − GH` 递推：

以下简式另要求过程噪声与**完整旧联合向量**不相关。新类 `propagate` 的
`ProcessNoise.priorCross` 必须是显式、维数正确的零矩阵；未知不得补零，其他输入
合法而交叉非零时返回 `kUnsupportedNoiseStructure`，且不发布状态。非零交叉需要
额外的 P/K 项，不在首版简式的域内；v1 的一般相关过程噪声行为不变。

```text
传播：P ← ΦPΦᵀ + Q，K ← ΦK，C 不变
更新：P⁺ = [T, −GA_J] Σ_(e,J) [T, −GA_J]ᵀ + G D Gᵀ
      K⁺ = T K − G A C，C⁺ = C
克隆：ζ_new = S e；C_new,old = S K；C_new,new = S P Sᵀ；K_new = P Sᵀ
reset：P ← L P Lᵀ，K ← L K；源留在各自建立时的切空间
```

**[v2 规范／待 B1b]** 生产轨迹和增益是给定输入，B 层不重新估计状态。
支持侧载荷固定在点变到世界系的 staging 状态，不能改用迟到 Add 调用时的当前状态。
帧源 17 维的 deskew/pose 两段分别取更新前、更新后状态；不能只克隆一次而混同阶段。

**[v2 规范／待 B1b]** Q1 读数以窗口相对增量为目标：`P_delta`、相邻增量的
交叉 `P_ab`、窗口起点联合量 `joint_start`；odom 绝对量仅为带状态的附带读数。
例如 `δ_ab ≈ J_b e_b − J_a ξ_a` 时：

```text
P_delta = J_b P J_bᵀ + J_a C_aa J_aᵀ − J_b K_a J_aᵀ − J_a K_aᵀ J_bᵀ
```

**[v2 规范／待 B1b]** 适配层按键取块组成读数，不把单帧 `FRAME_CONDITIONAL`
升级为跨帧结论。离线伴随的 `B_MODELED_EXACT` 仅指已建模项；前向复用界为
`B_MODELED_UPPER_BOUND`，附膨胀倍数。**[裁定 Q2]** `B_REENTRY_BOUNDED` 保留为
未来状态，首版不产生；Young 数学 oracle 不构成恢复通道。新因子需要已退役的
路由承载源时拒绝，并令该事件之后未闭合的读数为 `B_UNAVAILABLE(reason)`。
谱系断点、未镜像写者或容量拒绝也应为
`B_UNAVAILABLE(reason)`。读数同时带 `AMPLITUDE_ASSUMED`、`S4_UNMODELED`
和各未知幅度源的灵敏度矩阵；这几个标记不能用 `kKnown` 或 `kUpperBound` 消除。

### 2.5 保留和改变的不变式（delta §5）

| 族 | v2 要求 | 状态 |
| --- | --- | --- |
| I02 | 复制不增信息。近窗精确 no-op；压缩区拒绝，不能猜测重复。 | [v2 规范／待 B1b] |
| I03 | 新类扩展带类型身份、角色声明和复用预算；默认 designated 的全覆盖规则不回移至旧类，兼容边界见 Q1。 | [v2 规范／待 B1b；裁定 Q1] |
| I10 | **改变承载方式**：不维护全联合 root。每步通过联合合同变换加半正定项构造 P/K/C；P 更新可用相应主子阵。再对抽样主子阵做带主元 LDLT 审计，负主元不裁剪，返回 `kInvalidCovariance`；不对称也报错。抽样不是完整联合 PSD 的数值证明，不能替代构造推导。 | [v2 规范／待 B1b] |
| I14 | 未知不补零；幅度未知只传播带标签灵敏度，读数逐项列出。 | [v2 规范／待 B1b] |
| I18 | 提交实际增益后的 P 和 K，不能替换成最优后验或把只返回 P 的 v1 helper 当成联合推进。 | [v2 规范／待 B1b] |
| I21 | reset 对源为单位变换，只运输 active 一侧，保留交叉。 | [v2 规范／待 B1b] |
| I22 | 收窄：传播／reset 只作用于 active，generator 仅克隆 active。一般联合变换仍用 v1。 | [v2 规范／待 B1b] |
| I23 | 固定常量变固定 Budget，改状态前检查和原子拒绝不变。 | [v2 规范／待 B1b] |
| I25 | 首版状态递推只启用膨胀独立副本；Young 保留数学 oracle，不启用退役重入／恢复，也不启用成组近似。 | [v2 规范／待 B1b；裁定 Q2] |
| I26 | 仍需 `willNeverReuse` 才退役；新因子需要退役路由承载源时一律拒绝，不允许 Young 绕过。显式派生依赖退役不阻止经其他活的路由承载源使用 raw。近窗精确重发仍按 L2 处理。 | [v2 规范／待 B1b；裁定 Q1/Q2] |

**[裁定 Q2]** 保留 `willNeverReuse` 的硬退役语义，修订 delta I25：首版没有
单次退役重入事件入口。离线已知最后使用时刻时应保留源至最后一次引用；若实际
新事件仍需退役源，原子拒绝并报历史不可用，不能改 key、补独立源、冻结旧读数或
自动重试掩盖缺口。以后若要启用 Young 恢复，须另定保存的边缘矩、角色／消费身份、
一次性许可和完整联合矩更新合同，不能由 B1b 擅自扩展本接口。

**[裁定 Q2]** X5 只核数学命题：保留变量 v、丢失变量 w、已知或有证明上界的
边缘矩 Σ_v/C_w，给定映射 M/N、有限 `β > 0` 和独立新鲜噪声协方差 R 时，

```text
Cov(Mv + Nw + η) ⪯ (1+β) M Σ_v Mᵀ + (1+1/β) N C_w Nᵀ + R
```

此式界住完整保留联合向量；只膨胀 P 而保留原 K/C 不构成该证明。
X5 的数学成立与 X7 的 API 拒绝须同时验证；不得据 X5 PASS 宣称有重入能力。

**[v2 规范／待 B1b]** I01、I04–I07、I15–I17、I24 所对应的最优更新／条件化
运算在新类中不提供，按 delta 列为不适用；这不撤销 v1 的既有测试或新类事务要求。

**[v2 规范／待 B1b]** 生产谱系捕获还必须承载以下 B0a 更正；实现与数据由 B0b/B2
提供，本片没有加入任何捕获代码：

- 每个写 x 的事件须从 `map_generation` 起点镜像，或把未知外生项送入灵敏度通道；
  未处理事件之后的联合矩不可用。仅写 P 不直接改变真实误差，但 P/H/R 和
  `update_simple` 结果必须在对应阶段捕获，以闭合实际增益。
- [laserMapping.cpp](../src/laserMapping.cpp) 的 2684–2689 是
  `FASTLIO_TEST_HOOKS` 下的 pos/rot/vel 和 `change_x` 覆盖，不能列为只写 P；
  2691–2696 的 covariance 分支才是独立的 P-only 夹具。
- `clearPinned` 加 `AddPinnedPoints`（6698/6703）是先验替换谱系断点，保留 live 层
  不等于旧先验身份连续。`map_generation`／先验版本必须表示它以及 Build、clearLive
  的参考变化，不能只看 coordinate_epoch 或 map_reset 计数。
- [iVox::Build](../include/ivox/ivox.hpp) 内部的 `clear` 是同一次逻辑重建的子操作，
  不再单独计一代；谱系事件须按逻辑动作记账。
- Q4 采用 H1：`esekf::predict` 内默认关闭的累积
  `Φ_acc ← F_x1 Φ_acc`、`Q_acc ← F_x1 Q_acc F_x1ᵀ + Q_step`，加取值并清零访问器。
  必须覆盖空洞、dt_start 和尾段 predict，并证明关闭时生产输出逐位不变。
  这是后续捕获实现要求，本片未改 `esekfom.hpp`。

### 2.6 18 族解析 oracle 的扩展 X1–X12（delta §6）

**[v2 规范／待 B1b]** 以下是新增验收要求，不是当前测试 PASS。先做 X1、X8；
有理数参照优先，浮点容差必须给出来源，不能用实现自身的结果生成 oracle。

| 编号 | 要验证的行为与独立参照 | 状态 |
| --- | --- | --- |
| X1 | 源 DOF≤32，在两类共同支持的域内，同一传播／增广／实际增益／reset 序列的 P/K/C 与稠密内核一致；用 v1 `propagate` 表达 `[[I−GH, −GA], [0, I]]` 和完整过程矩。另测新类拒绝非零 process priorCross 且状态不变，v1 相关过程噪声能力保持。补齐 K3 §9 的既有覆盖缺口。 | [v2 规范／待 B1b] |
| X2 | 两帧插点／引用的标量联合矩与有理数手算逐项一致。 | [v2 规范／待 B1b] |
| X3 | 对明确的共模联合矩 P=K=C=1、H=1、A=−1、D=1，分别给定 G=0、1/2、1，得到 P⁺=1、5/4、2，K⁺=C⁺=1；各自的共模贡献均为 1。另测 H=−A 但非共模矩时 E 可非零。 | [v2 规范／待 B1b；裁定 Q3] |
| X4 | 显式保留每个复用回波噪声的稠密精确参照；从首次 query 起按登记 r 膨胀，比较完整 P/K/C 联合差及读数差半正定，等权且各次灵敏度同向相同时取等号。包含首次 query→克隆→后续 support、同帧共用及迟登记 3<4 的反例。 | [v2 规范／待 B1b] |
| X5 | 纯数学 oracle：按 2.5 的完整联合 Young 式与给定 β 手算逐项一致，差矩阵半正定；含只抬 P 失败的反例。与 X7 一起确认数学界不打开 API 重入通道，不产生 B_REENTRY_BOUNDED。 | [v2 规范／待 B1b；裁定 Q2] |
| X6 | 近窗重发 no-op；压缩区为 `kLedgerCompacted`；拒绝后数值和完整账本逐位不变。 | [v2 规范／待 B1b] |
| X7 | 退役帧 ordinal 不能再增广；新因子绑定退役源为 `kSourceHistoryUnavailable`，即便已知边缘矩可作 Young 界也不接受；拒绝后状态、预算和水位线不变。近窗精确重发由 X6 覆盖。 | [v2 规范／待 B1b；裁定 Q2] |
| X8 | 新类退役 derived 加活 designated 的 raw 可用，结果等于从未建立该 derived 的对照；退役源改为 designated 则拒绝。多活源未声明角色且只覆盖一个时：v1 接受，新类拒绝；新类显式把未覆盖源声明为 derived 后接受，全部 designated 均覆盖时接受。 | [v2 规范／待 B1b；裁定 Q1] |
| X9 | 超登记次数为 `kReuseBudgetExceeded`，不改变状态；同帧只消费一份。带预算条目的结果类别为 `kUpperBound`，后续传播、reset、退役和按键读块均不得降回 kKnown。 | [v2 规范／待 B1b] |
| X10 | 其他输入合法时，将 initialize 的 prior.knowledge 或 propose/appliedGain 的 factor.remainderKnowledge 设为 `kUpperBound`，三入口均返回 `kUnknownMoment` 且不发布状态。 | [v2 规范／待 B1b] |
| X11 | 近奇异序列的主子阵 LDLT 非负；人为注入不对称时报错，不能靠裁剪通过。 | [v2 规范／待 B1b] |
| X12 | 已知幅度常量源的灵敏度及其协方差贡献与普通显式源的稠密参照一致。 | [v2 规范／待 B1b] |

**[裁定 Q3]** 修正 delta X3 及 validation-plan OB2 的原判据：必须先声明共同支撑，
例如同坐标中 `e = ζ`（P=K=C），再用 H=−A 才得到 `E=0` 和 `S=D`。
实际增益下 `P_G = P + G D Gᵀ`，共模贡献不变，新增余项贡献仍在。
P 不变的充要条件为 `G D Gᵀ=0`；本合同 D 正定时等价于 G=0，不能通过把 D 设零
避开反例。v1 `I05PerfectCorrelationCancellationAndDoubling` 使用最优 `propose`，
该夹具的增益为零，仍保持既有断言。B1b 必须使用给定 G，不得为了过 X3 偷换增益。

### 2.7 对 K3 adapter-contract §6 的依赖影响（delta §7）

| 依赖 | 修订与兼容边界 | 状态 |
| --- | --- | --- |
| 四个容量常量 | 23/32/64/128 不变；新类预算不改 A 层限制。 | [v2 规范／待 B1b] |
| `Status` | 末尾追加两个值；穷尽 switch 的编译影响由 B1b 同片处理。 | [v2 规范／待 B1b] |
| `MomentKnowledge` | 末尾追加 `kUpperBound`；A 层只传 `kKnown`，旧类隔离新类别。 | [v2 规范／待 B1b] |
| `ModelDomain` / `Coordinates` | 不变。 | [v2 规范／待 B1b] |
| `Source` | 末尾追加带默认值的类型序号和逐 raw 角色；新类未声明默认 designated。旧类默认输入的接受／拒绝行为保持，A 层源列表为空。不能把新类全覆盖检查回移到旧类。 | [v2 规范／待 B1b；裁定 Q1] |
| `JointMoments` | 结构不变。 | [v2 规范／待 B1b] |
| `SharedRawSource` / `FactorIdentity` | 前者不变；后者末尾追加带默认值的 typed raw 和复用预算条目。保留按成员赋值和原顺序聚合初始化的可用性；缺字段初始化被视为错误时须在受影响调用处补齐。 | [v2 规范／待 B1b] |
| `LinearFactor` / `GainResult` | 不变。LinearFactor 已有 `remainderKnowledge`，GainResult 不新增知识类别或联合矩字段。 | [v2 规范／待 B1b] |
| `initialize/moments/propose/appliedGain` | 保持签名及原有合法输入行为；空源初始化、propose 只读、appliedGain 返回 active 均值／协方差、单因子身份限制和发布前检查五条继续有效。 | [v2 规范／待 B1b] |

**[v2 规范／待 B1b]** v1 `propose(factors)`、`appliedGain(factors,G)` 没有 prior
参数，X10 不得臆造一个。实例先验类别由 initialize 隔离，因子余项类别由
`remainderKnowledge` 隔离；两者均须继续使用“非 kKnown 即拒绝”的规则。
这证明新类别不会进入旧内核，不代表这两个只读入口能单独接收上界联合先验。

**[v2 规范／待 B1b]** B1b 在 K3-1 交付被接收后开始，检查追加枚举的 switch 和
聚合初始化影响；不读取或假定本基线之外正在开发的 K3-1 内容。B1b 须运行 X1–X12、
不改断言地通过既有 K1/K2 测试、执行 arm64 数学测试，并验证已集成 K3-1 的受影响测试。
X1/X8 的跨类比较只在共同支持的域内要求数值相等；Q1 明列的接受集合差异另锁断言。
本片仅交合同文本和审阅裁定；B0b 捕获和 B2 真实 trace 执行按各自依赖推进。

## 输入来源

下表是外部交付的封存身份，路径为任务工作区标识，不是仓库相对链接。

| 来源 | 固定路径与 pin |
| --- | --- |
| K1/K2 字典起点 | `/var/tmp/lio-workspaces/frontend-joint-source-k1k2-20261110-d427/output/source-dictionary.md`；SHA256 `fe485fb23bc9c3e3d60a2896426fa7c3a909dddf44843a740067944f7588d359`。源码不符处以第 1 部分为准。 |
| B 层设计 | `/var/tmp/lio-workspaces/fe-lidar-k2-contract-b-layer-design-20261110-d427/output/` 的 `k2-contract-delta.md`、`design.md`、`cost-model.md`、`validation-plan.md`、`slicing.md`；SUMS SHA256 `bbcdc3917d7b0d358e2e720caa07250c47885507c14c23c63bf831c8c68b4fc8`。 |
| B0a 清点 | `/var/tmp/lio-workspaces/fe-lidar-b0a-event-inventory-20261111-d427/output/` 的 `results.md`、`counts.json`；SUMS SHA256 `daaaf3c33848257fb6218d65a7fdea39a03c72fdf5ba5efde47798c87ec6e2d2`；counts SHA256 `91c5652aa9036ae8b2be2b8e752f758bdf1437db7ca201273a78d1cc444d85f9`。 |
| K3 依赖合同 | `/var/tmp/lio-workspaces/fe-lidar-k3-adapter-design-20261110-d427/output/adapter-contract.md` §6、§9；SUMS SHA256 `3f9fcec787dd7e61677f52af4fc588cae123d66d742fdf793aad5df61b79030e`。 |
| 已定口径与分片 | `/var/tmp/lio-workspaces/refactor-line-d427-20261102/decisions/` 下 `user-decision-b-layer-q1-q2q3-q4-q7-1ae2-20261111T0112Z.md`（SHA256 `d20ac966c327bead9907779b5c1df42e85974edc01c5871604ca341358d94eb1`）及 `user-decision-b1a-now-b1b-after-k3-1-1ae2-20261111T0245Z.md`（SHA256 `3080114bba449d558c24ff12c349b9aa6cd3c67e0c1c8a945a6e1b1a3eea3098`）。 |
