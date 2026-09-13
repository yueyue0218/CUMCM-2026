## 摘要

### 关键词

## 一、问题重述

复杂环境下无人移动终端（机器狗）对多频段无线电干扰源的定位与清除，本质上是一类结合测向几何推理、时空路径规划与多源状态决策的动态搜索优化问题。干扰源分布于指定目标圆域内，机器狗需要在遵循移动、频道切换、信号检测与清除等动作耗时约束的前提下，利用携带的测向设备采集示向度数据，并在存在有界测向误差的情形下实现对干扰源定位区域的精确构建与几何特性分析。

对于基础定位与覆盖判定，核心在于将测向楔形与目标圆域、接收上界求交形成连续凸定位区域，数值计算再采用其保守凸多边形外包。基于凸几何性质，计算外包的直径由至少一对最远顶点取得，然而其直径圆并不必然能够完全覆盖整个定位区域。因此，需要引入严格的几何判定准则，并在 Jung 定理给出的理论边界指导下，结合确定性增量算法精确求解最小包围圆，从而在给定清除半径的限制下评估单点清除的可行性。

在多源搜索与动态清除任务中，随着干扰源数量、占用频道及辐射特性信息的未知性增加，单个频道的几何定位需进一步融合物理圆域裁剪、信号接收范围以及否定性观测信息，实现对可行区域的动态更新与预测交会。针对全向源与定向源等不同物理场景，系统须在极其有限的实际运行与虚拟时间预算内，协调移动路径与观测策略，不仅要实时维护各频道的定位状态与排除表，更需在最坏情形或期望效益准则下平衡测向收益与动作成本，最终构建起一套兼顾鲁棒性与高效性的控制策略，确保以最短的总虚拟时间完成对目标区域内全部干扰源的可靠定位与清除。

## 二、问题分析

### 2.1 问题一

问题一只给出测向误差的确定性硬界，因而采用集合定位而非概率估计：每次观测形成一个可行测向楔形，多次求交得到干扰源定位区域。随后以凸集直径度量定位不确定性，并以最小包围圆和 Jung 定理判断直径信息能否支持圆覆盖，从而形成“硬界测向—集合交会—直径计算—最小包围圆判定”的几何主线。

### 2.2 问题二

针对首次示向检测后干扰源定位区域的不确定性问题，深入分析第二检测点的选择策略具有重要的工程应用价值与理论意义。在实际搜寻过程中，单次示向观测仅能将干扰源限制在一个自检测点向外辐射的狭长楔形区域内，无法对干扰源的真实距离进行有效约束。若机器狗盲目沿首次示向轴方向直线推进，两次观测的交角极小，极易导致交叉定位失效，使更新后的定位区域几何直径依然过大；反之，若机器狗仅沿着示向轴的垂直方向侧向移动，虽然能拉开观测基线、增大交叉角，却可能在移动过程中偏离测向楔形远端，甚至超出干扰源的有效信号接收半径，导致第二次检测面临无信号的风险。因此，第二检测点的选择本质上是在“移动时间成本”、“信号接收可靠性”与“交叉定位几何精度”三者之间寻优的贝叶斯实验设计问题。

为科学求解这一多目标决策问题，建模思路遵循“联合后验更新 — 时间与接收双重可达域 — 期望与最坏性能双轨优化”的逻辑主线。首先，将干扰源的空间位置 $G$ 与该干扰源固有的接收半径 $R$ 构建为联合隐状态 $(G, R)$，引入联合先验分布；在首次检测获得示向角后，基于贝叶斯定理推导联合后验分布，准确刻画位置支持集以及远端位置（$1000\,\mathrm{m} < d < 1500\,\mathrm{m}$）因接收概率衰减所形成的非均匀权重。同时，模型引入“精确可行集—闭合凸外包”双层几何表示，精确可行集用于贝叶斯更新和面积计算，而闭合凸外包用于直径、最小覆盖圆半径的保守认证。

在此基础上，结合机器狗的最大移动速度与本次任务的时限要求，构建局部位移坐标系下的时间可达域；同时根据联合后验计算候选点的信号接收概率，将接收概率达到置信水平的可达区域映射为第二检测点的候选决策空间。进一步地，考虑第二次检测可能出现的近距离响应（$\mathrm{near}$）、示向角响应（$\mathrm{direction}$）以及无信号响应（$\mathrm{no\_signal}$），计算第二次检测后定位区域的期望几何直径与期望面积；通过引入无量纲化的综合损失函数，将期望定位直径、期望面积与移动时间成本进行加权融合。

考虑到测向误差或接收距离可能存在失配，模型进一步通过 Minkowski 膨胀与边界扩大构造鲁棒安全可行域与最坏性能上界约束，形成极小极大（Minimax）或带最坏约束的贝叶斯优化选择准则。最后，采用成对蒙特卡洛仿真方法，将该最优策略与传统的“等距离垂直布点策略”在平均定位直径、尾部风险及信号接收率等指标上进行成对对比分析，从而为后续多频段、多目标的动态搜寻与清除任务提供坚实的单目标决策基础。

### 2.3 问题三

问题三的核心在于将静态几何定位与主动选点能力升维为全局视角下的多目标在线决策系统。从实际应用价值看，该问题真实模拟了复杂电磁环境下无人巡检设备对未知干扰源的自主处置过程，要求系统在干扰源总数未知、信号频段独立且有效接收半径存在差异的约束下，仅凭局部测向与交互反馈实现高效处置。从建模挑战看，该问打破了单目标定位的理想假设，引入了非完全信息博弈与多任务调度的复杂性。移动耗时、测向耗时、切频开销以及清除失败惩罚构成了非线性的虚拟时间成本，这使得决策不仅要追求几何定位精度，更要在信息获取探索与目标清除利用之间建立动态平衡。此外，如何在缺乏总源数先验的前提下给出具有数学保障的“全域无遗漏”终止证明，是该问最关键的理论突破点。

解决问题三需要构建一个融合了空间覆盖、状态估计、频道调度与路径规划的在线决策框架。基于全向源最小有效接收半径与目标圆域的几何约束，可设计具有完备覆盖保障的骨干扫描点阵，通过逐频段扫描建立全无信号即无源的确定性逻辑排除链，解决未知目标的完备搜索与无源频道的终止判定问题。在状态估计上，系统需要为每个频道建立动态状态机，利用概率与稳健几何的双层表达维护可行源集合闭合凸包，并融合有信号与无信号的贝叶斯观测实时更新定位评估接口。在选点与调度层面，可结合前两问的主动选点策略，在单频道内依据预期可行域缩减率或清除半径收敛速度选择最低成本检测点，并在多频道场景下将就绪的清除任务与待补测任务统一纳入调度池。通过开放式旅行商问题规划极小化移动与切频开销的路径，配合示向收缩几何逼近等后备机制，确保定位误差在有限次移动内强行压缩至清除半径内。最终在已知目标全被清除且其余频道均取得有效无信号证明时触发停止条件，输出总虚拟时间与加密日志。



根据 Q3 完备性证明、算法收敛性分析及 PPO 模型参数的最新更新，对“模型假设”与“符号说明”进行了全量集成与统一校对。

统一修订要点如下：

1. **几何与距离符号统一**：将目标点 $g$ 的径向距离统一为 $r_g = \Vert{}g\Vert{}$，避免与接收半径 $R$ 或包围圆半径混淆。
2. **补充 Q3 状态与多频道证明符号**：补齐实测证据变量 $\nu_{jk,t}$、极值验证与解的有效性指示变量 $\chi_t$、四类频道状态集合 $(\mathcal P_t, \mathcal C_t, \mathcal E_t, \mathcal U_t)$、凸外包覆盖半径 $r_*(K)$、执行误差上界 $\eta_c$ 及细分深度 $d(B)$。
3. **补充 PPO 与世界模型参数**：增加标准化时间尺度 $T_s$、自由选择步与有效价值步样本集 $(\mathcal I_{\mathrm{choice}}, \mathcal I_V)$ 以及策略更新标准化优势值 $\widetilde A_t$。
4. **假设体系完备化**：新增单频道单源硬假设、几何认证容差界、训练环境名义先验及固定动作配额接管策略。

## 三、模型假设

1. **环境与运动平坦性假设**：假设目标圆域及机器狗的运行轨迹均位于理想的二维欧几里得平面内，忽略地形起伏、障碍物阻挡以及三维空间高差对机器狗移动距离、移动速度及无线电波传播轨迹的影响。
2. **传感器误差有界性假设**：问题一只采用题目给定的测向误差硬界，即误差严格位于 $[-\delta,\delta]$（$\delta=1^\circ$）内，不依赖均匀分布、独立同分布等具体概率假设。
3. **信号传播与辐射稳定性假设**：假设干扰源在工作期间的发射功率、中心频率及辐射方向图（全向或特定定向特性）保持时间恒定，信号在平面介质中沿直线传播，不考虑多径效应、非线性衰减或突发性停播对示向度采集的干扰。
4. **动作耗时与状态切换确定性假设**：假设机器狗的移动速度、频道切换耗时、信号检测耗时以及执行清除动作的耗时均为确定性常数，且在任务执行过程中设备性能稳定，不发生随机故障或机械迟滞。
5. **清除机制的确定性与局域有效性假设**：假设机器狗发起的干扰清除动作在其指定的清除半径内具有绝对有效性，即位于清除圆域内的干扰源将被100%销毁并停止发射信号，而清除圆域外的干扰源完全不受影响。
6. **信号有效接收半径的先验分布与不变性假设**：假设全向干扰源的信号有效接收半径 $R$ 在 $[1000, 1500]\,\mathrm{m}$ 范围内服从均匀分布，且在同一干扰源的搜寻与定位过程中，其接收半径保持恒定不变，后续检测继承首次观测后的联合后验分布，不得重新独立抽取 $R$。
7. **干扰源位置的先验分布假设**：假设干扰源在目标大圆域 $\Omega = \{g \in \mathbb{R}^2 \mid \|g\| \le 1800\,\mathrm{m}\}$ 内按面积均匀分布，且在未发生任何观测前，干扰源位置 $G$ 与其接收半径 $R$ 相互独立。
8. **问题二的附加概率假设与重复观测假设**：仅在问题二的名义贝叶斯模型中，额外假设不同检测点的测向误差在 $[-\delta, \delta]$（$\delta = 1^\circ$）内独立同分布且服从均匀分布；该假设不是问题一几何结论的前提。同一地点的环境误差固定，重复观测不产生独立平均收益。

## 四、符号说明

本文所使用的符号及说明如表1所示。

**表1 符号及含义说明**

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| **几何与空间参数** |  |  |
| $\mathcal C_0$ / $R_{\text{target}}$ | 半径为 $R_{\text{target}}=1800\,\text{m}$ 的目标大圆域 / 其半径 | 区域 / $\text{m}$ |
| $R_{\text{clear}}$ | 机器狗干扰信号清除的有效半径 | $\text{m}$ |
| $g = (x, y)$ | 目标点二维空间坐标 | $\text{m}$ |
| $r_g$ | 目标点 $g$ 的径向距离，即 $r_g = \Vert{}g\Vert{}$ | $\text{m}$ |
| $S_i$ / $s_k$ | 问题一第 $i$ 个检测点坐标 / 第 $k$ 个骨干观测点坐标 | $\text{m}$ |
| $\theta_i$ | 机器狗在检测点 $S_i$ 处测得的干扰源示向度角 | $\text{rad}$ 或 $^\circ$ |
| $\delta$ | 设备最大测向角度误差硬界（问题一取 $\delta=1^\circ$） | $\text{rad}$ 或 $^\circ$ |
| $\mathcal{\Omega}$ | 由目标圆域、接收上界圆与示向度约束交会形成的连续凸定位区域 | 无（几何区域） |
| $D(\mathcal{\Omega})$ | 定位区域 $\mathcal{\Omega}$ 中任意两点间距离的最大值 | $\text{m}$ |
| $r^*(\mathcal{\Omega})$ / $r^*(K)$ | 定位区域/凸外包 $K$ 的最小包围圆半径 | $\text{m}$ |
| $c^*$ | 定位区域 $\mathcal{\Omega}$ 的最小包围圆圆心坐标 | $\text{m}$ |
| $\ell_{j,t}(g)$ | 频道 $j$ 在位置 $g$ 处的历史方向观测及下界消元函数 | — |
| $\eta_c$ | 输出位置相对计算点的执行误差上界 | $\text{m}$ |
| $\hbar(B), d(B)$ | 几何单元 $B$ 的空间尺寸与分割细分深度 | $\text{m}$, 无 |
| **耗时与物理参数** |  |  |
| $v$ | 机器狗的移动速度 | $\text{m/s}$ |
| $t_{\text{switch}}$ | 机器狗进行频道切换所需的时间 | $\text{s}$ |
| $t_{\text{detect}}$ | 单次无线电信号检测与示向度采集所需的固定耗时 | $\text{s}$ |
| $t_{\text{clear}}$ | 执行一次干扰清除动作所需的固定耗时 | $\text{s}$ |
| $T_{\text{virtual}}$ | 系统完成全部任务所消耗的总虚拟时间 | $\text{s}$ |
| $T_s$ | PPO/世界模型中的标准化时间尺度（默认 $T_s = 100$） | $\text{s}$ |
| **状态、事件与集合变量** |  |  |
| $f_k$ / $j$ | 频道编号 | 无 |
| $S_k$ | 频道 $f_k$ 当前对应的干扰源空间定位候选集/区域 | 无（几何区域） |
| $\mathcal{E}_k$ | 针对频道 $f_k$ 已排除的不存在干扰源的区域集合 | 无（几何区域） |
| $\nu_{jk,t}$ | 观测点 $s_k$ 对频道 $j$ 在时刻 $t$ 的实测证据（$\bot$:未测, $0$:无信号, $1$:有信号） | 无 |
| $\chi_t$ | 相关记录核实及硬约束无冲突的有效性指示变量 ($\chi_t \in \{0, 1\}$) | 无 |
| $\mathcal P_t$ | 时刻 $t$ 已确认有源的频道集合 | 无 |
| $\mathcal C_t$ | 时刻 $t$ 已成功清除的频道集合 | 无 |
| $\mathcal E_t$ | 时刻 $t$ 已证明无源的频道集合 | 无 |
| $\mathcal U_t$ | 时刻 $t$ 仍待判明存在性的频道集合 | 无 |
| $u_i$ | 机器狗在第 $i$ 步采取的决策动作（移动、切换频道、检测或清除） | 无 |
| $G = (X,Y)$ | 干扰源的真实空间位置坐标 | $\text{m}$ |
| $R$ | 干扰源的有效信号接收半径 ($1000 \sim 1500\,\text{m}$) | $\text{m}$ |
| $S$ / $q$ | 首次检测点坐标 / 第二检测点候选坐标 | $\text{m}$ |
| $H_1$ / $H_2$ | 首次观测历史信息 / 第二次观测后的累积历史信息 | — |
| $\pi_0(g,r)$ / $\pi_1(g,r \mid H_1)$ | 干扰源位置与接收半径的联合先验密度 / 联合后验密度 | $\text{m}^{-3}$ |
| $p_1(g \mid H_1)$ | 首次观测后干扰源位置的边缘后验概率密度 | $\text{m}^{-2}$ |
| $\mathcal{F}_1$ / $K_1$ | 首次检测后的精确位置支持集 / 闭合凸外包区域 | — |
| $W(S,\theta,\delta)$ | 由检测点 $S$、示向角 $\theta$ 与误差 $\delta$ 决定的测向楔形区域 | — |
| $Q_{\text{g}}$ | 保障绝对能接收到信号的极端稳健候选区域 | — |
| $M(T_{\max})$ | 在最大允许时间 $T_{\max}$ 内机器狗的时间可达域 | — |
| $\rho(q)$ | 第二检测点 $q$ 处的预测信号接收概率 | — |
| $J_{\boldsymbol{\omega}}(\xi)$ | 融合期望直径、期望面积与移动时间的无量纲综合损失函数 | — |
| $\omega_D, \omega_A, \omega_T$ | 综合损失函数中几何直径、区域面积与移动时间的权重系数 | — |
| **算法与训练参数** |  |  |
| $\mathcal I_{\mathrm{choice}}$ | 采样批次中真实的自由选择决策步集合 | 无 |
| $\mathcal I_V$ | 采样批次中有效价值估计样本集合 | 无 |
| $\widetilde A_t$ | 策略网络更新中使用的标准化优势值 | 无 |

## 五、模型建立与求解

### 5.1 问题一的模型建立与求解

### 5.1.1 有界测向误差下的集合定位模型

问题一采用集合估计（set-membership）与有界误差（bounded-error）模型。设第 $i$ 个检测点为 $S_i=(x_i,y_i)$，测得示向角 $\theta_i$，真实方位角为 $\theta_i^*$，则

$$\theta_i^*=\theta_i+\epsilon_i,\qquad \epsilon_i\in[-\delta,\delta],\qquad \delta=1^\circ.$$

该模型只使用误差硬界，即不要求误差服从均匀分布、高斯分布或独立同分布，也不通过重复观测平均缩小误差。

设干扰源位置为 $G=(x,y)$。则单次观测对应的两个半平面为

$$H_i^-:\ (x-x_i)\sin(\theta_i-\delta)-(y-y_i)\cos(\theta_i-\delta)\le 0,$$

$$H_i^+:\ -(x-x_i)\sin(\theta_i+\delta)+(y-y_i)\cos(\theta_i+\delta)\le 0,$$

从而测向楔形为 $W_i=H_i^-\cap H_i^+$。已经获得 direction 响应且接收半径最大可能为 $1500\,\mathrm m$，还给出真实源到检测点的确定性距离上界

$$\mathcal B(S_i,1500)=\{x\in\mathbb R^2:\|x-S_i\|_2\le1500\,\mathrm m\}.$$

该约束来自响应类型与接收半径上界。设目标大圆域记为

$$\mathcal C_0=\{(x,y)\in\mathbb R^2:x^2+y^2\le R_{\mathrm{target}}^2\},\qquad R_{\mathrm{target}}=1800\,\mathrm m.$$

综合 $n$ 次观测后，定位区域可精确为

$$\mathcal\Omega=\mathcal C_0\cap\bigcap_{i=1}^{n}\left[W_i\cap\mathcal B(S_i,1500)\right].$$

由于上述约束均为凸集，$\mathcal\Omega$ 是真实连续凸定位区域；其中包含圆域约束，故其边界一般含有圆弧，并非凸多边形。

![图 1 有界测向误差与楔形交会定位：(a) 单次误差楔形示意，为便于辨识对角宽作视觉放大；(b) $\delta=1^\circ$ 下的真实三站交会示例](results/figures/q1_bearing_wedge_intersection.png)

在后续计算中，对 $\mathcal C_0$ 和各个 $1500\,\mathrm m$ 接收上界圆均采用外切正多边形作保守OA（Outer Approximation），再与 bearing 半平面逐次裁剪，得到保守凸多边形外包 $K_{\mathrm{out}}$，满足 $\mathcal\Omega\subseteq K_{\mathrm{out}}$。若其顶点集为 $V$，则 $K_{\mathrm{out}}=\operatorname{conv}(V)$。为了保留所有的真实可行位置，我们选择将计算区域尽可能地保守化。

### 5.1.2 凸定位区域的直径计算

定位区域的几何直径定义为

$$D(\mathcal\Omega)=\sup_{A,B\in\mathcal\Omega}\|A-B\|_2.$$

旋转卡壳的实际计算对象是凸多边形 $K_{\mathrm{out}}$。若 $V=\{v_1,\ldots,v_m\}$ 为其顶点集，则

$$D(K_{\mathrm{out}})=\max_{1\le j<k\le m}\|v_j-v_k\|_2.$$

由 $\mathcal\Omega\subseteq K_{\mathrm{out}}$ 有 $D(\mathcal\Omega)\le D(K_{\mathrm{out}})$，得到真实定位区域直径的保守上界。对 $n$ 个输入点，我们先以 $O(n\log n)$ 时间构造凸包：设凸包顶点数为 $h$，再用旋转卡壳算法（Rotating Calipers）以 $O(h)$ 时间扫描对踵点，因此总复杂度为 $O(n\log n+h)$。

### 5.1.3 最小包围圆与直径圆覆盖判据

设 $p,q\in\mathcal\Omega$ 为一对距离达到 $D(\mathcal\Omega)$ 的点，以其为直径所得圆的半径为 $D(\mathcal\Omega)/2$。

**该直径圆并不一定覆盖 $\mathcal\Omega$**：对非退化锐角三角形，最长边的直径圆不能包含第三个顶点。

因此，仅知道直径 $D$ 尚不足以判断直径为 $D$ 的圆能否覆盖整个定位区域。
如图 2所示， 存在两个直径同为 $36\,\mathrm m$、但最小覆盖半径不同的凸集。

![图 2 相同直径下不同几何构型的最小包围圆与 $20\,\mathrm m$ 圆对比](results/figures/q1_same_diameter_coverage.png)

为获得准确覆盖判据，定义 $\mathcal\Omega$ 的最小包围圆（Minimum Enclosing Circle, MEC）半径

$$r^*(\mathcal\Omega)=\min_{c\in\mathbb R^2}\max_{x\in\mathcal\Omega}\|x-c\|_2.$$

对计算外包有 $K_{\mathrm{out}}=\operatorname{conv}(V)$。由于圆是凸集，任一包含全部顶点 $V$ 的圆必包含整个 $K_{\mathrm{out}}$，因此 $\operatorname{MEC}(K_{\mathrm{out}})=\operatorname{MEC}(V)$；同时由 $\mathcal\Omega\subseteq K_{\mathrm{out}}$ 有 $r^*(\mathcal\Omega)\le r^*(K_{\mathrm{out}})$。因而只需对 $K_{\mathrm{out}}$ 的顶点计算 MEC。

我们选择采用确定性增量 MEC 算法：先对 $V$ 排序、去重，再依次更新支撑边界。二维最小包围圆由不超过三个边界点确定，确定性实现的最坏时间复杂度为 $O(m^3)$。

Jung 定理给出任意二维紧凸集的严格界

$$\frac{D(\mathcal\Omega)}{2}\le r^*(\mathcal\Omega)\le\frac{D(\mathcal\Omega)}{\sqrt3}.$$

左端来自任意覆盖圆必须容纳一对相距 $D$ 的点，右端在正三角形构型达到。因此，问题一的核心判据可以改写为

$$\boxed{\ \text{存在直径为 }D(\mathcal\Omega)\text{ 的圆覆盖 }\mathcal\Omega\iff r^*(\mathcal\Omega)=\frac{D(\mathcal\Omega)}2\ }.$$

### 5.1.4 面向后续问题的安全计算接口

对精确定位区域，理论单点清除条件为 $r^*(\mathcal\Omega)\le20\,\mathrm m$。但在实际计算中我们使用 $D(K_{\mathrm{out}})$、$\operatorname{MEC}(K_{\mathrm{out}})$，并对最终输出圆心 $c_{\mathrm{out}}$ 在 $K_{\mathrm{out}}$ 上独立计算

$$r_{\max}=\max_{x\in K_{\mathrm{out}}}\|x-c_{\mathrm{out}}\|_2.$$

仅当 $r_{\max}\le19.999\,\mathrm m$ 时，我们可以标记 `CLEAR_READY`，否则标记 `COVERAGE_UNCERTAIN`。通过 $K_{\mathrm{out}}$ 的覆盖认证必然能够覆盖真实 $\mathcal\Omega$；外包未通过只表示当前计算不能安全确认覆盖，不等价于真实集合不存在半径 $20\,\mathrm m$ 的覆盖圆。该接口供后续问题调用，不在问题一中展开状态机或控制策略。

### 5.1.5 交会角敏感性分析

为分离交会角的影响，采用对称控制实验：两个检测点到真实源的距离均固定为 $L=500\,\mathrm m$，测角硬界固定为 $\delta=1^\circ$，交会角 $\alpha$ 作为观测几何的参数变量，取

$$\alpha\in\{5^\circ,10^\circ,15^\circ,20^\circ,30^\circ,45^\circ,60^\circ,75^\circ,90^\circ\}.$$

相应站间基线为 $2L\sin(\alpha/2)$，随 $\alpha$ 确定性变化。

在我们随机抽取数据，进行了多组试验后，结果表明，在该对称构型下，随着 $\alpha$ 从 $5^\circ$ 增至 $90^\circ$，定位区域直径 $D$ 与面积 $A$ 均显著下降，并随交会角继续增大而逐渐趋缓。

![图 3 两检测点距真实源均为 $L=500\,\mathrm m$、$\delta=1^\circ$ 时定位直径与面积的交会角敏感性](results/figures/q1_intersection_angle_sensitivity.png)

实验中出现的 $r^*=D/2$ 只属于该对称构型，不是一般凸集的性质。因此，第二检测点的选择应主动改善交会几何，同时兼顾移动成本和信号接收可靠性。

### 5.2 问题二的模型建立与求解

### 5.2.1 联合隐状态与贝叶斯联合后验更新

在首次检测点 $S = (x_S, y_S)$ 处，机器狗对已确认存在的全向干扰源获得示向度 $\theta$。设干扰源真实位置为 $G = (X, Y) \in \mathbb{R}^2$，有效信号接收半径为 $R \in [1000, 1500]\,\mathrm{m}$。系统将状态表示为联合隐状态 $(G, R)$。

#### (1) 联合先验分布

依据先验假设，干扰源在半径 $1800\,\mathrm{m}$ 的目标圆域 $\Omega = \{g \in \mathbb{R}^2 : \Vert{}g\Vert{} \le 1800\}$ 内按面积均匀分布，且接收半径 $R \sim \operatorname{Unif}[1000, 1500]$ 与位置 $G$ 独立。联合先验密度为：

$$\pi_0(g,r) = \frac{\mathbf 1_{\{g\in\Omega\}}}{\pi \cdot 1800^2} \frac{\mathbf 1_{\{1000\le r\le1500\}}}{500} \tag{1}$$

#### (2) 首次观测似然与联合后验

定义 $d_S(g) = \Vert{}g-S\Vert{}$，方位角 $h_S(g) = \operatorname{atan2}(g_y-S_y, g_x-S_x)$。对全向干扰源，首次检测响应 $Y_1$ 按距离分类：

$$Y_1 = \begin{cases} \mathrm{near}, & d_S(g) \le 5, \\ \mathrm{direction}(\theta), & 5 < d_S(g) \le r, \\ \mathrm{no\_signal}, & d_S(g) > r. \end{cases}$$

当 $Y_1 = \mathrm{direction}(\theta)$ 时，隐含条件 $d_S(g) > 5$。在测角误差服从密度 $f_\varepsilon$（主模型取 $\delta=1^\circ$ 均匀分布 $f_\varepsilon(e) = \frac{1}{2\delta} \mathbf{1}_{\{\vert{}e\vert{}\le \delta\}}$）下，观测似然为：

$$L_1(\theta\mid g,r,S) = \mathbf 1_{\{5<d_S(g)\le r\}}\, f_\varepsilon\!\left(\operatorname{wrap}\!\left[\theta-h_S(g)\right]\right) \tag{2}$$

记首次观测历史 $H_1 = (S, \mathrm{direction}, \theta)$，首次检测后的联合后验密度为：

$$\pi_1(g,r\mid H_1) = \frac{L_1(\theta\mid g,r,S)\pi_0(g,r)}{\int_{\Omega}\int_{1000}^{1500} L_1(\theta\mid \xi,\rho,S)\pi_0(\xi,\rho) \,\mathrm d\rho\,\mathrm d\xi} \tag{3}$$

对接收半径 $r$ 积分得到干扰源位置的边缘后验密度：

$$p_1(g\mid H_1) = \int_{1000}^{1500}\pi_1(g,r\mid H_1)\,\mathrm dr \propto \mathbf 1_{\{g\in\Omega\}} \mathbf 1_{\{5<d_S(g)\le1500\}} \mathbf 1_{\{\vert{}\operatorname{wrap}(\theta-h_S(g))\vert{}\le\delta\}} p_{\mathrm{rec}}\!\left(d_S(g)\right) \tag{4}$$

其中先验接收概率函数 $p_{\mathrm{rec}}(d)$ 为：

$$p_{\mathrm{rec}}(d) = \begin{cases} 1, & 0\le d\le1000, \\ \frac{1500-d}{500}, & 1000<d<1500, \\ 0, & d\ge1500. \end{cases} \tag{5}$$

式（4）表明：后验位置分布在楔形远端（$1000\,\mathrm{m} < d < 1500\,\mathrm{m}$）因接收概率衰减而并非均匀分布，这为后续精细化选点提供了准确的概率权重。由于同一干扰源的接收半径在一次任务中保持不变，后续检测继承联合后验 $\pi_1$，不能重新独立抽取 $R$。

---

### 5.2.2 定位几何可行域与决策候选空间构建

为了兼顾定位精度的提高与信号接收的可靠性，本模型采用“双层几何表示”对位置集合进行刻画。记测向楔形为 $W(S,\theta,\delta) = \{g: \vert{}\operatorname{wrap}(h_S(g)-\theta)\vert{} \le \delta\}$。

首次观测对应的精确位置支持集 $\mathcal{F}_1$ 及其闭合凸外包 $K_1$ 分别为：

$$\mathcal F_1 = \Omega\cap W(S,\theta,\delta) \cap\mathcal B(S,1500) \cap\{g:d_S(g)>5\} \tag{6}$$

$$K_1 = \Omega\cap W(S,\theta,\delta)\cap\mathcal B(S,1500), \qquad \mathcal F_1\subset K_1 \tag{7}$$

其中精确可行集用于贝叶斯更新与面积计算，闭合凸外包仅用于不改变其数值的直径与最小覆盖圆半径计算。

#### (1) 局部位移坐标系与时间可达域

为避免混淆全局坐标与移动决策，定义沿示向轴的单位方向向量 $u=(\cos\theta,\sin\theta)^{\mathsf T}$ 与法向向量 $n=(-\sin\theta,\cos\theta)^{\mathsf T}$，构造正交基矩阵 $B_\theta=[\,u\ \ n\,]$。定义局部决策变量 $\xi=(a,b)^{\mathsf T}$，其中 $a$ 为沿示向轴的纵向推进量，$b$ 为侧向基线偏移量。全局第二检测点坐标映射为：

$$q(\xi) = S + B_\theta\xi = S + a u + b n \tag{8}$$

当本次移动与检测的最大可用时间为 $T_{\max} > 5\,\mathrm{s}$（机器狗移动速度 $v=5\,\mathrm{m/s}$，检测耗时 $5\,\mathrm{s}$）时，允许的最大移动距离为 $R_T = 5(T_{\max}-5)$。局部坐标下的时间可达域表示为：

$$M(T_{\max}) = \left\{ S+a u+b n : a^2+b^2\le R_T^2 \right\} \tag{9}$$

#### (2) 高置信接收候选区域

第二检测点 $q$ 处的预测信号接收概率为：

$$\rho(q) = \int_{\Omega}\int_{1000}^{1500} \mathbf 1_{\{\Vert{}g-q\Vert{}\le r\}} \pi_1(g,r\mid H_1) \,\mathrm dr\,\mathrm dg \tag{10}$$

给定名义接收置信水平 $\eta_0$（如 $0.95$），为防止固定阈值导致候选域为空，定义有效自适应阈值 $\eta_{\mathrm{eff}} = \min\{\eta_0, \max_{q\in M(T_{\max})}\rho(q)\}$。第二检测点的最终候选决策空间为：

$$\mathcal C = \left\{ q\in M(T_{\max}) : \rho(q)\ge\eta_{\mathrm{eff}} \right\}, \qquad \Xi = \{\xi \in \mathbb{R}^2 : q(\xi) \in \mathcal{C}\} \tag{11}$$

---

### 5.2.3 第二次观测的分段贝叶斯更新与状态预测

设第二检测点为 $q=q(\xi)$，观测结果记为 $Z_q$。第二次观测似然在联合状态 $(G, R)$ 上定义：

$$L_2(z\mid g,r,q) = \begin{cases} \mathbf 1_{\{d_q(g)\le5\}}, & z=\mathrm{near},\\ \mathbf 1_{\{5<d_q(g)\le r\}}\, f_\varepsilon\!\left(\operatorname{wrap}[z-h_q(g)]\right), & z\in[-\pi,\pi)\quad(\mathrm{direction}),\\ \mathbf 1_{\{d_q(g)>r\}}, & z=\mathrm{no\_signal}. \end{cases} \tag{12}$$

更新后的二次联合后验密度与位置边缘后验分布分别为：

$$\pi_2(g,r\mid H_1,z,q) = \frac{L_2(z\mid g,r,q)\pi_1(g,r\mid H_1)}{\int_{\Omega}\int_{1000}^{1500} L_2(z\mid \xi,\rho,q)\pi_1(\xi,\rho\mid H_1) \,\mathrm d\rho\,\mathrm d\xi} \tag{13}$$

$$p_2(g\mid H_1,z,q) = \int_{1000}^{1500}\pi_2(g,r\mid H_1,z,q)\,\mathrm dr \tag{14}$$

对于连续方向响应 $z \in [-\pi, \pi)$，定义其预测次密度（Predictive Sub-density）：

$$\lambda_{\mathrm{dir}}(z\mid H_1,q) = \int_{\Omega}\int_{1000}^{1500} L_2(z\mid g,r,q)\pi_1(g,r\mid H_1) \,\mathrm dr\,\mathrm dg \tag{15}$$

二次观测后的几何位置闭合外包 $K_2(z,q)$ 按响应分类更新：

$$K_2(z,q) = \begin{cases} K_1\cap\mathcal B(q,5), & z=\mathrm{near},\\ K_1\cap W(q,z,\delta)\cap\mathcal B(q,1500), & z\in[-\pi,\pi),\\ K_1\setminus\mathcal B^\circ(q,1000), & z=\mathrm{no\_signal}. \end{cases} \tag{16}$$

当发生 $\mathrm{no\_signal}$ 时，$E_2^{\mathrm{no}}(q) = K_1\setminus\mathcal B^\circ(q,1000)$ 为非凸闭合外包，计算直径或覆盖圆时取其闭合凸包 $K_{2,\mathrm c}^{\mathrm{no}}(q) = \overline{\operatorname{conv}}(E_2^{\mathrm{no}}(q))$，由凸包性质知其直径与最小覆盖圆半径保持不变。

---

### 5.2.4 期望定位直径与综合损失优化模型

决策第二检测点 $q(\xi)$ 时，第二次观测结果尚未发生。全概率空间下的**期望定位直径** $\Psi_D(q)$ 与**期望定位面积** $\Psi_A(q)$ 计算如下：

$$\Psi_D(q) = \Pr(\mathrm{near}\mid H_1,q) D\!\left(K_2^{\mathrm{near}}(q)\right) + \Pr(\mathrm{no\_signal}\mid H_1,q) D\!\left(K_2^{\mathrm{no}}(q)\right) + \int_{-\pi}^{\pi} D\!\left(K_2^{\mathrm{dir}}(z,q)\right) \lambda_{\mathrm{dir}}(z\mid H_1,q)\,\mathrm dz \tag{17}$$

$$\Psi_A(q) = \mathbb E\!\left[ \mathcal A\!\left(K_2(Z_q,q)\right) \middle\vert{}H_1,q \right] \tag{18}$$

移动与检测的时间开销为 $T(q) = \frac{\Vert{}q-S\Vert{}}{5} + 5$。取初始无量纲化基准 $D_0 = D(K_1), A_0 = \mathcal{A}(K_1), T_0 = T_{\max}$，构造多目标综合损失函数 $J_{\boldsymbol\omega}(\xi)$：

$$\min_{\xi\in\Xi} J_{\boldsymbol\omega}(\xi) = \omega_D\frac{\Psi_D(q(\xi))}{D_0} + \omega_A\frac{\Psi_A(q(\xi))}{A_0} + \omega_T\frac{T(q(\xi))}{T_0} \tag{19}$$

$$\text{s.t.} \quad \omega_D,\omega_A,\omega_T\ge0, \qquad \omega_D+\omega_A+\omega_T=1, \qquad \omega_T>0 \tag{20}$$

由于显式要求 $\omega_T > 0$，避免了时间约束项退化失效。

---

### 5.2.5 鲁棒不确定集合与最坏性能约束模型

为使选点策略能够承受测角误差界限或接收距离的轻微失配，模型引入位置、测角、接收距离和时间的非负安全裕量 $\eta_g, \eta_\theta, \eta_r, \eta_T$。

#### (1) 鲁棒膨胀不确定集

定义膨胀后的位置支持集、放大测角界限与区间接收距离：

$$\mathcal S_1^{\mathrm{rob}} = \left(\mathcal S_1\oplus\mathcal B(0,\eta_g)\right)\cap\Omega, \quad \delta_{\mathrm{rob}}=\delta+\eta_\theta, \quad [r_{\min},r_{\max}] = [1000-\eta_r,\ 1500+\eta_r] \tag{21}$$

> **说明**：式（21）中的算子 $\oplus$ 表示 Minkowski 和，用于对位置不确定区域进行 $\eta_g$ 空间半径的几何膨胀，以防御空间位置的轻微扰动。
> 
> 

由此得到带时间裕量的安全可行域 $\Xi_{\mathrm{safe}}$ 与保证接收域 $\Xi_{\mathrm{rec}}$：

$$\Xi_{\mathrm{safe}} = \{ \xi\in\Xi : T(q(\xi))\le T_{\max}-\eta_T \}, \quad \Xi_{\mathrm{rec}} = \left\{ \xi\in\Xi_{\mathrm{safe}} : \sup_{g\in\mathcal S_1^{\mathrm{rob}}} \Vert{}q(\xi)-g\Vert{}\le r_{\min} \right\} \tag{22}$$

若任务要求第二次检测必然收到信号，取许可决策域 $\Xi_{\mathrm{adm}} = \Xi_{\mathrm{rec}}$；若 $\Xi_{\mathrm{rec}} = \varnothing$，则系统自动降级退回 $\Xi_{\mathrm{safe}}$ 并显式保留无信号分支（即退化至式 11 的自适应降级逻辑）。

#### (2) 最坏性能指标与双轨选点准则

对任意有效的鲁棒响应集合 $z \in \mathcal{Z}_{\mathrm{rob}}(q)$，定义最坏定位直径 $\overline{D}(\xi)$、最坏面积 $\overline{A}(\xi)$ 与最坏覆盖半径 $\overline{r}(\xi)$：

$$\overline D(\xi) = \sup_{z\in\mathcal Z_{\mathrm{rob}}} D\!\left(K_2^{\mathrm{rob}}(z,q(\xi))\right), \quad \overline A(\xi) = \sup_{z\in\mathcal Z_{\mathrm{rob}}} \mathcal A\!\left(K_2^{\mathrm{rob}}(z,q(\xi))\right), \quad \overline r(\xi) = \sup_{z\in\mathcal Z_{\mathrm{rob}}} r_*\!\left(K_2^{\mathrm{rob}}(z,q(\xi))\right) \tag{23}$$

根据信息完备度，选点准则分类如下：

$$\text{选点准则} = \begin{cases} \text{贝叶斯目标与最坏约束联合优化 (式 25)}, & f_\varepsilon \text{ 可信, } \Xi_{\mathrm{adm}}\ne\varnothing, \\ \text{纯极小极大几何最坏损失优化 (式 26)}, & \text{仅已知误差硬界, } \Xi_{\mathrm{adm}}\ne\varnothing, \\ \text{退回 } \Xi_{\mathrm{safe}} \text{ 并显式保留无信号分支}, & \Xi_{\mathrm{rec}}=\varnothing. \end{cases} \tag{24}$$

当概率密度 $f_\varepsilon$ 可信时，推荐优化模型为：

$$\min_{\xi\in\Xi_{\mathrm{adm}}} J_{\boldsymbol\omega}(\xi) \quad \text{s.t.} \quad \overline D(\xi)\le D_{\lim}, \quad \overline A(\xi)\le A_{\lim}, \quad (\text{若要求一次覆盖则加入 } \overline r(\xi)\le20) \tag{25}$$

当仅已知误差硬界时，采用纯极小极大（Minimax）优化：

$$\min_{\xi\in\Xi_{\mathrm{adm}}} \left[ \omega_D\frac{\overline D(\xi)}{D_0} + \omega_A\frac{\overline A(\xi)}{A_0} + \omega_T\frac{T(q(\xi))}{T_0} \right] \tag{26}$$

针对概率分布或参数偏好，可以在权重集合 $\mathcal{W}$ 上扫描生成鲁棒 Pareto 界面 $\mathcal{P}_{\mathrm{rob}}$，检验决策在时间与精度平衡上的稳定性。

---

### 5.2.6 求解算法流程与决策执行逻辑

为实现第二检测点选择与二次检测后的动态决策，设计了完整的算法流程。

---

> **【插入图 2：第二检测点贝叶斯实验设计与决策执行流程图】**
> *(建议在此处插入流程图。流程图绘制说明：包含“首次检测历史 $H_1$” $\rightarrow$ “构建时间可达域 $M(T_{\max})$ 与高置信候选域 $\mathcal{C}$” $\rightarrow$ “鲁棒可行性检验 $\Xi_{\mathrm{adm}}$” $\rightarrow$ “式(25)贝叶斯/式(26)极小极大优化” $\rightarrow$ “得到 $q^*$” $\rightarrow$ “执行二次检测” $\rightarrow$ “依据式(27)条件决策”的全流程。)*
> 
> 

---

#### (1) 求解算法核心步骤

1. **角度求积分离散化**：将连续角度 $[-\pi, \pi)$ 分解为 $M$ 个微元，中点为 $z_\ell$，采用步长逐级折半复合中点求积，直至相对误差小于 $\tau_{\mathrm{int}}=10^{-4}$。


2. **几何特征计算**：对每个离散角 $z_\ell$，调用问题一的半平面交算法求出 $K_2^{\mathrm{dir}}(z_\ell, q)$，利用**旋转卡壳算法**求极点间最大距离 $D(K_2)$，利用 Green 公式（含圆弧）或鞋盒公式求面积 $\mathcal{A}(K_2)$，并利用 Welzl 算法求 $r_*(K_2)$。


3. **最坏性能数值认证**：利用 Lipschitz 常数 $L_z(\xi)$ 对网格最大值进行上界补正，确保最坏损失的确定性上界认证 $\sup_{z} \mathcal{L}_{\mathrm{rob}}(z, \xi) \le \max_\ell \mathcal{L}_{\mathrm{rob}}(z_\ell, \xi) + \frac{L_z(\xi)\Delta z}{2}$。


4. **两阶段优化求解**：在局部坐标系 $(a,b)$ 内先进行极坐标粗网格扫描，找到极小值邻域后，以 SQP 算法求解连续变量 $\xi^* = (a^*, b^*)^{\mathsf T}$，得到全局最优检测点 $q^* = S + a^* u + b^* n$。



#### (2) 第二次观测后的条件决策规程

当机器狗在 $q^*$ 完成第二次检测并获得响应 $z$ 后，严格按如下决策规程执行：

$$a^*(z) = \begin{cases} \text{在 } q^* \text{ 处定位并清除}, & z = \text{near}, \\ \text{移动至鲁棒最小覆盖圆圆心 } c_{\mathrm{rob}}^* \text{ 后定位并清除}, & z = \text{direction}, \ r_*(K_2^{\mathrm{rob}}(z,q^*)) \le 20, \\ \text{保留联合后验 } \pi_2 \text{ 并准备继续测向}, & z = \text{direction}, \ r_*(K_2^{\mathrm{rob}}(z,q^*)) > 20, \\ \text{按 } \pi_2 \text{ 更新状态并在 } \Xi_{\mathrm{safe}} \text{ 中重新选点}, & z = \text{no\_signal}, \\ \text{检查观测并扩大鲁棒误差裕量重新计算}, & \int L_2 \pi_1 = 0 \text{ (观测冲突)}. \end{cases} \tag{27}$$

其中鲁棒最小覆盖圆圆心求解为 $c_{\mathrm{rob}}^* = \arg\min_c \max_{g\in K_2^{\mathrm{rob}}(z,q)} \Vert{}g-c\Vert{}$。当发生观测冲突（归一化常数为零）时，说明实际误差超出了预设界限，此时系统自动扩大安全裕量 $\eta_\theta, \eta_g$ 重新计算，而绝不强行归一化空后验。

---

### 5.2.7 成对蒙特卡洛验证与垂直布点对比分析

为了客观量化本模型优化策略 $q^*$ 相比于传统“沿首次示向轴垂直移动”策略（$q_\perp$）的几何与概率优势，建立成对蒙特卡洛（Pairwise Monte Carlo）仿真验证机制。

#### (1) 配对样本生成与对照点选择

试验生成 $N = 5000$ 组满足联合先验且经首次观测似然 $L_1$ 加权筛选的隐状态样本 $(G^{(k)}, R^{(k)})$，精确服从后验密度 $\pi_1(g,r \mid H_1)$。设定垂直对照点为移动相同距离 $L = \Vert{}q^* - S\Vert{}$ 的法线方向点 $q_\perp^\pm = S \pm L n$，并取两者中期望定位直径较优者作为保守对照：

$$q_\perp = \arg\min_{q \in \{q_\perp^+, q_\perp^-\}} \Psi_D(q) \tag{28}$$

在第二次检测采样时，保持样本自身的真实 $R^{(k)}$ 不变，仅独立生成新位置的测向误差。

---

> **【插入图 3：优化选点与垂直布点在后验分布下的交会几何与接收域对比图】**
> *(建议在此处插入几何对比图。图像绘制说明：画出测向楔形 $W$、高置信接收域 $\mathcal{C}$、时间可达域 $M(T_{\max})$、最优选点 $q^*$（兼顾纵向推进 $a^*$ 与横向偏移 $b^*$）以及垂直对照点 $q_\perp$。直观展示 $q^*$ 如何避开远端无信号风险并拉大有效交叉角。)*
> 
> 

---

#### (2) 性能评估指标与优越性验证

对策略 $s \in \{*, \perp\}$，统计平均定位直径 $\overline{D}_s$、$90\%$ 分位数尾部风险 $Q_{0.9, s}$、有效信号接收率 $\widehat{P}_{\mathrm{rec}, s}$ 以及单点可清除率 $\widehat{P}_{40, s}$（对应 $r_* \le 20\,\mathrm{m}$）。定义量化相对优势指标：

$$\Gamma_D = \frac{\overline D_\perp-\overline D_*}{\overline D_\perp}\times100\%, \qquad \Gamma_{0.9} = \frac{Q_{0.9,\perp}-Q_{0.9,*}}{Q_{0.9,\perp}}\times100\% \tag{29}$$

$$\Delta P_{\mathrm{rec}} = \widehat P_{\mathrm{rec},*} - \widehat P_{\mathrm{rec},\perp}, \qquad \Delta P_{40} = \widehat P_{40,*} - \widehat P_{40,\perp} \tag{30}$$

为了排除“优势仅来自垂直点无信号响应较多”的干扰因素，进一步在两种策略均成功收到示向角的配对子集 $\mathcal{I}_{\mathrm{both}} = \{k : Z_*^{(k)}=\mathrm{direction}, Z_\perp^{(k)}=\mathrm{direction}\}$ 上计算纯交叉几何改善率 $\Gamma_D^{\mathrm{both}}$。若仿真结果满足：

$$\Gamma_D > 0, \quad \Delta P_{\mathrm{rec}} > 0, \quad \Gamma_D^{\mathrm{both}} > 0 \tag{31}$$

则分别从**总体定位精度**、**信号接收可靠性**与**纯几何交会效果**三个维度充份证明了本文策略显著优于传统垂直布点策略。同时，对样本差值 $\Delta D_k = D_\perp^{(k)} - D_*^{(k)}$ 进行 Bootstrap 检验，确认其 $95\%$ 置信区间严格大于零，保证了量化优势的统计稳定性。

---

### 5.2.8 多频段全局决策的建模衔接接口说明

为了实现问题二单目标选点与后续多频段、多干扰源全局搜寻规划的无缝衔接，问题二模型向后续决策层输出标准化接口状态元组 $\mathcal{I}_2$：

$$\mathcal I_2 = \left( \pi_2,\ \mathcal S_2,\ \mathcal S_2^{\mathrm{rob}},\ K_2^{\mathrm{rob}},\ \mathcal A(K_2^{\mathrm{rob}}),\ D(K_2^{\mathrm{rob}}),\ r_*(K_2^{\mathrm{rob}}),\ c_{\mathrm{rob}}^* \right) \tag{32}$$

在后续多频道调度中，$\pi_2$ 用于指导机器狗对多频道的期望收益进行概率排序，$\mathcal{S}_2$ 保存精确几何约束。只有当满足 $r_*(K_2^{\mathrm{rob}}) \le 20\,\mathrm{m}$ 的确定性几何条件时，全局模型才允许向该频道分配“执行清除”动作，从而防止将高后验概率误判为已覆盖清除，确保了控制策略的绝对可靠性。
