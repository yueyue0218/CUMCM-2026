## 摘要

### 关键词

## 一、问题重述

复杂环境下无人移动终端（机器狗）对多频段无线电干扰源的定位与清除，本质上是一类结合测向几何推理、时空路径规划与多源状态决策的动态搜索优化问题。干扰源分布于指定目标圆域内，机器狗需要在遵循移动、频道切换、信号检测与清除等动作耗时约束的前提下，利用携带的测向设备采集示向度数据，并在存在有界测向误差的情形下实现对干扰源定位区域的精确构建与几何特性分析。

对于基础定位与覆盖判定，核心在于将各检测点采集到的方向范围转化为多半平面约束，进而交会形成待测区域的多边形边界。基于凸几何性质，该定位区域的直径由至少一对最远顶点取得，然而其直径圆并不必然能够完全覆盖整个定位区域。因此，需要引入严格的几何判定准则，并在 Jung 定理给出的理论边界指导下，结合确定性增量算法精确求解最小包围圆，从而在给定清除半径的限制下评估单点清除的可行性。

在多源搜索与动态清除任务中，随着干扰源数量、占用频道及辐射特性信息的未知性增加，单个频道的几何定位需进一步融合物理圆域裁剪、信号接收范围以及否定性观测信息，实现对可行区域的动态更新与预测交会。针对全向源与定向源等不同物理场景，系统须在极其有限的实际运行与虚拟时间预算内，协调移动路径与观测策略，不仅要实时维护各频道的定位状态与排除表，更需在最坏情形或期望效益准则下平衡测向收益与动作成本，最终构建起一套兼顾鲁棒性与高效性的控制策略，确保以最短的总虚拟时间完成对目标区域内全部干扰源的可靠定位与清除。

## 二、问题分析

### 2.1 问题一

问题一作为整个研究的基石，重点探讨在已知单一干扰源频道的理想化几何场景下，利用带有有界测量误差的示向度数据完成定位区域的构建、几何特征度量以及清除可行性判定。该问题的解决对于提升无人移动终端在复杂战场或无线电监控环境下的自主搜寻效能具有重要的工程意义与理论价值。通过对离散测量数据的几何化建模，能够将不确定的角度误差转化为确定性的可行空间约束，为后续多频段、多源复杂场景下的路径规划与动态决策提供坚重的空间状态表征基准。

在建模思路方面，问题一的核心在于由示向线构成的测向误差带交会算法及其后续的凸几何特性分析。首先，针对各观测点测得的方向角及给定的最大测向误差，可将每个观测点的示向度映射为以观测点为顶点的二维扇形区域。在有限空间内，这些扇形区域的边界在几何上对应于一系列半平面的交集。因此，定位区域的构建可归结为凸多边形（或半平面交）的计算问题，可通过逐次求交或基于对偶原理的半平面交算法实现定位区域顶点的精准提取。在获取定位区域的多边形表示后，其几何直径的计算转化为凸包顶点对之间的最远距离求解，可采用旋转卡壳算法高效确定至少一对最远顶点及其欧氏距离。

进一步分析定位区域与清除半径的关系可知，定位区域的直径圆（即以一对最远顶点为直径的圆）并不必然能够覆盖整个凸多边形区域。对于非退化锐角三角形，其最长边直径圆不能覆盖第三个顶点；对于直角三角形和钝角三角形，该直径圆可以覆盖第三个顶点。一般凸多边形仍须检查全部顶点或边界，不能仅由直径作出覆盖结论。为此，必须建立严格的几何覆盖判定模型。从理论边界来看，Jung定理给出了凸集直径与其最小外接圆半径之间的量化约束，为单点清除的可行性提供了重要的理论上限支撑；而在实际构造上，采用确定性增量最小包围圆算法，精准求得覆盖该定位区域所需的最小几何圆心与半径。若该最小外接圆半径小于或等于给定的清除半径，则说明机器狗仅需部署于该圆心位置施加一次清除动作即可完全覆盖定位区域；反之，若最小外接圆半径超出清除半径，则表明单次清除无法保证彻底销毁干扰源，需要后续增加观测点以缩小定位区域或采用多点联合清除策略。这一递进式建模思路成功地将无线电测向误差问题转化为了完备的计算几何优化问题。

### 2.2 问题二

针对首次示向检测后干扰源定位区域的不确定性问题，深入分析第二检测点的选择策略具有重要的工程应用价值与理论意义。在实际搜寻过程中，单次示向观测仅能将干扰源限制在一个自检测点向外辐射的狭长楔形区域内，无法对干扰源的真实距离进行有效约束。若机器狗盲目沿首次示向轴方向直线推进，两次观测的交角极小，极易导致交叉定位失效，使更新后的定位区域几何直径依然过大；反之，若机器狗仅沿着示向轴的垂直方向侧向移动，虽然能拉开观测基线、增大交叉角，却可能在移动过程中偏离测向楔形远端，甚至超出干扰源的有效信号接收半径，导致第二次检测面临无信号的风险。因此，第二检测点的选择本质上是在“移动时间成本”、“信号接收可靠性”与“交叉定位几何精度”三者之间寻优的贝叶斯实验设计问题。

为科学求解这一多目标决策问题，建模思路遵循“联合后验更新 — 时间与接收双重可达域 — 期望与最坏性能双轨优化”的逻辑主线。首先，将干扰源的空间位置 $G$ 与该干扰源固有的接收半径 $R$ 构建为联合隐状态 $(G, R)$，引入联合先验分布；在首次检测获得示向角后，基于贝叶斯定理推导联合后验分布，准确刻画位置支持集以及远端位置（$1000\,\mathrm{m} < d < 1500\,\mathrm{m}$）因接收概率衰减所形成的非均匀权重。同时，模型引入“精确可行集—闭合凸外包”双层几何表示，精确可行集用于贝叶斯更新和面积计算，而闭合凸外包用于直径、最小覆盖圆半径的保守认证。

在此基础上，结合机器狗的最大移动速度与本次任务的时限要求，构建局部位移坐标系下的时间可达域；同时根据联合后验计算候选点的信号接收概率，将接收概率达到置信水平的可达区域映射为第二检测点的候选决策空间。进一步地，考虑第二次检测可能出现的近距离响应（$\mathrm{near}$）、示向角响应（$\mathrm{direction}$）以及无信号响应（$\mathrm{no\_signal}$），计算第二次检测后定位区域的期望几何直径与期望面积；通过引入无量纲化的综合损失函数，将期望定位直径、期望面积与移动时间成本进行加权融合。

考虑到测向误差或接收距离可能存在失配，模型进一步通过 Minkowski 膨胀与边界扩大构造鲁棒安全可行域与最坏性能上界约束，形成极小极大（Minimax）或带最坏约束的贝叶斯优化选择准则。最后，采用成对蒙特卡洛仿真方法，将该最优策略与传统的“等距离垂直布点策略”在平均定位直径、尾部风险及信号接收率等指标上进行成对对比分析，从而为后续多频段、多目标的动态搜寻与清除任务提供坚实的单目标决策基础。

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

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| **几何与空间参数** |  |  |
| $R_{\text{target}}$ | 目标区域（大圆域）的半径 | $\text{m}$ |
| $R_{\text{clear}}$ | 机器狗干扰信号清除的有效半径 | $\text{m}$ |
| $P_i$ | 第 $i$ 个观测点（测向位置）的二维空间坐标 $(x_i, y_i)$ | $\text{m}$ |
| $\theta_i$ | 机器狗在观测点 $P_i$ 处测得的干扰源示向度角 | $\text{rad}$ 或 $^\circ$ |
| $\Delta \theta$ | 设备最大测向角度误差限界 | $\text{rad}$ 或 $^\circ$ |
| $\mathcal{\Omega}$ | 由示向度约束交会形成的干扰源可行定位区域（凸多边形） | 无（几何区域） |
| $D(\mathcal{\Omega})$ | 定位区域 $\mathcal{\Omega}$ 的几何直径（最远顶点对距离） | $\text{m}$ |
| $R_{\text{min}}$ | 定位区域 $\mathcal{\Omega}$ 的最小外接圆（最小包围圆）半径 | $\text{m}$ |
| $C_{\text{min}}$ | 定位区域 $\mathcal{\Omega}$ 的最小外接圆圆心坐标 | $\text{m}$ |
| **耗时与物理参数** |  |  |
| $v$ | 机器狗的移动速度 | $\text{m/s}$ |
| $t_{\text{switch}}$ | 机器狗进行频道切换所需的时间 | $\text{s}$ |
| $t_{\text{detect}}$ | 单次无线电信号检测与示向度采集所需的固定耗时 | $\text{s}$ |
| $t_{\text{clear}}$ | 执行一次干扰清除动作所需的固定耗时 | $\text{s}$ |
| $T_{\text{virtual}}$ | 系统完成全部任务所消耗的总虚拟时间 | $\text{s}$ |
| **状态与集合变量** |  |  |
| $f_k$ | 第 $k$ 个工作频道编号 | 无 |
| $S_k$ | 频道 $f_k$ 当前对应的干扰源空间定位候选集/区域 | 无（几何区域） |
| $\mathcal{E}_k$ | 针对频道 $f_k$ 已排除的不存在干扰源的区域集合 | 无（几何区域） |
| $u_i$ | 机器狗在第 $i$ 步采取的决策动作（移动、切换频道、检测或清除） | 无 |
| $G = (X,Y)$ | 干扰源的真实空间位置坐标 | $\text{m}$ |
| $R$ | 干扰源的有效信号接收半径 | $\text{m}$ |
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


## 五、模型建立与求解

### 5.1 问题一的模型建立与求解

### 5.1.1 干扰源定位区域的数学模型构建

在平面直角坐标系中，设目标区域为以原点为圆心、半径为 $R_{\text{target}}$ 的圆域 $\mathcal{C}_0 = \{(x,y) \in \mathbb{R}^2 \mid x^2 + y^2 \le R_{\text{target}}^2\}$。机器狗在 $n$ 个不同的观测点 $P_i = (x_i, y_i)$ ($i = 1, 2, \dots, n$) 处对某特定频道的干扰源进行测向，测得的示向度角为 $\theta_i$。

由于无线电测向设备存在已知最大角度误差限界 $\Delta \theta$，实际干扰源 $S = (x_s, y_s)$ 相对于观测点 $P_i$ 的真实方位角 $\theta_i^*$ 满足约束：


$$\theta_i - \Delta \theta \le \theta_i^* \le \theta_i + \Delta \theta$$

设从点 $P_i$ 出发、方向角为 $\phi$ 的射线半平面边界向量为 $\boldsymbol{n}(\phi) = (-\sin\phi, \cos\phi)^T$。则单次观测所确定的测向误差扇形区域可表示为两个半平面的交集。定义半平面方程：


$$H_{i,1}: (x - x_i)\sin(\theta_i - \Delta \theta) - (y - y_i)\cos(\theta_i - \Delta \theta) \le 0$$

$$H_{i,2}: -(x - x_i)\sin(\theta_i + \Delta \theta) + (y - y_i)\cos(\theta_i + \Delta \theta) \le 0$$

综合 $n$ 次观测信息以及干扰源必须位于目标大圆域 $\mathcal{C}_0$ 内的先验物理约束，干扰源的理论可行定位区域 $\mathcal{\Omega}$ 可精确表征为若干半平面与初始圆域的交集：


$$\mathcal{\Omega} = \mathcal{C}_0 \cap \left( \bigcap_{i=1}^n (H_{i,1} \cap H_{i,2}) \right)$$

在数学模型中，目标区域仍取真实圆域 $\mathcal{C}_0$。在算法实现中，采用外切正多边形作为该圆域的保守外近似，再利用 Sutherland-Hodgman 算法逐次进行半平面裁剪，得到计算用凸多边形 $K_{\text{out}}$ 及其顶点集合 $V = \{v_1, v_2, \dots, v_m\}$。真实圆域包含在此外切正多边形内，因而真实可行集合也包含于 $K_{\text{out}}$，不会因圆域离散化误删真实可行位置；代价是计算结果可能略偏保守。

---

### 5.1.2 定位区域几何直径的精确度量

凸多边形定位区域 $\mathcal{\Omega}$ 的几何直径 $D(\mathcal{\Omega})$ 定义为区域内任意两点间欧氏距离的上确界：


$$D(\mathcal{\Omega}) = \sup_{A, B \in \mathcal{\Omega}} \Vert{}A - B\Vert{}_2$$

基于凸集几何性质，凸多边形内部任意两点间的最大距离必在某对极点（顶点）处取得：


$$D(\mathcal{\Omega}) = \max_{1 \le j < k \le m} \Vert{}v_j - v_k\Vert{}_2$$

对一般的 $n$ 个输入点，当前主算法先以 $O(n\log n)$ 时间求凸包。设凸包顶点数为 $h$，再采用 **旋转卡壳算法（Rotating Calipers）** 以 $O(h)$ 时间扫描对踵点；总复杂度为 $O(n\log n+h)$。遍历全部凸包顶点对的 $O(h^2)$ 方法仅保留为独立穷举核验方法。旋转卡壳步骤如下：

1. 沿着凸多边形 $V$ 的边界按逆时针方向寻找具有平行切线的对踵点对（Antipodal Pairs）；
2. 依次计算所有对踵点对之间的距离；
3. 输出最大距离及取得该距离的至少一对顶点 $(v_p, v_q)$，即为定位区域的直径 $D(\mathcal{\Omega})$。

对应地，以该最远顶点对为直径的圆称为**直径圆** $\mathcal{C}_D$，其圆心 $C_D$ 与半径 $R_D$ 分别为：


$$C_D = \frac{v_p + v_q}{2}, \quad R_D = \frac{D(\mathcal{\Omega})}{2} = \frac{\Vert{}v_p - v_q\Vert{}_2}{2}$$

---

### 5.1.3 清除覆盖判定与最小包围圆模型

#### (1) 直径圆覆盖判定的局限性与几何分析

直观上，直径圆 $\mathcal{C}_D$ 提供了覆盖最远顶点对的最紧凑局部边界。然而，**直径圆并不必然能够完全包含凸多边形定位区域 $\mathcal{\Omega}$**。

对于非退化锐角三角形，最长边的直径圆不能覆盖第三个顶点；直角三角形的第三点位于圆周上，钝角三角形的第三点位于圆内。对于一般凸多边形，仍可能存在某些顶点 $v_k$ 满足 $\Vert{}v_k - C_D\Vert{}_2 > R_D$，故必须检查全部顶点或边界。因此，仅以 $R_D \le R_{\text{clear}}$ 作为单点清除的判定依据会导致漏清除风险。

如图 1 所示，对于几何直径同为 $D = 36\text{ m}$ 的正方形与正三角形定位区域，正方形的最小包围圆半径为 $r^* = 18\text{ m}$，半径为 $20\text{ m}$ 的清除圆域（灰色虚线）能够完全覆盖整个区域；而正三角形的最小包围圆半径达到了理论上限 $r^* = \frac{36}{\sqrt{3}} \approx 20.78\text{ m}$，此时半径为 $20\text{ m}$ 的清除圆域无法完全覆盖该区域。这直观证明了定位区域的几何构型对其最小覆盖外接圆尺寸有着决定性影响。

![图 1 相同直径下不同几何构型的最小包围圆与清除覆盖对比](results/figures/q1_same_diameter_coverage.png)

#### (2) 最小包围圆（MEC）的精确求解

为了确保机器狗施加一次清除动作（覆盖半径为 $R_{\text{clear}}$）即可 $100\%$ 销毁区域内任意位置的干扰源，必须求解能够完全包含凸多边形 $\mathcal{\Omega}$ 的**最小包围圆（Minimum Enclosing Circle, MEC）**，记为 $\mathcal{C}_{\text{min}}(C_{\text{min}}, R_{\text{min}})$。

该问题可转化为如下极小化极大约束优化问题：


$$\min_{C \in \mathbb{R}^2} \max_{k=1,\dots,m} \Vert{}v_k - C\Vert{}_2$$

$$\text{s.t.} \quad R_{\text{min}} = \max_{k=1,\dots,m} \Vert{}v_k - C_{\text{min}}\Vert{}_2$$

当前实现采用**确定性增量最小包围圆算法**。首先对顶点排序、去重，再依次扫描各点：若新点位于当前圆内，则圆保持不变；若新点位于当前圆外，则该点必须参与更新后最小包围圆的支撑边界。算法在嵌套扫描中依次考虑由一个边界点、两个边界点所确定的直径圆，以及三个非共线边界点所确定的外接圆。根据二维最小包围圆由不超过三个边界点支撑的性质，可由这些候选确定最优圆。当前直接实现的最坏时间复杂度为 $O(m^3)$。

最终输出唯一的最小包围圆圆心 $C_{\text{min}}$ 及最小外接半径 $R_{\text{min}}$。

#### (3) 基于 Jung 定理的理论上下界判定与可行性准则

在理论几何上，**Jung 定理（Jung's Theorem）** 给出了任意二维紧凸集直径 $D(\mathcal{\Omega})$ 与其最小外接圆半径 $R_{\text{min}}$ 之间的严格几何限界：


$$\frac{D(\mathcal{\Omega})}{2} \le R_{\text{min}} \le \frac{D(\mathcal{\Omega})}{\sqrt{3}}$$

* **下限 $\frac{D(\mathcal{\Omega})}{2}$**：对应于多边形顶点能完全被其直径圆覆盖的情形（如矩形或直角/钝角三角形构型）；
* **上限 $\frac{D(\mathcal{\Omega})}{\sqrt{3}}$**：对应于最坏几何构型（如正三角形构型）。

结合 Jung 定理与确定性增量算法计算得到的 $R_{\text{min}}$，可建立如下确定性的单点清除可行性判定准则：

$$\text{单点清除可行性} =  \begin{cases}  \text{完全可行 } (100\% \text{ 覆盖}), & \text{若 } R_{\text{min}} \le R_{\text{clear}} \\  \text{不可行 (需补充观测或多点清除)}, & \text{若 } R_{\text{min}} > R_{\text{clear}}  \end{cases}$$

对于真实可行集合的精确最小包围圆，若 $R_{\text{min}} \le R_{\text{clear}}=20\,\mathrm{m}$，则存在单圆覆盖；反之则不存在半径 $20\,\mathrm{m}$ 的单圆覆盖。

工程执行时不能只依赖最小包围圆例程返回的理论半径。对最终实际输出圆心 $c_{\text{out}}$，还需在保守外包 $K_{\text{out}}$ 上独立计算

$$r_{\max}=\max_{x\in K_{\text{out}}}\Vert{}x-c_{\text{out}}\Vert{}_2.$$

当前实现采用 fail-closed 工程裕量：仅当 $r_{\max}\le 19.999\,\mathrm{m}$ 时标记 `CLEAR_READY`；否则返回 `COVERAGE_UNCERTAIN`，不得直接下发清除。由于 $K_{\text{out}}$ 是保守外包，外包未通过认证只能表示当前计算不足以安全确认覆盖，不能据此证明真实可行区域不存在半径 $20\,\mathrm{m}$ 的覆盖圆。


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
