## 问题二：基于联合后验与期望定位直径的第二检测点选择

### 1 问题分析

设机器狗在首次检测点 \(S=(x_S,y_S)\) 对某一已确认存在的全向干扰源获得示向度 \(\theta\)。一次示向观测只能将源位置限制在狭长楔形区域内，不能确定其距离。若第二检测点与首次示向轴近似共线，则两次方位约束交角过小，定位区域直径仍然较大；若仅沿示向轴的垂直方向移动，则又可能远离楔形远端，导致无法接收信号。因此，本问归结为如下贝叶斯实验设计问题：

\[
\boxed{\text{在保证或高概率接收的候选区域内，选择使第二次检测后期望定位直径最小的检测点。}}
\tag{1}
\]

模型采用问题一建立的“精确可行集—闭合凸外包”双层表示。精确可行集用于贝叶斯更新，凸外包用于面积、直径及接收保证的可靠计算。题设只给出测向误差硬界而未给出其概率密度，故先以一般密度 \(f_\varepsilon\) 推导；进行平均意义下的策略比较时，再补充

\[
f_\varepsilon(e)=
\begin{cases}
\dfrac{1}{2\delta},& |e|\le\delta,\\[4pt]
0,& |e|>\delta,
\end{cases}
\qquad \delta=1^\circ .
\tag{2}
\]

其中 \(1.005^\circ\) 仅作为考虑额外舍入机制时的敏感性情景，不与主模型中的 \(1^\circ\) 硬界重复叠加。

### 2 首次检测后的联合后验

#### 2.1 先验分布

记干扰源位置为 \(G=(X,Y)\)，有效接收半径为 \(R\)。依据新增假设，源位置在半径 \(1800\,\mathrm m\) 的目标圆域

\[
\Omega=\{g\in\mathbb R^2:\|g\|\le1800\}
\tag{3}
\]

内按面积均匀分布，且

\[
G\sim\operatorname{Unif}(\Omega),\qquad
R\sim\operatorname{Unif}[1000,1500],\qquad G\perp R.
\tag{4}
\]

于是联合先验为

\[
\pi_0(g,r)=
\frac{\mathbf 1_{\{g\in\Omega\}}}{\pi\,1800^2}
\frac{\mathbf 1_{\{1000\le r\le1500\}}}{500}.
\tag{5}
\]

式（4）仅表示观测前独立。接收事件发生后，\(G\) 与 \(R\) 通常产生后验相关性；由于同一干扰源的接收半径在一次任务中保持不变，后续检测必须沿用其联合后验，不能重新独立抽取 \(R\)。

#### 2.2 首次观测似然

定义

\[
d_s(g)=\|g-s\|,\qquad
h_s(g)=\operatorname{atan2}(g_y-s_y,g_x-s_x),
\tag{6}
\]

并以 \(\operatorname{wrap}(\alpha)\in[-\pi,\pi)\) 表示环形角差。首次检测结果为示向度 \(\theta\) 时，

\[
L_1(\theta\mid g,r,S)
=
\mathbf 1_{\{5<d_S(g)\le r\}}\,
f_\varepsilon\!\left(
\operatorname{wrap}\!\left[\theta-h_S(g)\right]
\right).
\tag{7}
\]

因此，首次检测后的联合后验为

\[
\boxed{
\pi_1(g,r\mid H_1)
=
\frac{L_1(\theta\mid g,r,S)\pi_0(g,r)}
{\displaystyle
\int_{\Omega}\int_{1000}^{1500}
L_1(\theta\mid \xi,\rho,S)\pi_0(\xi,\rho)
\,\mathrm d\rho\,\mathrm d\xi}},
\tag{8}
\]

其中 \(H_1=(S,\mathrm{direction},\theta)\)。位置边缘后验为

\[
p_1(g\mid H_1)=\int_{1000}^{1500}\pi_1(g,r\mid H_1)\,\mathrm dr .
\tag{9}
\]

由 \(R\sim\operatorname{Unif}[1000,1500]\) 可得距离为 \(d\) 时的先验接收概率

\[
p_{\mathrm{rec}}(d)=
\begin{cases}
1,&0\le d\le1000,\\[2pt]
\dfrac{1500-d}{500},&1000<d<1500,\\[6pt]
0,&d\ge1500.
\end{cases}
\tag{10}
\]

当采用式（2）的均匀测角误差时，式（9）可写为

\[
p_1(g\mid H_1)\propto
\mathbf 1_{\{g\in\Omega\}}
\mathbf 1_{\{5<d_S(g)\le1500\}}
\mathbf 1_{\{|\operatorname{wrap}(\theta-h_S(g))|\le\delta\}}
p_{\mathrm{rec}}\!\left(d_S(g)\right).
\tag{11}
\]

由式（11）可见，虽然全局位置先验为面积均匀分布，首次观测后的后验一般不在定位楔形内均匀：当 \(1000<d_S(g)<1500\) 时，远端位置因接收概率较低而获得较小权重。仅在所考察区域全部满足 \(d_S(g)\le1000\)，且采用式（2）时，位置后验才退化为区域内均匀分布。

#### 2.3 几何可行域

记测向楔形为

\[
W(S,\theta,\delta)=
\left\{
g:
\left|\operatorname{wrap}\!\left[h_S(g)-\theta\right]\right|
\le\delta
\right\}.
\tag{12}
\]

首次观测对应的精确位置支持集及闭合凸外包分别为

\[
\mathcal F_1
=
\Omega\cap W(S,\theta,\delta)
\cap\mathcal B(S,1500)
\cap\{g:d_S(g)>5\},
\tag{13}
\]

\[
\boxed{
K_1=
\Omega\cap W(S,\theta,\delta)\cap\mathcal B(S,1500),
\qquad \mathcal F_1\subset K_1 .
}
\tag{14}
\]

其中，\(5\,\mathrm m\) 内排除单独保存，不强行并入凸集 \(K_1\)。

### 3 定位区域的面积、直径与覆盖半径

设问题一输出的凸多边形近似为

\[
P=\operatorname{conv}\{V_1,\ldots,V_m\},\qquad
V_j=(x_j,y_j),\quad V_{m+1}=V_1 .
\tag{15}
\]

若边界仅由线段组成，则定位区域面积为

\[
\boxed{
\mathcal A(P)=
\frac12\left|
\sum_{j=1}^{m}
(x_jy_{j+1}-x_{j+1}y_j)
\right|.
}
\tag{16}
\]

若 \(\Omega\) 或接收圆盘参与裁剪而产生圆弧边界，则采用格林公式

\[
\boxed{
\mathcal A(K)=\frac12\oint_{\partial K}(x\,\mathrm dy-y\,\mathrm dx).
}
\tag{17}
\]

对圆心 \(C=(c_x,c_y)\)、半径 \(\rho\)、参数区间 \(t\in[\alpha,\beta]\) 的逆时针圆弧，其面积贡献为

\[
\mathcal A_{\mathrm{arc}}
=\frac12\!\left[
\rho c_x(\sin\beta-\sin\alpha)
+\rho c_y(\cos\alpha-\cos\beta)
+\rho^2(\beta-\alpha)
\right].
\tag{18}
\]

定位区域直径定义为

\[
D(K)=\max_{x,y\in K}\|x-y\|.
\tag{19}
\]

相应地，

\[
D(P)=\max_{1\le i<j\le m}\|V_i-V_j\|,
\tag{20}
\]

而含圆弧的紧凸区域采用支持函数

\[
\boxed{
D(K)=
\max_{\varphi\in[0,2\pi)}
\left[
h_K(e_\varphi)+h_K(-e_\varphi)
\right],
\quad
h_K(v)=\max_{x\in K}v^{\mathsf T}x,
\quad
e_\varphi=(\cos\varphi,\sin\varphi)^{\mathsf T}.
}
\tag{21}
\]

为判断区域是否能被一次光学定位完整覆盖，再定义最小覆盖圆半径

\[
r_*(K)=\min_{c\in\mathbb R^2}\max_{g\in K}\|g-c\|.
\tag{22}
\]

由 Jung 不等式，

\[
\frac12D(K)\le r_*(K)\le\frac{1}{\sqrt3}D(K).
\tag{23}
\]

因此，

\[
\begin{cases}
D(K)>40\,\mathrm m
\ \Longrightarrow\ r_*(K)>20\,\mathrm m,\\[2pt]
D(K)\le20\sqrt3\,\mathrm m
\ \Longrightarrow\ r_*(K)\le20\,\mathrm m,\\[2pt]
20\sqrt3<D(K)\le40
\ \Longrightarrow\ \text{须直接计算 }r_*(K).
\end{cases}
\tag{24}
\]

即 \(D(K)\le40\,\mathrm m\) 只是存在 \(20\,\mathrm m\) 覆盖圆的必要条件，\(D(K)\le20\sqrt3\,\mathrm m\) 才是充分条件；最终仍以 \(r_*(K)\le20\,\mathrm m\) 为准确判据。

### 4 第二检测点的候选区域

#### 4.1 保证接收区域

令

\[
\mathcal S_1=\operatorname{supp}\bigl(p_1(\cdot\mid H_1)\bigr).
\tag{25}
\]

由于所有全向源的接收半径均不小于 \(1000\,\mathrm m\)，若第二检测点 \(q\) 满足

\[
\boxed{
Q_{\mathrm g}
=
\left\{
q:\sup_{g\in\mathcal S_1}\|q-g\|\le1000
\right\}
=
\bigcap_{g\in\mathcal S_1}\mathcal B(g,1000),
}
\tag{26}
\]

则无论真实接收半径取何值，第二次检测均能获得 \(\mathrm{near}\) 或 \(\mathrm{direction}\)，不会出现 \(\mathrm{no\_signal}\)。

若用凸多边形

\[
\widehat P_1=\operatorname{conv}\{V_1,\ldots,V_m\}
\supseteq\operatorname{conv}(\mathcal S_1)
\tag{27}
\]

对位置支持集作保守外包，则

\[
\boxed{
\widehat Q_{\mathrm g}
=\bigcap_{j=1}^{m}\mathcal B(V_j,1000)
\subseteq Q_{\mathrm g}.
}
\tag{28}
\]

式（28）把无限约束化为有限圆盘交，且保持“必能接收”的保证。

#### 4.2 局部坐标下的显式条件

令

\[
u=(\cos\theta,\sin\theta)^{\mathsf T},\qquad
n=(-\sin\theta,\cos\theta)^{\mathsf T},\qquad
q=S+a u+b n .
\tag{29}
\]

不利用目标圆域对楔形的进一步裁剪时，可用

\[
T=\operatorname{conv}
\{S,\ S+1500u+w n,\ S+1500u-w n\},
\qquad w=1500\tan\delta
\tag{30}
\]

保守外包 \(K_1\)。于是第二检测点的一个显式保证接收子区域为

\[
\boxed{
\widehat Q_{\mathrm g}^{\,T}
=
\left\{S+a u+b n:
\begin{array}{l}
a^2+b^2\le1000^2,\\
(a-1500)^2+(b-w)^2\le1000^2,\\
(a-1500)^2+(b+w)^2\le1000^2
\end{array}
\right\}.
}
\tag{31}
\]

主模型取 \(\delta=1^\circ\) 时 \(w\approx26.183\,\mathrm m\)；敏感性情景取 \(\delta=1.005^\circ\) 时 \(w\approx26.314\,\mathrm m\)。式（31）同时要求检测点覆盖楔形近端与两个远端，因而自然形成“沿示向方向前进并适度横移”的候选区域。

#### 4.3 高概率接收区域与时间约束

若保证接收区域为空，或其与任务剩余时间约束无交，则根据联合后验定义接收概率

\[
\rho(q)
=
\int_{\Omega}\int_{1000}^{1500}
\mathbf 1_{\{\|g-q\|\le r\}}
\pi_1(g,r\mid H_1)
\,\mathrm dr\,\mathrm dg
\tag{32}
\]

及置信候选区域

\[
Q_\eta=\{q:\rho(q)\ge\eta\},\qquad 0<\eta<1.
\tag{33}
\]

若本次移动与检测的可用时间为 \(T_{\max}\)，则

\[
M(T_{\max})
=
\left\{
q:\frac{\|q-S\|}{5}+5\le T_{\max}
\right\}.
\tag{34}
\]

最终候选区域按条件选取：

\[
\boxed{
\mathcal C=
\begin{cases}
\widehat Q_{\mathrm g}\cap M(T_{\max}),
&\widehat Q_{\mathrm g}\cap M(T_{\max})\ne\varnothing,\\[2pt]
Q_\eta\cap M(T_{\max}),
&\widehat Q_{\mathrm g}\cap M(T_{\max})=\varnothing .
\end{cases}}
\tag{35}
\]

第一种情形提供确定性接收保证；第二种情形是受时间或几何限制时的概率退化方案。

### 5 第二次检测的分段贝叶斯更新

设第二检测点为 \(q\)，结果记为 \(Z_q\)。对不同检测位置采用误差独立的工作近似，而同一位置的环境误差固定：

\[
\varepsilon(q)=
\begin{cases}
\varepsilon(S),&q=S,\\
\text{与 }\varepsilon(S)\text{ 独立},&q\ne S.
\end{cases}
\]

故原地重复检测不产生新的独立信息。由于 \(R\) 是同一源的固定属性，第二次观测似然应在联合状态 \((G,R)\) 上定义：

\[
L_2(z\mid g,r,q)=
\begin{cases}
\mathbf 1_{\{d_q(g)\le5\}},
&z=\mathrm{near},\\[4pt]
\mathbf 1_{\{5<d_q(g)\le r\}}\,
f_\varepsilon\!\left(
\operatorname{wrap}[z-h_q(g)]
\right),
&z\in[-\pi,\pi),\\[4pt]
\mathbf 1_{\{d_q(g)>r\}},
&z=\mathrm{no\_signal}.
\end{cases}
\tag{36}
\]

于是

\[
\boxed{
\pi_2(g,r\mid H_1,z,q)
=
\frac{L_2(z\mid g,r,q)\pi_1(g,r\mid H_1)}
{\displaystyle
\int_{\Omega}\int_{1000}^{1500}
L_2(z\mid \xi,\rho,q)\pi_1(\xi,\rho\mid H_1)
\,\mathrm d\rho\,\mathrm d\xi},
}
\tag{37}
\]

\[
p_2(g\mid H_1,z,q)
=\int_{1000}^{1500}\pi_2(g,r\mid H_1,z,q)\,\mathrm dr .
\tag{38}
\]

相应的位置支持集可按检测结果分类为

\[
\mathcal S_2(z,q)=
\begin{cases}
\mathcal S_1\cap\mathcal B(q,5),
&z=\mathrm{near},\\[3pt]
\operatorname{proj}_g\!\left\{
(g,r)\in\operatorname{supp}(\pi_1):
\begin{array}{l}
5<d_q(g)\le r,\\
|\operatorname{wrap}[z-h_q(g)]|\le\delta
\end{array}
\right\},
&z\in[-\pi,\pi),\\[9pt]
\operatorname{proj}_g\!\left\{
(g,r)\in\operatorname{supp}(\pi_1):d_q(g)>r
\right\},
&z=\mathrm{no\_signal}.
\end{cases}
\tag{39}
\]

为进行可靠几何计算，可分别采用如下闭合外包：

\[
\mathcal S_2(z,q)\subseteq K_2(z,q)=
\begin{cases}
K_1\cap\mathcal B(q,5),
&z=\mathrm{near},\\[2pt]
K_1\cap W(q,z,\delta)\cap\mathcal B(q,1500),
&z\in[-\pi,\pi),\\[2pt]
K_1\setminus\mathcal B^\circ(q,1000),
&z=\mathrm{no\_signal}.
\end{cases}
\tag{40}
\]

其中，\(\mathrm{direction}\) 情形的 \(5\,\mathrm m\) 内排除及 \(\mathrm{no\_signal}\) 情形的非凸排除均单独保存。若 \(q\in Q_{\mathrm g}\)，则第三种结果的预测概率为零。

### 6 距离后验检验

令第二检测点到干扰源的真实距离为

\[
R_q=\|G-q\|.
\tag{41}
\]

在观测历史 \(H_2=(H_1,q,z)\) 下，其后验分布函数为

\[
\boxed{
F_{R_q}(d\mid H_2)
=
\int_{\{g:\|g-q\|\le d\}}
p_2(g\mid H_2)\,\mathrm dg .
}
\tag{42}
\]

给定显著性水平 \(\alpha\)，等尾可信区间为

\[
I_{1-\alpha}(H_2)
=
\left[
F_{R_q}^{-1}\!\left(\frac{\alpha}{2}\middle|H_2\right),
F_{R_q}^{-1}\!\left(1-\frac{\alpha}{2}\middle|H_2\right)
\right].
\tag{43}
\]

对 \(N\) 次独立仿真，定义经验覆盖率

\[
\widehat C_{1-\alpha}
=
\frac1N\sum_{k=1}^{N}
\mathbf 1_{\{R_q^{(k)}\in I_{1-\alpha}^{(k)}\}}.
\tag{44}
\]

若模型校准合理，则应有

\[
\widehat C_{1-\alpha}\approx1-\alpha.
\tag{45}
\]

此外，任取拟清除位置 \(c\)，源位于其 \(20\,\mathrm m\) 范围内的后验概率为

\[
P_{\mathrm{clr}}(c\mid H_2)
=
\int_{\mathcal B(c,20)}p_2(g\mid H_2)\,\mathrm dg .
\tag{46}
\]

式（46）可用于概率排序，但不能替代确定性清除判据。只有

\[
\sup_{g\in\mathcal S_2}\|g-c\|\le20
\quad\Longleftrightarrow\quad
r_*(\mathcal S_2)\le20
\tag{47}
\]

时，才能在硬误差模型下保证一次光学定位覆盖全部可能位置。

### 7 最小期望定位直径模型

第二次检测结果尚未发生时，其后验预测分布为

\[
p(z\mid H_1,q)
=
\int_{\Omega}\int_{1000}^{1500}
L_2(z\mid g,r,q)\pi_1(g,r\mid H_1)
\,\mathrm dr\,\mathrm dg .
\tag{48}
\]

定义第二次检测后的期望定位直径

\[
\begin{aligned}
\Psi_D(q)
=\;&
\Pr(\mathrm{near}\mid H_1,q)
D\!\left(K_2^{\mathrm{near}}(q)\right)\\
&+
\Pr(\mathrm{no\_signal}\mid H_1,q)
D\!\left(K_2^{\mathrm{no}}(q)\right)\\
&+
\int_{-\pi}^{\pi}
D\!\left(K_2^{\mathrm{dir}}(z,q)\right)
p(z,\mathrm{direction}\mid H_1,q)\,\mathrm dz .
\end{aligned}
\tag{49}
\]

同理定义期望面积

\[
\Psi_A(q)
=
\mathbb E\!\left[
\mathcal A\!\left(K_2(Z_q,q)\right)
\middle|H_1,q
\right],
\tag{50}
\]

以及单次移动检测时间

\[
T(q)=\frac{\|q-S\|}{5}+5.
\tag{51}
\]

为避免将“米、平方米、秒”以任意权重直接相加，采用字典序目标：

\[
\boxed{
q^*
\in
\operatorname*{arg\,lexmin}_{q\in\mathcal C}
\left(
\Psi_D(q),\,
\Psi_A(q),\,
T(q)
\right).
}
\tag{52}
\]

其中仅对预测概率为正的观测结果计入期望；约定空集的面积与直径均为零。即先最小化期望定位直径；若多个检测点的直径指标在计算精度内相同，则依次选择期望面积更小、移动检测时间更短者。由此得到的最优策略可写为

\[
\boxed{
\text{先沿示向方向覆盖远端，再以横向位移增大两次方位线交角，并由后验期望直径确定二者比例。}
}
\tag{53}
\]

若不愿为测向误差补充概率密度，则式（49）中的期望不可辨识，应改用仅依赖硬边界的极小极大准则：

\[
q_{\mathrm{mm}}^*
\in
\arg\min_{q\in\mathcal C}
\sup_{z\in\mathcal Z(q)}
D\!\left(K_2(z,q)\right).
\tag{54}
\]

因此，选点模式按可用信息分类为

\[
\boxed{
\text{选点准则}=
\begin{cases}
\text{贝叶斯期望直径最小化，式（52）},
& f_\varepsilon\ \text{已给定},\\[2pt]
\text{最坏情形直径最小化，式（54）},
& \text{仅已知误差硬界}.
\end{cases}}
\tag{55}
\]

### 8 第二次观测后的条件决策

第二次检测完成后，按下列条件执行：

\[
\boxed{
a^*(z)=
\begin{cases}
\text{在 }q\text{ 处定位并清除},
&z=\mathrm{near},\\[2pt]
\text{移动至最小覆盖圆圆心 }c^*\text{ 后定位并清除},
&z=\mathrm{direction},\ r_*(\mathcal S_2)\le20,\\[2pt]
\text{保留后验并继续测量},
&z=\mathrm{direction},\ r_*(\mathcal S_2)>20,\\[2pt]
\text{按 }\pi_2\text{ 更新并重新选点},
&z=\mathrm{no\_signal},\\[2pt]
\text{检查观测、角度环绕及数值容差},
&\displaystyle\int L_2\pi_1=0.
\end{cases}}
\tag{56}
\]

其中

\[
c^*\in\arg\min_c\max_{g\in\mathcal S_2}\|g-c\|.
\tag{57}
\]

后验为空或归一化常数为零表示新观测与既有硬约束冲突，不能将该频道误判为已清除。

### 9 蒙特卡洛验证与垂直布点比较

为量化所提策略相对“沿首次示向轴垂直移动”的优势，采用成对蒙特卡洛试验。每个样本按式（5）生成 \((G,R)\)，以首次观测似然式（7）加权或筛选，使样本服从 \(\pi_1(g,r\mid H_1)\)；第二次检测时保持同一 \(R\) 不变，并仅对新检测位置生成一次独立测向误差。该过程避免将样本错误地直接均匀撒在 \(K_1\) 中。

令

\[
L=\|q^*-S\|,\qquad
q_\perp^\pm=S\pm L n ,
\tag{58}
\]

则 \(q^*\) 与垂直布点 \(q_\perp^\pm\) 的移动距离相同，二者的时间成本可比。对两侧垂直点分别评价，取性能较好者作为保守对照：

\[
q_\perp
\in
\arg\min_{q\in\{q_\perp^+,q_\perp^-\}}\Psi_D(q).
\tag{59}
\]

对策略 \(s\in\{*,\perp\}\)，定义

\[
\overline D_s=\frac1N\sum_{k=1}^{N}D_s^{(k)},
\qquad
Q_{0.9,s}=\operatorname{Quantile}_{0.9}
\{D_s^{(1)},\ldots,D_s^{(N)}\},
\tag{60}
\]

\[
\widehat P_{\mathrm{rec},s}
=\frac1N\sum_{k=1}^{N}
\mathbf 1_{\{Z_s^{(k)}\ne\mathrm{no\_signal}\}},
\qquad
\widehat P_{40,s}
=\frac1N\sum_{k=1}^{N}
\mathbf 1_{\{D_s^{(k)}\le40\}}.
\tag{61}
\]

所提策略相对垂直布点的量化优势分别为

\[
\boxed{
\Gamma_D
=
\frac{\overline D_\perp-\overline D_*}
{\overline D_\perp}\times100\%,
\qquad
\Gamma_{0.9}
=
\frac{Q_{0.9,\perp}-Q_{0.9,*}}
{Q_{0.9,\perp}}\times100\%,
}
\tag{62}
\]

\[
\boxed{
\Delta P_{\mathrm{rec}}
=
\widehat P_{\mathrm{rec},*}
-\widehat P_{\mathrm{rec},\perp},
\qquad
\Delta P_{40}
=
\widehat P_{40,*}
-\widehat P_{40,\perp}.
}
\tag{63}
\]

为排除“优势仅来自垂直点无信号较多”的解释，还应在两种策略均收到 \(\mathrm{direction}\) 的配对样本集合

\[
\mathcal I_{\mathrm{both}}
=
\{k:Z_*^{(k)}=\mathrm{direction},
\ Z_\perp^{(k)}=\mathrm{direction}\}
\tag{64}
\]

上重新计算 \(\Gamma_D^{\mathrm{both}}\)。若

\[
\Gamma_D>0,\quad
\Delta P_{\mathrm{rec}}>0,\quad
\Gamma_D^{\mathrm{both}}>0,
\tag{65}
\]

则可分别从总体定位效果、接收可靠性和纯交会几何三个层面说明所提策略优于等距离垂直布点。另以式（44）检验距离可信区间覆盖率，并对成对差值

\[
\Delta D_k=D_\perp^{(k)}-D_*^{(k)}
\tag{66}
\]

进行 Bootstrap，若其均值的 \(95\%\) 置信区间整体大于零，则优势具有统计稳定性。具体百分比应由最终参数、实际首次检测位置及独立仿真结果计算，不沿用基于“\(K_1\) 内均匀后验”简化所得的数值。

### 10 模型结论

本问以问题一的定位区域为几何基础，将源位置与固定接收半径组成联合隐状态，通过首次示向结果形成联合后验；随后分别构造保证接收区域 \(Q_{\mathrm g}\) 与高概率接收区域 \(Q_\eta\)，并以期望定位直径、期望面积和移动检测时间构成字典序目标。第二次观测按照 \(\mathrm{near}\)、\(\mathrm{direction}\) 和 \(\mathrm{no\_signal}\) 分段更新，距离后验则通过可信区间覆盖率进行校准检验。

该模型的核心不是预设“始终垂直于首次示向方向移动”，而是在式（31）的可接收几何约束下，同时配置纵向推进量 \(a\) 与横向基线 \(b\)，再由式（52）确定最优比例。由式（62）—（66）可在相同移动成本下量化其相对垂直布点在平均直径、尾部风险、接收率及可清除概率方面的优势。

### P.S. 建模衔接说明（不纳入正式论文）

以下内容仅用于课题组内部统一问题二与后续问题的变量、状态和程序接口，提交竞赛论文时应删除，不属于问题二的正式论证，也不作为正文结论或算法步骤参与评审。

问题二向后续模型输出

\[
\mathcal I_2=
\left(
\pi_2,\ \mathcal S_2,\ K_2,
\mathcal A(K_2),\ D(K_2),\ r_*(K_2),\ c^*
\right).
\tag{67}
\]

其中，\(\pi_2\) 用于后续检测点的概率排序，\(\mathcal S_2\) 保存全部精确硬约束，\(K_2\) 是可靠几何计算所需的闭合外包，\(r_*(K_2)\le20\) 可作为“允许尝试保证清除”的状态条件。后续全局模型只负责在多个频道之间安排移动、检测和清除顺序，不得把高后验概率误写成已经保证清除，也不得在新观测后把 \(\pi_2\) 重置为 \(K_2\) 上的均匀分布。
