## 问题二：基于联合后验与期望定位直径的第二检测点选择

### 1 问题分析

设机器狗在首次检测点 \(S=(x_S,y_S)\) 对某一已确认存在的全向干扰源获得示向度 \(\theta\)。一次示向观测只能将源位置限制在狭长楔形区域内，不能确定其距离。若第二检测点与首次示向轴近似共线，则两次方位约束交角过小，定位区域直径仍然较大；若仅沿示向轴的垂直方向移动，则又可能远离楔形远端，导致无法接收信号。因此，本问归结为如下贝叶斯实验设计问题：

\[
\boxed{\text{在保证或高概率接收的候选区域内，选择使第二次检测后期望定位直径最小的检测点。}}
\tag{1}
\]

模型采用问题一建立的“精确可行集—闭合凸外包”双层表示。精确可行集用于贝叶斯更新和面积计算；闭合凸外包仅用于不改变其数值的直径、最小覆盖圆半径计算，以及接收保证的保守认证。题设只给出测向误差硬界而未给出其概率密度，故先以一般密度 \(f_\varepsilon\) 推导；进行平均意义下的策略比较时，再补充

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

并以 \(\operatorname{wrap}(\alpha)\in[-\pi,\pi)\) 表示环形角差。对全向干扰源，首次检测响应应先按距离分类：

\[
Y_1=
\begin{cases}
\mathrm{near},&d_S(g)\le5,\\
\mathrm{direction}(\theta),&5<d_S(g)\le r,\\
\mathrm{no\_signal},&d_S(g)>r.
\end{cases}
\]

若 \(Y_1=\mathrm{near}\)，则保留 \(\mathcal B(S,5)\) 内的全部可能位置，并直接在 \(S\) 处执行光学定位和清除，不再选择第二检测点。本问所研究的情形是 \(Y_1=\mathrm{direction}(\theta)\)，故条件事件本身蕴含 \(d_S(g)>5\)，其观测似然为

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

其中，\(d_S(g)>5\) 仅是收到 \(\mathrm{direction}\) 后的条件约束，并不表示 \(5\,\mathrm m\) 内为无源盲区。若首次响应为 \(\mathrm{near}\)，应改用 \(\Omega\cap\mathcal B(S,5)\) 作为位置支持集并直接进入清除流程；只有在已观测到示向度的当前分支中，才将 \(5\,\mathrm m\) 内区域从 \(\mathcal F_1\) 排除。

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

#### 4.1 极端稳健接收域

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

式（28）把无限约束化为有限圆盘交，且保持“必能接收”的保证。然而，该集合同时按最短接收半径覆盖全部可能源位置，通常明显压缩选点自由度，并可能带来较大的移动时间。因此，\(\widehat Q_{\mathrm g}\) 仅作为极端稳健方案和接收风险校验基准，不再作为主策略的硬候选域。

#### 4.2 时间可达域的局部表达

令

\[
u=(\cos\theta,\sin\theta)^{\mathsf T},\qquad
n=(-\sin\theta,\cos\theta)^{\mathsf T},\qquad
q=S+a u+b n .
\tag{29}
\]

为避免混淆全局位置与局部位移，定义正交基矩阵及局部决策变量

\[
B_\theta=[\,u\ \ n\,],\qquad
\xi=(a,b)^{\mathsf T},\qquad
q(\xi)=S+B_\theta\xi .
\]

由于 \(B_\theta^{\mathsf T}B_\theta=I_2\)，有

\[
L(q)=\|q-S\|=\|B_\theta\xi\|
=\|\xi\|=\sqrt{a^2+b^2}.
\]

后文中的优化变量统一取 \(\xi\in\mathbb R^2\)，而似然、后验及几何集合仍以对应的全局检测位置 \(q(\xi)\) 表示；两种写法通过 \(q=S+B_\theta\xi\) 一一对应。

当本次移动与检测的可用时间为 \(T_{\max}>5\) 时，允许的最大移动距离为

\[
R_T=5(T_{\max}-5).
\tag{30}
\]

因此，时间可达域在局部坐标下可直接写为

\[
\boxed{
M(T_{\max})
=
\left\{S+a u+b n:
a^2+b^2\le R_T^2
\right\}.
}
\tag{31}
\]

式（31）只刻画物理可达性，不再强制检测点同时位于三个半径 \(1000\,\mathrm m\) 的圆盘内。纵向位移 \(a\) 与横向位移 \(b\) 的比例由后验接收概率和综合目标共同确定。

#### 4.3 高概率接收区域与时间约束

根据联合后验定义第二检测点的接收概率

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

给定名义接收置信水平 \(\eta_0\)，为避免固定阈值导致候选域为空，定义有效阈值

\[
\eta_{\mathrm{eff}}
=
\min\left\{
\eta_0,
\max_{q\in M(T_{\max})}\rho(q)
\right\}.
\tag{34}
\]

最终候选区域取为

\[
\boxed{
\mathcal C
=
\left\{
q\in M(T_{\max}):
\rho(q)\ge\eta_{\mathrm{eff}}
\right\}.
}
\tag{35}
\]

若 \(Q_{\eta_0}\cap M(T_{\max})\ne\varnothing\)，则 \(\eta_{\mathrm{eff}}=\eta_0\)；否则阈值自动降至时间可达域内的最大接收概率，只保留接收概率最高的可达点。由此，候选域同时反映时间约束和接收可靠性，而 \(\widehat Q_{\mathrm g}\cap\mathcal C\) 仅作为其中具有绝对接收保证的可选子集。

将全局候选域映射到局部决策空间，记

\[
\Xi=
\left\{
\xi\in\mathbb R^2:q(\xi)\in\mathcal C
\right\}.
\]

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

其中，\(\mathrm{direction}\) 情形的 \(5\,\mathrm m\) 内排除及 \(\mathrm{no\_signal}\) 情形的圆盘排除均单独保存。特别地，

\[
E_2^{\mathrm{no}}(q)
=K_1\setminus\mathcal B^\circ(q,1000)
\]

是 \(\mathcal S_2(\mathrm{no\_signal},q)\) 的非凸闭合外包。贝叶斯更新仍以式（38）—（39）的精确支持集和后验密度为准；需要保守几何认证时才使用 \(E_2^{\mathrm{no}}(q)\)，二者均不以凸包代替面积。若直径或最小覆盖圆算法要求凸输入，则另取闭合凸包

\[
K_{2,\mathrm c}^{\mathrm{no}}(q)
=\overline{\operatorname{conv}}
\left(E_2^{\mathrm{no}}(q)\right).
\]

对任意非空紧集 \(E\)，均有

\[
D(E)=D\!\left(\overline{\operatorname{conv}}E\right),
\qquad
r_*(E)=r_*\!\left(\overline{\operatorname{conv}}E\right),
\]

故对同一集合 \(E\) 而言，凸包化不会改变直径与最小覆盖圆半径：计算精确后验指标时取 \(E=\mathcal S_2\)，进行保守外包认证时取 \(E=E_2^{\mathrm{no}}\)。但一般有
\(\mathcal A(E)\ne
\mathcal A(\overline{\operatorname{conv}}E)\)，因此面积指标必须由相应的原非凸集合计算。若 \(q\in Q_{\mathrm g}\)，则 \(\mathrm{no\_signal}\) 的预测概率为零。

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

第二次检测结果尚未发生时，\(Z_q\) 同时包含离散响应类型与连续示向角。为避免混用概率质量与概率密度，对方向响应定义预测次密度

\[
\begin{aligned}
\lambda_{\mathrm{dir}}(z\mid H_1,q)
&:=
\int_{\Omega}\int_{1000}^{1500}
L_2(z\mid g,r,q)\pi_1(g,r\mid H_1)
\,\mathrm dr\,\mathrm dg\\
&=
\Pr(\mathrm{direction}\mid H_1,q)\,
p(z\mid\mathrm{direction},H_1,q),
\qquad z\in[-\pi,\pi),\\
\int_{-\pi}^{\pi}\lambda_{\mathrm{dir}}(z\mid H_1,q)\,\mathrm dz
&=\Pr(\mathrm{direction}\mid H_1,q).
\end{aligned}
\tag{48}
\]

式（48）的乘积分解仅在 \(\Pr(\mathrm{direction}\mid H_1,q)>0\) 时使用；若该概率为零，则约定 \(\lambda_{\mathrm{dir}}(z\mid H_1,q)\equiv0\)，无须定义相应的条件角度密度。

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
\lambda_{\mathrm{dir}}(z\mid H_1,q)\,\mathrm dz .
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

式（49）与式（50）的方向响应分支通常难以解析积分。数值求解时，将角度区间等分为 \(M\) 段，令

\[
\Delta z=\frac{2\pi}{M},\qquad
z_\ell=-\pi+\left(\ell+\frac12\right)\Delta z,\qquad
\int_{-\pi}^{\pi}\phi(z)\lambda_{\mathrm{dir}}(z\mid H_1,q)\,\mathrm dz
\approx
\Delta z\sum_{\ell=0}^{M-1}
\phi(z_\ell)\lambda_{\mathrm{dir}}(z_\ell\mid H_1,q),
\]

其中 \(\phi(z)\) 分别取 \(D(K_2^{\mathrm{dir}}(z,q))\) 与 \(\mathcal A(K_2^{\mathrm{dir}}(z,q))\)。采用步长逐级折半的复合中点求积，并以相邻两级的目标函数相对差不超过 \(\tau_{\mathrm{int}}\)、最优局部坐标差不超过 \(\tau_\xi\) 作为积分离散化停止准则；同时校验三类响应的预测概率之和为 \(1\)。

单次移动检测时间定义为

\[
T(q)=\frac{\|q-S\|}{5}+5.
\tag{51}
\]

取尺度量

\[
D_0=D(K_1),\qquad
A_0=\mathcal A(K_1),\qquad
T_0=T_{\max}.
\]

当 \(A_0>0\) 时，将定位直径、区域面积与移动检测时间无量纲化，并构造综合损失

\[
\boxed{
J_{\boldsymbol\omega}(\xi)
=
\omega_D\frac{\Psi_D(q(\xi))}{D_0}
+\omega_A\frac{\Psi_A(q(\xi))}{A_0}
+\omega_T\frac{T(q(\xi))}{T_0},
\qquad
\xi^*\in\arg\min_{\xi\in\Xi}J_{\boldsymbol\omega}(\xi),
\qquad
q^*=q(\xi^*).
}
\tag{52}
\]

其中仅对预测概率为正的观测结果计入期望，并约定空集的面积与直径均为零。权重满足

\[
\boxed{
\omega_D,\omega_A,\omega_T\ge0,\qquad
\omega_D+\omega_A+\omega_T=1,\qquad
\omega_T>0.
}
\tag{53}
\]

由于 \(\omega_T>0\)，任意额外移动都会产生显式损失，时间项不会像字典序目标那样退化失效。若初始区域退化为零面积集合，则删除面积项并对其余权重重新归一化。

#### 7.1 鲁棒不确定集合

式（52）刻画标称概率模型下的平均性能，但其最优解未必能承受误差边界或接收距离的轻微失配。为此，引入位置、测角、接收距离和时间计算的非负安全裕量
\(\eta_g,\eta_\theta,\eta_r,\eta_T\)，并定义

\[
\boxed{
\mathcal S_1^{\mathrm{rob}}
=
\left(\mathcal S_1\oplus\mathcal B(0,\eta_g)\right)\cap\Omega,
\qquad
\delta_{\mathrm{rob}}=\delta+\eta_\theta,
\qquad
[r_{\min},r_{\max}]
=
[1000-\eta_r,\ 1500+\eta_r].
}
\tag{54}
\]

其中要求 \(r_{\min}>5\)，\(\oplus\) 表示 Minkowski 和。用
\(\mathcal S_1^{\mathrm{rob}}\)、\(\delta_{\mathrm{rob}}\) 和
\([r_{\min},r_{\max}]\) 替换式（39）—（40）中的相应集合与参数，分别得到鲁棒支持集
\(\mathcal S_2^{\mathrm{rob}}(z,q)\) 及其闭合外包
\(K_2^{\mathrm{rob}}(z,q)\)。若真实误差与接收距离均落在式（54）的扩大边界内，则

\[
\boxed{
\mathcal S_2^{\mathrm{true}}(z,q)
\subseteq
\mathcal S_2^{\mathrm{rob}}(z,q)
\subseteq
K_2^{\mathrm{rob}}(z,q).
}
\tag{55}
\]

#### 7.2 鲁棒可行域与最坏性能

定义带时间裕量的安全可行域和保证接收域

\[
\boxed{
\begin{aligned}
\Xi_{\mathrm{safe}}
&=
\left\{
\xi\in\Xi:
T(q(\xi))\le T_{\max}-\eta_T
\right\},\\
\Xi_{\mathrm{rec}}
&=
\left\{
\xi\in\Xi_{\mathrm{safe}}:
\sup_{g\in\mathcal S_1^{\mathrm{rob}}}
\|q(\xi)-g\|\le r_{\min}
\right\}.
\end{aligned}}
\tag{56}
\]

若任务要求第二次检测必然收到 \(\mathrm{near}\) 或
\(\mathrm{direction}\)，则取 \(\Xi_{\mathrm{adm}}=\Xi_{\mathrm{rec}}\)；
否则取 \(\Xi_{\mathrm{adm}}=\Xi_{\mathrm{safe}}\)，并将
\(\mathrm{no\_signal}\) 保留在最坏情况分析中。特别地，
\(\Xi_{\mathrm{rec}}=\varnothing\) 表明在当前时间和误差边界下不存在保证接收的策略，此时不得宣称绝对接收保证。

第二次检测的混合响应空间及鲁棒有效响应集合为

\[
\boxed{
\mathcal Y=
\{\mathrm{near},\mathrm{no\_signal}\}
\sqcup[-\pi,\pi),
\qquad
\mathcal Z_{\mathrm{rob}}(q)
=
\left\{
z\in\mathcal Y:
\mathcal S_2^{\mathrm{rob}}(z,q)\ne\varnothing
\right\}.
}
\tag{57}
\]

其中角度 \(z\in[-\pi,\pi)\) 表示 \(\mathrm{direction}(z)\)，符号
\(\sqcup\) 表示离散响应与连续角度的互不相交并。对任意
\(\xi\in\Xi_{\mathrm{safe}}\)，定义

\[
\boxed{
\begin{aligned}
\overline D(\xi)
&=
\sup_{z\in\mathcal Z_{\mathrm{rob}}(q(\xi))}
D\!\left(K_2^{\mathrm{rob}}(z,q(\xi))\right),\\
\overline A(\xi)
&=
\sup_{z\in\mathcal Z_{\mathrm{rob}}(q(\xi))}
\mathcal A\!\left(K_2^{\mathrm{rob}}(z,q(\xi))\right),\\
\overline r(\xi)
&=
\sup_{z\in\mathcal Z_{\mathrm{rob}}(q(\xi))}
r_*\!\left(K_2^{\mathrm{rob}}(z,q(\xi))\right).
\end{aligned}}
\tag{58}
\]

#### 7.3 带最坏约束的选点模型

当 \(f_\varepsilon\) 可信时，推荐在标称贝叶斯目标外加入最坏性能上界：

\[
\boxed{
\begin{aligned}
\xi_{\mathrm{rob}}^*
\in\arg\min_{\xi\in\Xi_{\mathrm{adm}}}\quad
&J_{\boldsymbol\omega}(\xi),\\
\mathrm{s.t.}\quad
&\overline D(\xi)\le D_{\lim},\\
&\overline A(\xi)\le A_{\lim},
\qquad
q_{\mathrm{rob}}^*=q(\xi_{\mathrm{rob}}^*).
\end{aligned}}
\tag{59}
\]

其中，\(D_{\lim}\) 与 \(A_{\lim}\) 分别为任务允许的最坏定位直径和面积上界；若题设未给出固定阈值，则通过可行性扫描与鲁棒 Pareto 前沿给出二者的可达范围，再由剩余任务时间确定取值。

若任务要求第二次检测后必然具备一次光学覆盖条件，则再加入

\[
\boxed{\overline r(\xi)\le20.}
\tag{60}
\]

若只知道误差硬界而没有可信概率密度，则式（49）的期望不可辨识，改用纯极小极大模型

\[
\boxed{
\xi_{\mathrm{mm}}^*
\in
\arg\min_{\xi\in\Xi_{\mathrm{adm}}}
\left[
\omega_D\frac{\overline D(\xi)}{D_0}
+\omega_A\frac{\overline A(\xi)}{A_0}
+\omega_T\frac{T(q(\xi))}{T_0}
\right],
\qquad
q_{\mathrm{mm}}^*=q(\xi_{\mathrm{mm}}^*).
}
\tag{61}
\]

式（59）优先保证标称平均性能，式（61）则完全由不确定集合决定。由式（55）及集合指标的单调性，对任一实际响应 \(z\) 均有

\[
\boxed{
\begin{aligned}
D\!\left(\mathcal S_2^{\mathrm{true}}(z,q)\right)
&\le\overline D(\xi),\\
\mathcal A\!\left(\mathcal S_2^{\mathrm{true}}(z,q)\right)
&\le\overline A(\xi),\\
r_*\!\left(\mathcal S_2^{\mathrm{true}}(z,q)\right)
&\le\overline r(\xi).
\end{aligned}}
\tag{62}
\]

因此，式（59）—（60）的约束一旦成立，即可分别给出定位直径、区域面积和一次覆盖半径的确定性上界，而不依赖蒙特卡洛样本是否覆盖到最不利状态。

#### 7.4 最坏情况的数值认证

有限角度网格上的最大值一般只是方向响应分支最坏损失的下界，不能直接作为鲁棒保证。令
\(\mathcal L_{\mathrm{rob}}(z,\xi)\) 表示直径、面积、覆盖半径或式（61）中的加权损失。若其关于连续示向角 \(z\) 的 Lipschitz 常数为
\(L_z(\xi)\)，则对最大网格间距 \(\Delta z\) 有

\[
\boxed{
\sup_{\substack{
z\in[-\pi,\pi)\\
z\in\mathcal Z_{\mathrm{rob}}(q(\xi))
}}
\mathcal L_{\mathrm{rob}}(z,\xi)
\le
\max_\ell\mathcal L_{\mathrm{rob}}(z_\ell,\xi)
+\frac{L_z(\xi)\Delta z}{2}.
}
\tag{63}
\]

完整的最坏损失取式（63）右端与可行离散响应
\(\mathrm{near}\)、\(\mathrm{no\_signal}\) 损失的最大值。若无法给出可靠的 \(L_z(\xi)\)，则对角度区间采用自适应区间细分或分支定界，直接计算每个子区间的损失上界。记 \(J_{\mathrm{rob}}^*\) 为所选鲁棒模型的全局最优目标值，\(\widehat J_{\mathrm{rob}}\) 为经上界认证的可行解目标值；再记全局优化误差和角度离散误差分别为
\(\varepsilon_{\mathrm{opt}}\) 与 \(\varepsilon_{\mathrm{grid}}\)，最终应报告

\[
\boxed{
0\le
\widehat J_{\mathrm{rob}}-J_{\mathrm{rob}}^*
\le
\varepsilon_{\mathrm{opt}}+\varepsilon_{\mathrm{grid}}.
}
\tag{64}
\]

选点模式据此按条件分类为

\[
\boxed{
\text{选点准则}=
\begin{cases}
\text{贝叶斯目标与最坏约束联合优化，式（59）},
&f_\varepsilon\ \text{可信},\
\Xi_{\mathrm{adm}}\ne\varnothing,\\[2pt]
\text{几何最坏损失与时间成本联合最小化，式（61）},
&\text{仅已知误差硬界},\
\Xi_{\mathrm{adm}}\ne\varnothing,\\[2pt]
\text{退回 }\Xi_{\mathrm{safe}}\text{ 并显式保留无信号分支},
&\Xi_{\mathrm{rec}}=\varnothing.
\end{cases}}
\tag{65}
\]

后文以 \(q^*\) 统称按照式（65）选出的
\(q_{\mathrm{rob}}^*\) 或 \(q_{\mathrm{mm}}^*\)。

为降低主观权重对结论的影响，在给定权重集合 \(\mathcal W\) 上进行参数扫描，并保留非支配检测点构成 Pareto 集

\[
\mathcal P_{\mathrm{rob}}=
\left\{
q(\xi):\xi\in\Xi_{\mathrm{adm}},\
\nexists \xi'\in\Xi_{\mathrm{adm}},\
\begin{array}{l}
\overline D(\xi')\le\overline D(\xi),\
\overline A(\xi')\le\overline A(\xi),\
T(q(\xi'))\le T(q(\xi)),\\
\text{且至少一个不等式严格成立}
\end{array}
\right\}.
\]

若 \(q^*(\boldsymbol\omega)\) 在一段权重区间内保持不变或仅小幅移动，则所选策略对时间—精度偏好具有稳定性；否则报告鲁棒 Pareto 集，由任务剩余时间和允许的最坏损失共同决定最终选点。

### 8 第二次观测后的条件决策

第二次检测完成后，按下列条件执行：

\[
\boxed{
a^*(z)=
\begin{cases}
\text{在 }q\text{ 处定位并清除},
&z=\mathrm{near},\\[2pt]
\text{移动至鲁棒最小覆盖圆圆心 }c_{\mathrm{rob}}^*\text{ 后定位并清除},
&z=\mathrm{direction},\
r_*(K_2^{\mathrm{rob}}(z,q))\le20,\\[2pt]
\text{保留后验并继续测量},
&z=\mathrm{direction},\
r_*(K_2^{\mathrm{rob}}(z,q))>20,\\[2pt]
\text{按 }\pi_2\text{ 更新并重新选点},
&z=\mathrm{no\_signal},\\[2pt]
\text{检查观测并扩大鲁棒误差边界},
&\displaystyle\int L_2\pi_1=0.
\end{cases}}
\tag{66}
\]

其中

\[
c_{\mathrm{rob}}^*
\in
\arg\min_c
\max_{g\in K_2^{\mathrm{rob}}(z,q)}
\|g-c\|.
\tag{67}
\]

后验为空或归一化常数为零表示新观测与既有硬约束冲突，不能将该频道误判为已清除。此时应先检查观测、角度环绕和数值容差；若观测可信，则扩大式（54）的相应安全裕量并重新计算，而不是强行归一化空后验。

### 9 蒙特卡洛验证与垂直布点比较

为量化所提策略相对“沿首次示向轴垂直移动”的优势，采用成对蒙特卡洛试验。该试验只评价标称平均性能和有限样本下的经验稳定性，不能替代式（55）、式（62）—（64）给出的确定性鲁棒认证。每个样本按式（5）生成 \((G,R)\)，以首次观测似然式（7）加权或筛选，使样本服从 \(\pi_1(g,r\mid H_1)\)；第二次检测时保持同一 \(R\) 不变，并仅对新检测位置生成一次独立测向误差。该过程避免将样本错误地直接均匀撒在 \(K_1\) 中。

令

\[
L=\|q^*-S\|,\qquad
q_\perp^\pm=S\pm L n ,
\tag{68}
\]

则 \(q^*\) 与垂直布点 \(q_\perp^\pm\) 的移动距离相同，二者的时间成本可比。对两侧垂直点分别评价，取性能较好者作为保守对照：

\[
q_\perp
\in
\arg\min_{q\in\{q_\perp^+,q_\perp^-\}}\Psi_D(q).
\tag{69}
\]

对策略 \(s\in\{*,\perp\}\)，定义

\[
\overline D_s=\frac1N\sum_{k=1}^{N}D_s^{(k)},
\qquad
Q_{0.9,s}=\operatorname{Quantile}_{0.9}
\{D_s^{(1)},\ldots,D_s^{(N)}\},
\tag{70}
\]

\[
\widehat P_{\mathrm{rec},s}
=\frac1N\sum_{k=1}^{N}
\mathbf 1_{\{Z_s^{(k)}\ne\mathrm{no\_signal}\}},
\qquad
\widehat P_{40,s}
=\frac1N\sum_{k=1}^{N}
\mathbf 1_{\{D_s^{(k)}\le40\}}.
\tag{71}
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
\tag{72}
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
\tag{73}
\]

为排除“优势仅来自垂直点无信号较多”的解释，还应在两种策略均收到 \(\mathrm{direction}\) 的配对样本集合

\[
\mathcal I_{\mathrm{both}}
=
\{k:Z_*^{(k)}=\mathrm{direction},
\ Z_\perp^{(k)}=\mathrm{direction}\}
\tag{74}
\]

上重新计算 \(\Gamma_D^{\mathrm{both}}\)。若

\[
\Gamma_D>0,\quad
\Delta P_{\mathrm{rec}}>0,\quad
\Gamma_D^{\mathrm{both}}>0,
\tag{75}
\]

则可分别从总体定位效果、接收可靠性和纯交会几何三个层面说明所提策略优于等距离垂直布点。另以式（44）检验距离可信区间覆盖率，并对成对差值

\[
\Delta D_k=D_\perp^{(k)}-D_*^{(k)}
\tag{76}
\]

进行 Bootstrap，若其均值的 \(95\%\) 置信区间整体大于零，则优势具有统计稳定性。具体百分比应由最终参数、实际首次检测位置及独立仿真结果计算，不沿用基于“\(K_1\) 内均匀后验”简化所得的数值。

### 10 模型结论

本问以问题一的定位区域为几何基础，将源位置与固定接收半径组成联合隐状态，并在首次响应分类后形成联合后验；其中 \(\mathrm{near}\) 直接进入清除流程，只有 \(\mathrm{direction}\) 分支进入第二检测点选择。随后以时间可达域和后验接收概率构造标称候选区域，并以无量纲的期望定位直径、期望面积和移动检测时间构成综合损失。在此基础上，通过扩大位置、测角、接收距离和时间误差边界，进一步构造安全可行域与保证接收域，并对最坏定位直径、面积和最小覆盖圆半径施加确定性上界。第二次观测继续按照 \(\mathrm{near}\)、\(\mathrm{direction}\) 和 \(\mathrm{no\_signal}\) 分段更新，距离后验则通过可信区间覆盖率进行校准检验。

该模型的核心不是预设“始终垂直于首次示向方向移动”，而是在式（31）的时间可达域内同时配置纵向推进量 \(a\) 与横向基线 \(b\)。当概率模型可信时，由式（59）在贝叶斯平均性能与最坏情形约束之间取得平衡；仅有误差硬界时，则由式（61）直接最小化最坏损失。式（62）—（64）给出可审计的确定性与数值误差保证，鲁棒 Pareto 集用于检验时间—精度偏好的稳定性；式（72）—（76）只负责量化其相对垂直布点的经验优势。

### P.S. 建模衔接说明（不纳入正式论文）

以下内容仅用于课题组内部统一问题二与后续问题的变量、状态和程序接口，提交竞赛论文时应删除，不属于问题二的正式论证，也不作为正文结论或算法步骤参与评审。

问题二向后续模型输出

\[
\mathcal I_2=
\left(
\pi_2,\ \mathcal S_2,\ \mathcal S_2^{\mathrm{rob}},
\ K_2^{\mathrm{rob}},
\mathcal A(K_2^{\mathrm{rob}}),\
D(K_2^{\mathrm{rob}}),\
r_*(K_2^{\mathrm{rob}}),\
c_{\mathrm{rob}}^*
\right).
\tag{77}
\]

其中，\(\pi_2\) 用于后续检测点的概率排序，\(\mathcal S_2\) 保存标称精确约束，\(\mathcal S_2^{\mathrm{rob}}\) 与 \(K_2^{\mathrm{rob}}\) 分别保存扩大误差边界后的支持集及闭合外包；\(r_*(K_2^{\mathrm{rob}})\le20\) 才可作为“允许尝试保证清除”的鲁棒状态条件。后续全局模型只负责在多个频道之间安排移动、检测和清除顺序，不得把高后验概率误写成已经保证清除，也不得在新观测后把 \(\pi_2\) 重置为 \(K_2^{\mathrm{rob}}\) 上的均匀分布。
