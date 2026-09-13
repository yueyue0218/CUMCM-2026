import os
os.environ["PATH"] += os.pathsep + r'D:\Graphviz\bin'
from graphviz import Digraph

dot = Digraph('ns_ppo_final', format='png')
dot.attr(
    rankdir='LR',
    splines='ortho',       # 强制横平竖直直角连线
    nodesep='0.45',
    ranksep='1.1',         # 拉大行间距离，防止线条缠绕
    dpi='300',
    fontname='Microsoft YaHei,SimHei,Arial'
)
dot.attr('node', fontname='Microsoft YaHei,SimHei,Arial', fontsize='9', margin='0.15,0.2')
dot.attr('edge', fontname='Microsoft YaHei,SimHei,Arial', fontsize='8')

# ========== 上侧主流程 ==========
dot.node('env',
'''复杂无线电环境
(World Model)
•多频段/多源
•时空耗时模型''',
shape='box', style='filled,rounded', fillcolor='#E1F5FE', color='#0288D1', penwidth='1.2')

dot.node('stateTable',
'''更新知识状态表
•P(有源) / C(候选)
•E(排除) / U(未定)''',
shape='box', style='filled,rounded', fillcolor='#FFF3E0', color='#F57C00', penwidth='1.2')

with dot.subgraph(name='cluster_tracker') as c_tracker:
    c_tracker.attr(label='三值逻辑追踪器 (Logic Tracker)', style='dashed', color='#F57C00', bgcolor='#FFF8E1')
    c_tracker.node('elim',
    '''下界消元与
八叉树几何排除
(Geometric Elim.)''',
    shape='box', style='filled,rounded', fillcolor='#FFF3E0', color='#F57C00')

dot.node('poolGen',
'''路线候选池生成器
(Candidate Pool Gen)
•全局近邻 TSP (r^TSP)
•局域高增益 (r^IG)
•快速消除 (r^EC)''',
shape='box', style='filled,rounded', fillcolor='#F3E5F5', color='#7B1FA2', penwidth='1.2')

with dot.subgraph(name='cluster_nsppo') as c_nsppo:
    c_nsppo.attr(label='NS-PPO 决策引擎', style='dashed', color='#388E3C', bgcolor='#E8F5E9')
    c_nsppo.node('masker', '''符号掩码
(Symbolic Masker)
屏蔽不可行路线/状态''',
                 shape='box', style='filled,rounded', fillcolor='#E8F5E9', color='#388E3C')
    c_nsppo.node('actor', '''PPO Actor 网络
路线选择策略''',
                 shape='box', style='filled,rounded', fillcolor='#E8F5E9', color='#388E3C')
    c_nsppo.node('critic', '''PPO Critic 网络
价值评估''',
                 shape='box', style='filled,rounded', fillcolor='#E8F5E9', color='#388E3C')

# ========== 右下分支组：横向排布 judgeDiamond → fallbackBox → exec ==========
with dot.subgraph() as sub_bottom:
    sub_bottom.attr(rank='same')
    sub_bottom.node('judgeDiamond', '计算/步数\n超限?',
        shape='diamond', style='filled', fillcolor='#FBE9E7', color='#D84315', penwidth='1.2')
    sub_bottom.node('fallbackBox', '''确定性解析接管
(Fallback Branch)
•双点交叉定位 & TSP''',
        shape='box', style='filled,rounded', fillcolor='#FBE9E7', color='#D84315')
    sub_bottom.node('exec', '''执行最优路线 r*
•移动 / 切频
•检测 / 清除''',
        shape='box', style='filled,rounded', fillcolor='#ECEFF1', color='#455A64', penwidth='1.2')
    # 不可见边强制横向顺序：菱形 → 接管 → 执行
    sub_bottom.edge('judgeDiamond', 'fallbackBox', style='invis')
    sub_bottom.edge('fallbackBox', 'exec', style='invis')

# ========== 主流程连线 ==========
dot.edge('env', 'stateTable')
dot.edge('stateTable', 'elim')
dot.edge('elim', 'poolGen', label='约束状态 s_t')
dot.edge('poolGen', 'masker', label='候选集 R_t')
dot.edge('masker', 'actor')
dot.edge('masker', 'critic', style='dashed')
dot.edge('actor', 'critic', style='dashed', label='更新')
dot.edge('stateTable', 'masker', label='状态特征', style='dashed')

# Actor向右，再向下拐弯连接到右下菱形
dot.edge('actor', 'judgeDiamond', label='动作选择 a_t = k*')

# 分支逻辑
dot.edge('judgeDiamond', 'fallbackBox', label='是')
dot.edge('judgeDiamond', 'exec', label='否')
dot.edge('fallbackBox', 'exec')

# 系统反馈，长连线回到最左侧环境
dot.edge('exec', 'env', label='反馈状态', constraint="false")

dot.render('ns_ppo_architecture_clean', view=False)
print("图片输出完成：ns_ppo_architecture_clean.png")
