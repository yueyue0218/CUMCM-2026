"""对照组：上一版 Q4 路线优化算法，消融信息感知选点与可见性筛选。"""
from src.q4.refined import RefinedController


class ControlRefinedController(RefinedController):
    """Same Q4 physics and safeguards, before the final sensing improvement."""
