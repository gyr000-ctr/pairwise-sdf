"""sdf 核心 API（初始未实现）。"""


class Sdf:
    def circle(self, p, center, radius):
        """圆距离"""
        raise NotImplementedError

    def rect(self, p, rect):
        """矩形距离"""
        raise NotImplementedError

    def segment(self, p, a, b):
        """线段距离"""
        raise NotImplementedError
