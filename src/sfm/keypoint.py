class Keypoint:
    """Minimal implementation of a keypoint used for feature matching."""

    def __init__(self, x, y, des=None):
        self.x = x
        self.y = y
        self._des = des

    @property
    def pt(self):
        return (self.x, self.y)

    @pt.setter
    def pt(self, pt):
        self.x, self.y = pt

    @property
    def des(self):
        if self._des is None:
            raise ValueError("Descriptor not set")
        return self._des

    @des.setter
    def des(self, des):
        self._des = des

    def __repr__(self):
        return f"Keypoint(x={self.x}, y={self.y})"

    def __str__(self):
        return f"Keypoint(x={self.x}, y={self.y})"
