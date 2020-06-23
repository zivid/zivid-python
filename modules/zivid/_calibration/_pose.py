import _zivid
import numpy as np


class Pose:
    def __init__(self, transformation_matrix):
        self.__impl = _zivid.calibration.Pose(transformation_matrix)

    def to_matrix(self):
        return self.__impl.to_matrix()

    def __str__(self):
        return str(self.__impl)
