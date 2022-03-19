import numpy as np
import logging

logger = logging.getLogger(__name__)

class NominalRange:
    """
    This class specifies the range for a nominal attribute. It contains a list of integers which
    represent the values which are in the range.
    
    e.g. possible_values = [5,2] 
    """

    def __init__(self, possible_values, null_value=None, is_not_null_condition=False):
        self.is_not_null_condition = is_not_null_condition
        self.possible_values = np.array(possible_values, dtype=np.int64)
        self.null_value = null_value

    def is_impossible(self):
        return len(self.possible_values) == 0

    def get_ranges(self):
        return self.possible_values

    def in_range(self, val):
        return np.isin(self.possible_values, val).any()

class NumericRange:
    """
    This class specifies the range for a numeric attribute. It contains a list of intervals which
    represents the values which are valid. Inclusive Intervals specifies whether upper and lower bound are included.
    
    e.g. ranges = [[10,15],[22,23]] if valid values are between 10 and 15 plus 22 and 23 (bounds inclusive)
    """

    def __init__(self, ranges, inclusive_intervals=None, null_value=None, is_not_null_condition=False):
        self.is_not_null_condition = is_not_null_condition
        self.ranges = ranges
        self.null_value = null_value
        self.inclusive_intervals = inclusive_intervals
        if self.inclusive_intervals is None:
            self.inclusive_intervals = []
            for interval in self.ranges:
                self.inclusive_intervals.append([True, True])

    def is_impossible(self):
        return len(self.ranges) == 0

    def get_ranges(self):
        return self.ranges

    def in_range(self, val):
        def in_r(r, inc, val):
            if inc[0]: 
                if inc[1]:
                    return val >= r[0] and val <= r[1]
                else:
                    return val >= r[0] and val < r[1]
            else:
                if inc[1]:
                    return val > r[0] and val <= r[1]
                else:
                    return val > r[0] and val < r[1]

        return np.array([in_r(r, self.inclusive_intervals[i], val) for i, r in enumerate(self.ranges)], dtype=bool).any()
        
