import itertools
import random
import bisect
import math


class Utils:

    class Random:

        @staticmethod
        def choice(items, weights):
            cumdist = list(itertools.accumulate(weights))
            x = random.random() * cumdist[-1]
            return items[bisect.bisect(cumdist, x)]


