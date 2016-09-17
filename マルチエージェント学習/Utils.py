import random
import itertools
import bisect


class Utils:

    class Random:

        @staticmethod
        def choice(items, weights):
            """
            リストの中から重み付きランダムで選択する
            :param items:
            :param weights:
            :return:
            """
            cumdist = list(itertools.accumulate(weights))
            x = random.random() * cumdist[-1]
            return items[bisect.bisect(cumdist, x)]