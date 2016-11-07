class QObject(object):

    def __init__(self, q_value, n, e=0):
        self.q_value = q_value
        self.n = n
        self.e = e

    def __repr__(self):
        return 'q_value: {}, n: {}, e: {}'.format(self.q_value, self.n, self.e)

    def __str__(self):
        return 'q_value: {}, n: {}, e: {}'.format(self.q_value, self.n, self.e)