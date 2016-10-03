class QObject(object):

    def __init__(self, q_value, n):
        self.q_value = q_value
        self.n = n

    def __repr__(self):
        return 'q_value: {}, n: {}'.format(self.q_value, self.n)

    def __str__(self):
        return 'q_value: {}, n: {}'.format(self.q_value, self.n)