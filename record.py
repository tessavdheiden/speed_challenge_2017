from collections import namedtuple


Record = namedtuple('TrainingRecord', ['ep', 'losses'])


class EvalRecords(object):
    def __init__(self):
        self.buffer = []

    def add(self, r):
        self.buffer.append(r)

    def get_losses(self):
        return [r.ep for r in self.buffer], [r.losses.data for r in self.buffer]

    def get_mse(self):
        losses = [r.losses for r in self.buffer]
        return sum(losses) / len(losses)


class TestRecords(object):
    def __init__(self):
        self.buffer = []

    def add(self, r):
        self.buffer.append(r)

    def get_pred(self):
        return [r for r in self.buffer]
