from collections import namedtuple


Record = namedtuple('TrainingRecord', ['ep', 'losses'])


class TestRecords(object):
    def __init__(self):
        self.buffer = []

    def add(self, r):
        self.buffer.append(r)

    def get_losses(self):
        return [r.ep for r in self.buffer], [r.losses.data for r in self.buffer]

    def get_mse(self):
        losses = [r.losses.data for r in self.buffer]
        return sum(losses) / len(losses)
