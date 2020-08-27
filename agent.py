from network import MLPNetwork
import torch
import torch.optim as optim


class Agent(object):
    def __init__(self):
        self.model = MLPNetwork()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def load_param(self, path):
        pass

    def predict(self, state):
        return self.model(state)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def prep_train(self, device='cpu'):
        self.model.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        self.model = fn(self.model)

    def prep_eval(self, device='cpu'):
        self.model.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        self.model = fn(self.model)

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_train(device='cpu')            # move parameters to CPU before saving
        torch.save(self.model.state_dict(), filename)

    def init_from_save(self, filename):
        """
        Instantiate instance from file created by 'save' method
        """
        return self.model.load_params(torch.load(filename))
