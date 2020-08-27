import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable


# local imports
from agent import Agent
from env import Env
from preprocess import pre_process
from record import Record, TestRecords


parser = argparse.ArgumentParser(description='Test agent to detect speed in video')
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--model_name', type=str, default='cnn_net_params')
parser.add_argument("--device", type=str, default='cpu')
args = parser.parse_args()


if __name__ == "__main__":
    agent = Agent()
    agent.load_param(path='param/cnn_net_params.pkl')
    agent.prep_eval()

    env = Env()
    env.load_video(video_path='data/train.mp4', data_path='data/train.txt')
    env.prep_eval()
    criterion = nn.MSELoss()

    test_records = TestRecords()
    running_score = 0

    for i_ep in range(10):
        score = 0
        state, labels = env.get_data()
        torch_state = Variable(torch.from_numpy(state))
        torch_labels = Variable(torch.from_numpy(labels))

        # forward
        outputs = agent.predict(torch_state)
        loss = criterion(outputs, torch_labels)

        test_records.add(Record(i_ep, loss))

        print('Step {}\t Test Loss: {:.2f}'.format(
            i_ep, loss * env.norm_const))

