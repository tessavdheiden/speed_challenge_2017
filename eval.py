import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# local imports
from agent import Agent
from env import Env
from record import Record, EvalRecords


parser = argparse.ArgumentParser(description='Test agent to detect speed in video')
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--model_name', type=str, default='cnn_net_params')
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_imgs", type=int, default=16)
parser.add_argument("--seed", default=1, type=int, help="Random seed")
args = parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = Agent(args.n_imgs)
    agent.init_from_save(filename=f'{args.output_dir}/{args.model_name}.pkl')
    agent.prep_eval()

    env = Env(args.n_imgs)
    env.batch_size = args.batch_size
    env.load_labels(data_path='data/train.txt')
    env.load_video(video_path='data/train.mp4')
    env.shuffle_data()
    env.prep_eval()
    criterion = nn.MSELoss()

    test_records = EvalRecords()

    for i_ep in range(1000):
        state, labels = env.get_data()
        torch_state = Variable(torch.from_numpy(state))
        torch_labels = Variable(torch.from_numpy(labels))

        # forward
        outputs = agent.predict(torch_state)
        loss = criterion(outputs, torch_labels)

        pred = outputs.detach().numpy()[0][0] * env.norm_const
        lab = torch_labels.detach().numpy()[0][0] * env.norm_const
        test_records.add(Record(i_ep, (pred-lab)**2))
        print(f'Step {i_ep} \t Prediction: {pred:.2f} \t Label: {lab:.2f}  Test Loss: {(pred-lab)**2:.2f}')

    print(f'Final MSE: {test_records.get_mse()}')