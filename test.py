import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# local imports
from agent import Agent
from env import Env
from record import Record, TestRecords


parser = argparse.ArgumentParser(description='Test agent to detect speed in video')
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--model_name', type=str, default='cnn_net_params')
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--batch_size", type=int, default=1)
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
    env.capacity = 10798
    env.load_video(video_path='data/test.mp4')
    env.prep_train()    # we don't want to split
    criterion = nn.MSELoss()

    test_records = TestRecords()

    for i in range(env.capacity - env.n_imgs):
        state = env.get_img(i)
        torch_state = Variable(torch.from_numpy(state)).view((1, )+state.shape)
        # forward
        outputs = agent.predict(torch_state)

        pred = outputs.detach().numpy()[0][0] * env.norm_const
        print(f'Step {i} \t Prediction: {pred:.2f}')
        test_records.add(pred)

    pred = test_records.get_pred()
    first_elem = pred[0]
    for _ in range(args.n_imgs):
        pred = np.insert(pred, 0, first_elem, axis=0)
    print(pred.shape[0])
    np.savetxt('data/test.txt', pred, fmt='%1.6f')