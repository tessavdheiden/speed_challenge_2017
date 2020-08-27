import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

# local imports
from agent import Agent
from env import Env
from record import Record, TestRecords


parser = argparse.ArgumentParser(description='Test agent to detect speed in video')
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--model_name', type=str, default='cnn_net_params')
parser.add_argument("--n_episodes", default=500, type=int)
parser.add_argument("--log_interval", default=10, type=int)
parser.add_argument("--eval_interval", default=50, type=int)
parser.add_argument("--device", type=str, default='cpu')
args = parser.parse_args()


if __name__ == "__main__":
    agent = Agent()
    agent.prep_train()

    env = Env()
    env.load_video(video_path='data/train.mp4', data_path='data/train.txt')
    env.prep_train()
    criterion = nn.MSELoss()

    test_records = TestRecords()

    if args.device == 'gpu':
        fn = lambda x: x.cuda()
    else:
        fn = lambda x: x.cpu()

    for i_ep in range(args.n_episodes):
        score = 0
        state, labels = env.get_data()
        torch_state = fn(Variable(torch.from_numpy(state)))
        torch_labels = fn(Variable(torch.from_numpy(labels)))

        # forward
        outputs = agent.predict(torch_state)
        loss = criterion(outputs, torch_labels)
        agent.update(loss)

        if i_ep % args.log_interval == 0:
            print('Step {}\t train loss: {:.4f}'.format(i_ep, loss * env.norm_const))

        if i_ep % args.eval_interval == 0:
            agent.prep_eval()
            env.prep_eval()
            state, labels = env.get_data()
            torch_state = Variable(torch.from_numpy(state))
            outputs = agent.predict(torch_state)
            loss = criterion(outputs, torch_labels)
            print('Step {}\t evalution loss: {:.4f}'.format(i_ep, loss * env.norm_const))
            test_records.add(Record(i_ep, loss))

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            agent.save(filename=f'{args.output_dir}/{args.model_name}.pkl')
            agent.prep_train()
            env.prep_train()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    eps, losses = test_records.get_losses()
    plt.plot(eps, losses)
    plt.title('Loss')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig(f"{args.output_dir}/train_loss.png")



        # if test_records.get_mse() * env.norm_const < 3:
        #     print("Solved! Loss is now {}!".format(test_records.get_mse()))
        #     break