import torch.nn as nn

# local imports
from agent import Agent
from env import Env

if __name__ == "__main__":
    agent = Agent()
    agent.load_param(path='param/cnn_net_params.pkl')
    env = Env()
    env.load_video(video_path='data/train.mp4', data_path='data/test.mp4')
    criterion = nn.MSELoss()

    test_records = []
    running_score = 0

    for i_ep in range(10):
        score = 0
        state, labels = env.get_data()

        # forward
        outputs = agent(state)
        loss = criterion(outputs, labels)

        running_score += loss
        test_records.append(Record(i_ep, running_score))

        if i_ep % args.log_interval == 0:
            print('Step {}\tAverage score: {:.2f}'.format(
                i_ep, running_score))

    if test_records.get_mse() < 3:
        print("Solved! Loss is now {}!".format(test_records.get_mse()))