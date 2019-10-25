import math
import copy
import sys
import time
import argparse
import initenv


parser = argparse.ArgumentParser('Train Agent in Environment:')
parser.add_argument('-framework', type=str, default='', help='name of training framework')
parser.add_argument('-env', type=str, default='', help='name of environment to use')
parser.add_argument('-env_params', type=str, default='', help='string of environment parameters')
parser.add_argument('-pool_frms', type=str, default='', help='string of frame pooling parameters (e.g.: size=2,type=\'max\')')
parser.add_argument('-actrep', type=int, default=1, help='how many times to repeat action')
parser.add_argument('-random_starts', type=int, default=0, help='play action 0 between 1 and random_starts number of times at \
    the start of each training episode')

parser.add_argument('-name', type=str, default='', help='filename used for saving network and training history')
parser.add_argument('-network', type=str, default='', help='reload pretrained network')
parser.add_argument('-agent', type=str, default='', help='name of agent file to use')
parser.add_argument('-agent_params', type=str, default='', help='string of agent parameters')
parser.add_argument('-seed', type=int, default=1, help='fixed input seed for repeatable experiments')
parser.add_argument('-saveNetworkParams', action='store_true', help='saves the agent network in a separate file')
parser.add_argument('-prog_freq', type=int, default=5e3, help='frequency of progress output')
parser.add_argument('-save_freq', type=int, default=5e4, help='the model is saved every save_freq steps')
parser.add_argument('-eval_freq', type=int, default=1e4, help='frequency of greedy evaluation')
parser.add_argument('-save_versions', type=int, default=0, help='')

parser.add_argument('-steps', type=int, default=1e5, help='number of training steps to perform')
parser.add_argument('-eval_steps', type=int, default=1e5, help='number of evaluation steps')

parser.add_argument('-verbose', type=int, default=2, help='the higher the level, the more information is printed to screen')
parser.add_argument('-threads', type=int, default=1, help='number of BLAS threads')
parser.add_argument('-gpu', type=int, default=-1, help='gpu flag')

parser.add_argument('-best', action='store_true', help='load the best performing checkpoint instead of the latest \
    (only has effect when loading a saved model)')

opt = parser.parse_args()
opt = vars(opt)


# general setup
game_env, agent, opt = initenv.setup(opt)

# override print to always flush the output
old_print = print
def print(*args, **kwargs):
    old_print(*args, **kwargs)
    sys.stdout.flush()

learn_start = agent.learn_start
start_time = time.time()
reward_counts = []
episode_counts = []
time_history = []
v_history = []
qmax_history = []
td_history = []
reward_history = []
step = 0
time_history.append(0)

screen = game_env.reset()
reward = 0
terminal = False

print('Iteration ..', step)
while step < opt['steps']:
    step += 1
    action_index = agent.perceive(reward, screen, terminal)

    # game over? get next game!
    if not terminal:
        screen, reward, terminal, _ = game_env.step(action_index, episodic_life=True)
    else:
        screen = game_env.reset()
        reward = 0
        terminal = False

    if step % opt['prog_freq'] == 0:
        assert step == agent.numSteps, 'trainer step: %d & agent.numSteps: %d' % (step, agent.numSteps)
        print('Steps:', step)
        agent.report()


    if step % opt['eval_freq'] == 0 and step > learn_start:
        screen = game_env.reset(noop_max=0)
        reward = 0
        terminal = False

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        eval_time = time.time()
        for estep in range(opt['eval_steps']):
            action_index = agent.perceive(reward=reward, rawstate=screen, terminal=terminal, testing=True, testing_ep=0.05)

            screen, reward, terminal, _ = game_env.step(action_index)
            game_env.render() # TODO: temporary...

            # record every reward
            episode_reward = episode_reward + reward
            if reward != 0:
               nrewards = nrewards + 1

            if terminal:
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                screen = game_env.reset()
                reward = 0
                terminal = False

        eval_time = time.time() - eval_time
        start_time = start_time + eval_time
        agent.compute_validation_statistics()
        ind = len(reward_history)
        total_reward = total_reward / max(1, nepisodes)

        if len(reward_history) == 0 or total_reward > max(reward_history):
            agent.best_network = copy.deepcopy(agent.network)

        if agent.v_avg:
            v_history.append(agent.v_avg)
            td_history.append(agent.tderr_avg)
            qmax_history.append(agent.q_max)
        print('V', v_history[ind], 'TD error', td_history[ind], 'Qmax', qmax_history[ind])

        reward_history.append(total_reward)
        reward_counts.append(nrewards)
        episode_counts.append(nepisodes)

        time_history.append(time.time() - start_time)

        time_dif = time_history[ind + 1] - time_history[ind]

        training_rate = opt['actrep'] * opt['eval_freq'] / time_dif

        print('\nSteps: %d (frames: %d)' % (step, step * opt['actrep']))
        print('reward: %.2f' % total_reward)
        print('epsilon: %.2f' % agent.ep)
        print('lr: %G' % agent.lr)
        print('training time: %ds' % time_dif)
        print('training rate: %dfps' % training_rate)
        print('testing time: %ds' % eval_time)
        print('testing rate: %dfps' % (opt['actrep'] * opt['eval_steps'] / eval_time))
        print('num. ep.: %d' % nepisodes)
        print('num. rewards: %d' % nrewards)
        print('-' * 44)

    if step % opt['save_freq'] == 0 or step == opt['steps']:
        s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2, agent.valid_term = None, None, None, None, None, None, None
        optimizer = agent.optimizer
        agent.optimizer = None

        filename = opt['name']
        if opt['save_versions'] > 0:
            filename = filename + '_' + math.floor(step / opt['save_versions'])
        torch.save({
            'agent': agent,
            'model': agent.network,
            'best_model': agent.best_network,
            'reward_history': reward_history,
            'reward_counts': reward_counts,
            'episode_counts': episode_counts,
            'time_history': time_history,
            'v_history': v_history,
            'td_history': td_history,
            'qmax_history': qmax_history,
            'arguments': opt
        }, filename + '.tar')
        if opt['saveNetworkParams']:
            nets = {'network': agent.network.parameters()}
            torch.save(nets, filename + '.params.tar') # TODO: save as ASCII?

        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2, agent.valid_term = s, a, r, s2, term
        agent.optimizer = optimizer
        print('Saved:', filename + '.tar')
