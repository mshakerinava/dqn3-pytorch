import sys
import copy
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from initenv import dqn
from reshape import Reshape
from nnutils import get_weight_norms, get_grad_norms


def set_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


class nql:
    def __init__(self, **kwargs):
        self.state_dim  = kwargs.get('state_dim') # State dimensionality.
        self.actions    = kwargs.get('actions')
        self.n_actions  = len(self.actions)
        self.verbose    = kwargs.get('verbose')
        self.best       = kwargs.get('best')

        ## epsilon annealing
        self.ep_start   = kwargs.get('ep', 1)
        self.ep         = self.ep_start # Exploration probability.
        self.ep_end     = kwargs.get('ep_end', self.ep)
        self.ep_endt    = kwargs.get('ep_endt', 1000000)

        ### learning rate annealing
        self.lr_start       = kwargs.get('lr', 0.01) # Learning rate.
        self.lr             = self.lr_start
        self.lr_end         = kwargs.get('lr_end', self.lr)
        self.lr_endt        = kwargs.get('lr_endt', 1000000)
        self.wc             = kwargs.get('wc', 0) # L2 weight cost.
        self.minibatch_size = kwargs.get('minibatch_size', 1)
        self.valid_size     = kwargs.get('valid_size', 500)

        ## Q-learning parameters
        self.discount       = kwargs.get('discount', 0.99) # Discount factor.
        self.update_freq    = kwargs.get('update_freq', 1)
        # Number of points to replay per learning step.
        self.n_replay       = kwargs.get('n_replay', 1)
        # Number of steps after which learning starts.
        self.learn_start    = kwargs.get('learn_start', 0)
         # Size of the transition table.
        self.replay_memory  = kwargs.get('replay_memory', 1000000)
        self.hist_len       = kwargs.get('hist_len', 1)
        self.rescale_r      = kwargs.get('rescale_r')
        self.max_reward     = kwargs.get('max_reward')
        self.min_reward     = kwargs.get('min_reward')
        self.clip_delta     = kwargs.get('clip_delta')
        self.target_q       = kwargs.get('target_q')
        self.bestq          = 0

        self.gpu            = kwargs.get('gpu')

        self.ncols          = kwargs.get('ncols', 1)  # number of color channels in input
        self.input_dims     = kwargs.get('input_dims', [self.hist_len * self.ncols, 84, 84])
        self.preproc        = kwargs.get('preproc')  # name of preprocessing network
        self.histType       = kwargs.get('histType', 'linear')  # history type to use
        self.histSpacing    = kwargs.get('histSpacing', 1)
        self.nonTermProb    = kwargs.get('nonTermProb', 1)
        self.bufferSize     = kwargs.get('bufferSize', 512)

        self.transition_params = kwargs.get('transition_params', {})

        self.network    = kwargs.get('network', self.createNetwork())

        # Check whether there is a network file.
        if type(self.network) != str:
            sys.exit('The type of the network provided in NeuralQLearner'
                ' is not a string!')

        try:
            create_network = importlib.import_module(self.network).create_network
        except:
            try:
                exp = torch.load(self.network)
                if self.best and exp['best_model']:
                    self.network = exp['best_model']
                else:
                    self.network = exp['model']
            except:
                sys.exit('Could not find network file.')

        print('Creating Agent Network from ' + self.network)
        self.network = create_network(**vars(self))

        if self.gpu is not None and self.gpu >= 0:
            self.network.cuda()
        else:
            self.network.cpu()

        # Load preprocessing network.
        if type(self.preproc) != str:
            sys.exit('The preprocessing is not a string')

        try:
            create_network = importlib.import_module(self.preproc).create_network
        except:
            sys.exit('Error loading preprocessing net')

        self.preproc = create_network(**vars(self))
        self.preproc.cpu()

        # Create transition table.
        ### Assuming the transition table always gets floating point input
        ### (Float tensors) and always returns one of the two, as required
        ### internally it always uses ByteTensors for states, scaling and
        ### converting accordingly.
        transition_args = {
            'gpu': self.gpu,
            'histLen': self.hist_len,
            'maxSize': self.replay_memory,
            'stateDim': self.state_dim,
            'histType': self.histType,
            'numActions': self.n_actions,
            'bufferSize': self.bufferSize,
            'histSpacing': self.histSpacing,
            'nonTermProb': self.nonTermProb
        }

        self.transitions = dqn['TransitionTable'](**transition_args)

        self.numSteps = 0 # Number of perceived states.
        self.lastState = None
        self.lastAction = None
        self.v_avg = 0 # V average.
        self.tderr_avg = 0 # TD error average.

        self.q_max = 1
        self.r_max = 1

        self.optimizer = optim.RMSprop(params=self.network.parameters(), lr=self.lr, alpha=0.95, eps=0.01, weight_decay=self.wc, momentum=0.95, centered=False)
        self.network.zero_grad()

        if self.target_q is not None:
            self.target_network = copy.deepcopy(self.network)


    def reset(self, state):
        if not state:
            return
        self.best_network = state['best_network']
        self.network = state['model']
        self.optimizer.zero_grad()
        self.numSteps = 0
        print('RESET STATE SUCCESFULLY')


    def preprocess(self, rawstate):
        if self.preproc is not None:
            return self.preproc(rawstate).clone().view([self.state_dim])
        return rawstate


    def getQUpdate(self, **kwargs):
        s = kwargs['s']
        a = kwargs['a']
        r = kwargs['r']
        s2 = kwargs['s2']
        term = kwargs['term']
        assert s.shape[0] == a.shape[0] == r.shape[0] == s2.shape[0] == term.shape[0]
        B = s.shape[0]

        # The order of calls to forward is a bit odd in order
        # to avoid unnecessary calls (we only need 2).

        # delta = r + (1 - terminal) * gamma * max_a Q(s2, a) - Q(s, a)
        term = (1 - term).to(torch.float32)

        if self.target_q:
            target_q_net = self.target_network
        else:
            target_q_net = self.network

        # Compute max_a Q(s_2, a).
        with torch.no_grad():
            q2_max = target_q_net(s2).max(axis=1).values

        # Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
        q2 = q2_max * self.discount * term

        delta = r.clone()

        if self.rescale_r is not None:
            delta /= self.r_max
        delta += q2

        # q = Q(s,a)
        q_all = self.network(s)
        q = torch.zeros(B, dtype=torch.float32)
        for i in range(q_all.shape[0]):
            q[i] = q_all[i, a[i]]
        delta -= q

        if self.clip_delta:
            delta[delta >= +self.clip_delta] = +self.clip_delta
            delta[delta <= -self.clip_delta] = -self.clip_delta

        targets = torch.zeros_like(q_all)
        for i in range(B):
            targets[i, a[i]] = delta[i]

        if self.gpu is not None and self.gpu >= 0:
            targets = targets.cuda()

        return targets, delta, q2_max, q_all


    def qLearnMinibatch(self):
        # Perform a minibatch Q-learning update:
        # w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
        assert self.transitions.size() > self.minibatch_size

        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)

        targets, delta, q2_max, q_all = self.getQUpdate(s=s, a=a, r=r, s2=s2,
            term=term, update_qmax=True)

        # Zero gradients of parameters.
        self.optimizer.zero_grad()

        # Get new gradients.
        (-targets.detach() * q_all).sum().backward()

        # Compute linearly annealed learning rate.
        t = max(0, self.numSteps - self.learn_start)
        self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t) / self.lr_endt + self.lr_end
        self.lr = max(self.lr, self.lr_end)

        set_lr(self.optimizer, self.lr)
        self.optimizer.step()


    def sample_validation_data(self):
        s, a, r, s2, term = self.transitions.sample(self.valid_size)
        self.valid_s    = s.clone()
        self.valid_a    = a.clone()
        self.valid_r    = r.clone()
        self.valid_s2   = s2.clone()
        self.valid_term = term.clone()


    def compute_validation_statistics(self):
        targets, delta, q2_max, q_all = self.getQUpdate(s=self.valid_s, a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term)

        self.v_avg = (self.q_max * q2_max.mean()).item()
        self.tderr_avg = (delta.clone().abs().mean()).item()


    def perceive(self, reward, rawstate, terminal, testing=False, testing_ep=None):
        # Preprocess state (will be set to None if terminal)
        state = self.preprocess(rawstate)

        if self.max_reward is not None:
            reward = min(reward, self.max_reward)
        if self.min_reward is not None:
            reward = max(reward, self.min_reward)
        if self.rescale_r is not None:
            self.r_max = max(self.r_max, reward)

        self.transitions.add_recent_state(state, terminal)

        # store transition (s, a, r, s')
        if self.lastState is not None and not testing:
            self.transitions.add(self.lastState, self.lastAction, reward, self.lastTerminal)

        if self.numSteps == self.learn_start + 1 and not testing:
            self.sample_validation_data()

        curState = self.transitions.get_recent()
        curState = curState.view(1, *self.input_dims)

        # Select action
        actionIndex = 0
        if not terminal:
            actionIndex = self.eGreedy(curState, testing_ep)

        self.transitions.add_recent_action(actionIndex)

        # Do some Q-learning updates
        if self.numSteps > self.learn_start and not testing and self.numSteps % self.update_freq == 0:
            for i in range(self.n_replay):
                self.qLearnMinibatch()

        if not testing:
            self.numSteps += 1

        self.lastState = state.clone()
        self.lastAction = actionIndex
        self.lastTerminal = terminal

        if self.target_q and self.numSteps % self.target_q == 1:
            self.target_network = copy.deepcopy(self.network)

        return actionIndex


    def eGreedy(self, state, testing_ep=None):
        self.ep = testing_ep or (self.ep_end +
                  max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                  max(0, self.numSteps - self.learn_start)) / self.ep_endt))
        # Epsilon greedy
        if torch.rand(size=()) < self.ep:
            return torch.randint(self.n_actions, size=())
        else:
            return self.greedy(state)


    def greedy(self, state):
        if state.dim() == 2:
            assert False, 'Input must be at least 3D'

        if self.gpu is not None and self.gpu >= 0:
            state = state.cuda()

        with torch.no_grad():
            q = self.network(state).cpu().squeeze()
        maxq = q[0]
        besta = [0]

        # Evaluate all other actions (with random tie-breaking)
        for a in range(1, self.n_actions):
            if q[a] > maxq:
                besta = [a]
                maxq = q[a]
            elif q[a] == maxq:
                besta.append(a)
        self.bestq = maxq

        r = torch.randint(len(besta), size=())

        self.lastAction = besta[r]

        return besta[r]


    def createNetwork(self):
        n_hid = 128
        mlp = []
        mlp.append(Reshape([self.hist_len * self.ncols * self.state_dim]))
        mlp.append(nn.Linear(in_features=self.hist_len * self.ncols * self.state_dim, out_features=n_hid))
        mlp.append(nn.ReLU())
        mlp.append(nn.Linear(in_features=n_hid, out_features=n_hid))
        mlp.append(nn.ReLU())
        mlp.append(nn.Linear(in_features=n_hid, out_features=self.n_actions))
        mlp = nn.Sequential(*mlp)
        return mlp


    def _loadNet(self):
        net = self.network
        if self.gpu is not None and self.gpu >= 0:
            net.cuda()
        else:
            net.cpu()
        return net


    def init(self, **kwargs):
        self.actions = kwargs.get('actions')
        self.n_actions = len(self.actions)
        self.network = self._loadNet()
        # Generate targets.
        self.transitions.empty()


    def report(self):
        print(get_weight_norms(self.network), end='')
        print(get_grad_norms(self.network), end='')
        print('=' * 30)
        print()


dqn['NeuralQLearner'] = nql
