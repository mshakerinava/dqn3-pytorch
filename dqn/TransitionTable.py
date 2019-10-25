import math
import torch
from initenv import dqn


class TransitionTable:
    def __init__(self, **kwargs):
        self.stateDim = kwargs.get('stateDim')
        self.numActions = kwargs.get('numActions')
        self.histLen = kwargs.get('histLen')
        self.maxSize = kwargs.get('maxSize', 1024 ** 2)
        self.bufferSize = kwargs.get('bufferSize', 1024)
        self.histType = kwargs.get('histType', 'linear')
        self.histSpacing = kwargs.get('histSpacing', 1)
        self.zeroFrames = kwargs.get('zeroFrames', 1)
        self.nonTermProb = kwargs.get('nonTermProb', 1)
        self.nonEventProb = kwargs.get('nonEventProb', 1)
        self.gpu = kwargs.get('gpu')
        self.numEntries = 0
        self.insertIndex = 0

        histLen = self.histLen
        self.histIndices = [None] * histLen
        if self.histType == "linear":
            # History is the last histLen frames.
            self.recentMemSize = self.histSpacing * histLen
            for i in range(histLen):
                self.histIndices[i] = i * self.histSpacing
        elif self.histType == "exp2":
            # The ith history frame is from 2^(i-1) frames ago.
            self.recentMemSize = 2 ^ (histLen - 1)
            self.histIndices[0] = 1
            for i in range(histLen - 1):
                self.histIndices[i + 1] = self.histIndices[i] + 2 ** (7 - i)
        elif self.histType == "exp1.25":
            # The ith history frame is from 1.25^(i-1) frames ago.
            self.histIndices[histLen] = 1
            for i in range(histLen - 2, -1, -1):
                self.histIndices[i] = math.ceil(1.25 * self.histIndices[i + 1]) + 1
            self.recentMemSize = self.histIndices[0]
            for i in range(histLen):
                self.histIndices[i] = self.recentMemSize - self.histIndices[i] + 1

        min_idx = min(self.histIndices)
        for i in range(histLen):
            self.histIndices[i] -= min_idx
        self.recentMemSize = self.histIndices[-1] + 1

        self.s = torch.zeros([self.maxSize, self.stateDim], dtype=torch.uint8)
        self.a = torch.zeros([self.maxSize], dtype=torch.int64)
        self.r = torch.zeros([self.maxSize], dtype=torch.float32)
        self.t = torch.zeros([self.maxSize], dtype=torch.uint8)

        # Tables for storing the last histLen states. They are used for
        # constructing the most recent agent state more easily.
        self.recent_s = []
        self.recent_a = []
        self.recent_t = []

        self.buf_a      = torch.zeros([self.bufferSize], dtype=torch.int64)
        self.buf_r      = torch.zeros([self.bufferSize], dtype=torch.float32)
        self.buf_term   = torch.zeros([self.bufferSize], dtype=torch.uint8)
        self.buf_s      = torch.zeros([self.bufferSize, self.stateDim * self.histLen], dtype=torch.float32)
        self.buf_s2     = torch.zeros([self.bufferSize, self.stateDim * self.histLen], dtype=torch.float32)

        if self.gpu is not None and self.gpu >= 0:
            self.gpu_s  = self.buf_s.cuda()
            self.gpu_s2 = self.buf_s2.cuda()


    def reset(self):
        self.numEntries = 0
        self.insertIndex = 0


    def size(self):
        return self.numEntries


    def empty(self):
        return self.numEntries == 0


    def fill_buffer(self):
        assert self.numEntries >= self.bufferSize
        # clear CPU buffers
        self.buf_ind = 0
        for buf_ind in range(self.bufferSize):
            s, a, r, s2, term = self.sample_one()
            s = s.to(torch.float32).div_(255).view([-1])
            s2 = s2.to(torch.float32).div_(255).view([-1])
            self.buf_s[buf_ind].copy_(s)
            self.buf_a[buf_ind] = a
            self.buf_r[buf_ind] = r
            self.buf_s2[buf_ind].copy_(s2)
            self.buf_term[buf_ind] = term
        if self.gpu is not None and self.gpu >= 0:
            self.gpu_s.copy_(self.buf_s)
            self.gpu_s2.copy_(self.buf_s2)


    def sample_one(self):
        assert self.numEntries > 1
        valid = False
        while not valid:
            # start at 1 because of previous action
            index = torch.randint(1, self.numEntries - self.recentMemSize, size=())
            if self.t[index + self.recentMemSize - 1] == 0:
                valid = True
            if self.nonTermProb < 1 and self.t[index + self.recentMemSize] == 0 and torch.rand(size=()) > self.nonTermProb:
                # Discard non-terminal states with probability (1-nonTermProb).
                # Note that this is the terminal flag for s_{t+1}.
                valid = False
            if self.nonEventProb < 1 and self.t[index + self.recentMemSize] == 0 and self.r[index + self.recentMemSize - 1] == 0 and torch.rand(size=()) > self.nonTermProb:
                # Discard non-terminal or non-reward states with
                # probability (1-nonTermProb).
                valid = False

        return self.get(index)


    def sample(self, batch_size=None):
        batch_size = batch_size or 1
        assert batch_size < self.bufferSize

        if 'buf_ind' not in vars(self) or self.buf_ind + batch_size - 1 > self.bufferSize:
            self.fill_buffer()

        rng = range(self.buf_ind, self.buf_ind + batch_size)
        self.buf_ind += batch_size

        buf_s, buf_s2, buf_a, buf_r, buf_term = self.buf_s, self.buf_s2, self.buf_a, self.buf_r, self.buf_term
        if self.gpu and self.gpu >= 0:
            buf_s = self.gpu_s
            buf_s2 = self.gpu_s2

        return buf_s[rng], buf_a[rng], buf_r[rng], buf_s2[rng], buf_term[rng]


    def concatFrames(self, index, use_recent=False):
        # returns s_{`index + self.recentMemSize - 1`}
        if use_recent:
            s, t = self.recent_s, self.recent_t
        else:
            s, t = self.s, self.t

        fullstate = torch.zeros([self.histLen, self.stateDim], dtype=torch.uint8)

        # Zero out frames from all but the most recent episode.
        zero_out = False
        episode_start = self.histLen - 1

        for i in range(self.histLen - 1, 0, -1):
            if not zero_out:
                for j in range(index + self.histIndices[i - 1], index + self.histIndices[i]):
                    if t[j] == 1:
                        zero_out = True
                        break

            if not zero_out:
                episode_start = i - 1

        if self.zeroFrames == 0:
            episode_start = 0

        # Copy frames from the current episode.
        for i in range(episode_start, self.histLen):
            fullstate[i].copy_(s[index + self.histIndices[i]])

        return fullstate


    def get_recent(self):
        # Assumes that the most recent state has been added, but the action has not
        return self.concatFrames(0, True).float().div(255)


    def get(self, index):
        s = self.concatFrames(index)
        s2 = self.concatFrames(index + 1)
        ar_index = index + self.recentMemSize - 1

        return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index + 1]


    def add(self, s, a, r, term):
        assert s is not None, 'State cannot be None'
        assert a is not None, 'Action cannot be None'
        assert r is not None, 'Reward cannot be None'

        # increment until at full capacity
        if self.numEntries < self.maxSize:
            self.numEntries += 1

        # Overwrite (s, a, r, t) at insertIndex
        self.s[self.insertIndex] = s.clone().mul(255).byte()
        self.a[self.insertIndex] = a
        self.r[self.insertIndex] = r
        self.t[self.insertIndex] = 1 if term else 0

        # always insert at next index, then wrap around
        # overwrite oldest experience once at capacity
        self.insertIndex = (self.insertIndex + 1) % self.maxSize


    def add_recent_state(self, s, term):
        s = s.clone().mul(255).byte()
        if len(self.recent_s) == 0:
            for i in range(self.recentMemSize):
                self.recent_s.append(s.clone().zero_())
                self.recent_t.append(1)

        self.recent_s.append(s)
        if term:
            self.recent_t.append(1)
        else:
            self.recent_t.append(0)

        # keep `recentMemSize` states
        if len(self.recent_s) > self.recentMemSize:
            self.recent_s = self.recent_s[1:]
            self.recent_t = self.recent_t[1:]


    def add_recent_action(self, a):
        if len(self.recent_a) == 0:
            for i in range(self.recentMemSize):
                self.recent_a.append(0)

        self.recent_a.append(a)

        # Keep `recentMemSize` steps.
        if len(self.recent_a) > self.recentMemSize:
            self.recent_a = self.recent_a[1:]


    # Override the write function to serialize this class into a file.
    # We do not want to store anything into the file, just the necessary info
    # to create an empty transition table.

    # @param file (FILE object ) @see torch.DiskFile
    def write(self, file): # TODO!
        torch.save([
            self.stateDim,
            self.numActions,
            self.histLen,
            self.maxSize,
            self.bufferSize,
            self.numEntries,
            self.insertIndex,
            self.recentMemSize,
            self.histIndices
        ], file)


    # Override the read function to desearialize this class from file.
    # Recreates an empty table.

    # @param file (FILE object ) @see torch.DiskFile
    def read(self, file):
        stateDim, numActions, histLen, maxSize, bufferSize, numEntries, insertIndex, recentMemSize, histIndices = torch.load(file)
        self.stateDim = stateDim
        self.numActions = numActions
        self.histLen = histLen
        self.maxSize = maxSize
        self.bufferSize = bufferSize
        self.recentMemSize = recentMemSize
        self.histIndices = histIndices
        self.numEntries = 0
        self.insertIndex = 0

        self.s = torch.zeros(self.maxSize, dtype=torch.uint8)
        self.a = torch.zeros(self.maxSize, dtype=torch.int64)
        self.r = torch.zeros(self.maxSize, dtype=torch.float32)
        self.t = torch.zeros(self.maxSize, dtype=torch.uint8)

        # Tables for storing the last histLen states. They are used for
        # constructing the most recent agent state more easily.
        self.recent_s = []
        self.recent_a = []
        self.recent_t = []

        self.buf_a      = torch.zeros(self.bufferSize, dtype=torch.int64)
        self.buf_r      = torch.zeros(self.bufferSize, dtype=torch.float32)
        self.buf_term   = torch.zeros(self.bufferSize, dtype=torch.uint8)
        self.buf_s      = torch.zeros(self.bufferSize, self.stateDim * self.histLen, dtype=torch.float32)
        self.buf_s2     = torch.zeros(self.bufferSize, self.stateDim * self.histLen, dtype=torch.float32)

        if self.gpu is not None and self.gpu >= 0:
            self.gpu_s  = self.buf_s.cuda()
            self.gpu_s2 = self.buf_s2.cuda()


dqn['TransitionTable'] = TransitionTable
