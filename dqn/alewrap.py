import os
import gym
import numpy as np
import torch

os.environ.setdefault('PATH', '')


class PoolEnv(gym.Wrapper):
    def __init__(self, env, pool_type='mean', pool_size=2):
        """`pool_type`-pool the last `pool_size` frames"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for pooling across time steps)
        self._obs_buffer = np.zeros((pool_size,)+env.observation_space.shape, dtype=np.uint8)
        self._buffer_idx = 0
        self._pool_func = eval('np.%s' % pool_type)
        self._pool_size = pool_size

    def step(self, action, *args, **kwargs):
        """pool last observations."""
        assert args == ()
        obs, reward, done, info = self.env.step(action, **kwargs)
        self._buffer_idx = (self._buffer_idx + 1) % self._pool_size
        self._obs_buffer[self._buffer_idx] = obs
        pooled_frame = self._pool_func(self._obs_buffer, axis=0)
        return pooled_frame, reward, done, info

    def reset(self, *args, **kwargs):
        assert args == ()
        obs = self.env.reset(**kwargs)
        self._obs_buffer[:, :] = obs
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action, episodic_life=False, *args, **kwargs):
        assert args == ()
        obs, reward, done, info = self.env.step(action, **kwargs)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if episodic_life and lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        assert args == ()
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        # no-op step to advance from terminal/lost life state
        obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=1):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, noop_max=None, *args, **kwargs):
        """ Do no-op action for a number of steps in [1, `noop_max`]."""
        assert args == ()
        obs = self.env.reset(**kwargs)
        if noop_max is None:
            noop_max = self.noop_max
        if noop_max > 0:
            noops = self.unwrapped.np_random.randint(1, noop_max + 1) #pylint: disable=E1101
        else:
            noops = 0
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac, *args, **kwargs):
        assert args == ()
        return self.env.step(ac, **kwargs)


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=1):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action, *args, **kwargs):
        """Repeat action and sum reward over last observations."""
        assert args == ()
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action, **kwargs)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, *args, **kwargs):
        assert args == ()
        obs = self.env.reset(**kwargs)
        return obs


class ScaledFloatFrame(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    @staticmethod
    def _process_obs(obs):
        return torch.tensor(obs, dtype=torch.float32).div_(255.0)

    def step(self, action, *args, **kwargs):
        assert args == ()
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = ScaledFloatFrame._process_obs(obs)
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        assert args == ()
        obs = self.env.reset(**kwargs)
        obs = ScaledFloatFrame._process_obs(obs)
        return obs


def alewrap(env, actrep=1, random_starts=1, pool_type='mean', pool_size=2, **kwargs):
    env = PoolEnv(env, pool_type=pool_type, pool_size=pool_size)
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=random_starts)
    env = SkipEnv(env, skip=actrep)
    env = ScaledFloatFrame(env)
    return env


def make_env(env, *args, **kwargs):
    if kwargs.get('verbose', 0) > 0:
        print('Playing: %s' % env)
    env = gym.make(env + 'NoFrameskip-v4')
    env = alewrap(env, *args, **kwargs)
    return env
