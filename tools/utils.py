from collections import namedtuple
import torch
import numpy as np
from collections import OrderedDict

Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))


class nor_std():
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape,dtype=np.float32)
        self._S = np.zeros(shape,dtype = np.float32)
        self.shape = self._M.shape

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            old = self._M.copy()
            self._M[...] = old + (x - old) / self._n
            self._S[...] = self._S + (x - old) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)


class Normalization():
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.sms = nor_std(shape)

    def __call__(self, x, update=True):
        if update:
            self.sms.push(x)
        if self.demean:
            x = x - self.sms.mean
        if self.destd:
            x = x / (self.sms.std + 1e-10)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def get_mean_std(self):
        return self.sms.mean, self.sms.std

    def output_shape(self, input_space):
        return input_space.shape


class Memory():
    def __init__(self):
        self.memory = []
    def push(self, *args):
        self.memory.append(Transition(*args))
    def clear(self):
        self.memory = []
    def sample(self):
        return Transition(*zip(*self.memory))  # zip each dimension(each dimension contains each runs) (state = (1,2,3,4,5..), action = (1,2,3,4,5..))


    def __len__(self):
        return len(self.memory)

def update_init_params(target, old, step_size = 0.1):
    """Apply one step of gradient descent on the loss function `loss`, with
    step-size `step_size`, and returns the updated parameters of the neural
    meta_policy.
    """
    updated = OrderedDict()
    for ((name_old, oldp), (name_target, targetp)) in zip(old.items(), target.items()):
        assert name_old == name_target, "target and old params are different"
        updated[name_old] = oldp + step_size * (targetp - oldp) # grad ascent so its a plus
    return updated
if __name__ == '__main__':
    mean = 0
    count =0
    for i in range(100):
        i+=1
        count +=i
        old = mean
        mean = old + (i - old)/i
        print('cumulated mean :{}'.format(mean))
        print('right mean :{}'.format(count/i))