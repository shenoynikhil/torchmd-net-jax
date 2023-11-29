from flax import linen as nn

class TorchMDET(nn.Module):

    @nn.compact
    def __call__(self, x):
        raise NotImplementedError

class TorchMDNet(nn.Module):

    @nn.compact
    def __call__(self, x):
        raise NotImplementedError

