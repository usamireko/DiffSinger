from .wavenet import WaveNet
from .lynxnet import LYNXNet
from .lynxnet2 import LYNXNet2

BACKBONES = {
    "wavenet": WaveNet,
    "lynxnet": LYNXNet,
    "lynxnet2": LYNXNet2,
}
