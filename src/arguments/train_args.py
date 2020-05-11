from . import common_args, modify_args
from common.util import str2bool


def get_args(dev=False):
    parser = common_args.get_args()
    parser.add_argument('--isTrain', default=True, type=str2bool, help='is train?')
    ## Training settings
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--bs', default=256, type=int, help='default: 256')
    parser.add_argument('--epochs', default=50, type=int, help='50000 for vggface2')
    parser.add_argument('--decay_steps', default='10, 20, 30, 35, 40, 45, 50, 55', type=str, help='20000, 30000, 40000 for vggface2')
    ## Other setting
    parser.add_argument('--check_freq', default=200, type=int, help='frequency of accuracy check during an epoch')
    parser.add_argument('--eval_freq', default=1, type=int, help='frequency of evaluation')
    parser.add_argument('--name', type=str, default='master', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
    args = modify_args.run(parser, dev=dev)
    return args
