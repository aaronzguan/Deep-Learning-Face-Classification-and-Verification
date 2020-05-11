from . import common_args, modify_args
from common.util import str2bool


def get_args():
    parser = common_args.get_args()
    parser.add_argument('--isTrain', default=False, type=str2bool, help='is train?')
    parser.add_argument('--class_num', default=4300, type=int, help='number of class for classification')
    
    # parser.add_argument('--verification_threshold', default=0.1572, type=float, help='threshold for evaluating the verification')
    parser.add_argument('--_backbone_model', default='/home/ubuntu/HomeworkPart2/Homework2/checkpoints/master_20200301-173725_spherenet20_amsoftmax/41_net_backbone.pth', type=str)
    parser.add_argument('--_criterion_model', default='/home/ubuntu/HomeworkPart2/Homework2/checkpoints/master_20200301-173725_spherenet20_amsoftmax/41_net_criterion.pth', type=str)
    args = modify_args.run(parser, dev=True)
    return args



