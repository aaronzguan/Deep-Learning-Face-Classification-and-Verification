import argparse
from common.util import str2bool

def get_args():
    parser = argparse.ArgumentParser(description='Deep face recognition')
    ## Loss settings
    parser.add_argument('--loss_type', default='centerloss', type=str,
                        help='softmax/focalloss/asoftmax/arcface/amsoftmax')
    parser.add_argument('--use_f_norm', default=True, type=str2bool, help='feature normalization?')
    parser.add_argument('--use_w_norm', default=True, type=str2bool, help='weight normalization?')
    parser.add_argument('--s', default=30, type=float, help='to re-scale feature norm')
    parser.add_argument('--m_1', default=5, type=float, help='margin of SphereFace')
    parser.add_argument('--m_2', default=0.5, type=float, help='margin of ArcFace')
    parser.add_argument('--m_3', default=0.4, type=float, help='margin of CosineFace')
    parser.add_argument('--lamb', default=2, type=float, help='lambda of Center loss')
    ## Model settings
    parser.add_argument('--backbone', default='spherenet36', type=str, help='spherenetx/seresnet50/densenet121/densenet161')
    parser.add_argument('--double_depth', default=True, type=str2bool, help='double the depth of model (except shortcut)?')
    parser.add_argument('--use_batchnorm', default=True, type=str2bool, help='batch normalization?')
    parser.add_argument('--use_pool', default=False, type=str2bool, help='use global pooling?')
    parser.add_argument('--use_dropout', default=False, type=str2bool, help='use dropout?')
    parser.add_argument('--feature_dim', default=1024, type=int, help='feature dimension')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu ids: e.g. 0 0,1,2, 0,2.')
    ## Dataset Path Settings
    # parser.add_argument('--_data_root', default='/home/ubuntu/data/hw2/', type=str, help='dataset root path')
    parser.add_argument('--_data_root', default='/Users/aaron/Desktop/11785-Intro to Deep Learning/Homework Part2/Homework2/data/', type=str, help='dataset root path')
    parser.add_argument('--_val_identification_set', default='val_data/validation_classification', type=str, help='identification validation dataset')
    parser.add_argument('--_test_identification_set', default='test_data/test_classification/medium', type=str, help='identification test dataset')
    parser.add_argument('--_test_identification_order', default='test_data/test_order_classification.txt', type=str,
                        help='identification test order')
    parser.add_argument('--_val_verification_set', default='val_data/validation_verification', type=str,
                        help='verification validation dataset')
    parser.add_argument('--_val_verification_pairs', default='val_data/validation_trials_verification.txt', type=str,
                        help='verification validation pairs')
    parser.add_argument('--_test_verification_set', default='test_data/test_verification', type=str,
                        help='verification test dataset')
    parser.add_argument('--_test_verification_pairs', default='test_data/test_trials_verification_student.txt', type=str,
                        help='verification test pairs')
    ## Others
    parser.add_argument('--image_size', default=64, type=int, help='image size')
    return parser