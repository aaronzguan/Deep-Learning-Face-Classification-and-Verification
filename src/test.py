import torch
import torch.nn as nn
from models.models import CreateModel
from arguments import test_args
import pandas as pd
import test_identification
import test_verification

_backbone_model = '/Users/aaron/Desktop/11785-Intro to Deep Learning/Homework Part2/Homework2/src/checkpoints/0223-1/19_net_backbone.pth'
_criterion_model = '/Users/aaron/Desktop/11785-Intro to Deep Learning/Homework Part2/Homework2/src/checkpoints/0223-1/19_net_criterion.pth'

if __name__ == '__main__':
    args = test_args.get_args()
    model = CreateModel(args, class_num=args.class_num)
    model.backbone.load_state_dict(torch.load(_backbone_model))
    model.criterion.load_state_dict(torch.load(_criterion_model))

    if torch.cuda.is_available():
        if len(args.gpu_ids) > 1:
            model.backbone = nn.DataParallel(model.backbone)

    test_identification_id, test_identification_label = test_identification.run(model, args)
    d = {'id': test_identification_id, 'Category': test_identification_label}
    df = pd.DataFrame(data=d)
    df.to_csv('hw2p2_identification_test.csv', header=True, index=False)
    print('Identification testing is done, result is saved to hw2p2_identification_test.csv')

    test_verification_trial, test_verification_score = test_verification.run(model.backbone, args)
    d = {'trial': test_verification_trial, 'score': test_verification_score}
    df = pd.DataFrame(data=d)
    df.to_csv('hw2p2_verification_test.csv', header=True, index=False)
    print('Verification testing is done, result is saved to hw2p2_verification_test.csv')