import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
import os
import sys
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def eval_acc(threshold, diff):
    y_predict = np.int32(diff[:,0]>threshold)
    y_true = np.int32(diff[:,1])
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def get_auc(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return auc


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def save_results(results, path):
    data_frame = pd.DataFrame(data=results)
    data_frame.to_csv(path + 'train_results.csv')


class L2Norm(nn.Module):
    def forward(self, input, dim=1):
        return F.normalize(input, p=2, dim=dim)


def face_ToTensor(img):
    return (ToTensor()(img) - 0.5) * 2


def save_log(message, args):
    log_name = os.path.join(args.checkpoints_dir, args.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('\n' + message)


def tensor_pair_cosine_distance(features11, features12, features21, features22, type='normal'):
    if type == 'concat':
        features1 = torch.cat((features11, features12), dim=1)
        features2 = torch.cat((features21, features22), dim=1)
    elif type == 'sum':
        features1 = features11 + features12
        features2 = features21 + features22
    elif type == 'normal':
        features1 = features11
        features2 = features21
    else:
        print('tensor_pair_cosine_distance unspported type!')
        sys.exit()
    scores = torch.nn.CosineSimilarity()(features1, features2)
    scores = scores.cpu().numpy().reshape(-1, 1)
    return scores


def tensor_pair_cosine_distance_matrix(features11, features12, features21, features22, type='normal'):
    if type == 'concat':
        features1 = torch.cat((features11, features12), dim=1)
        features2 = torch.cat((features21, features22), dim=1)
    elif type == 'sum':
        features1 = features11 + features12
        features2 = features21 + features22
    elif type == 'normal':
        features1 = features11
        features2 = features21
    else:
        print('tensor_pair_cosine_distance_matrix unspported type!')
        sys.exit()
    features1_np = features1.cpu().numpy()
    features2_np = features2.cpu().numpy()
    scores = 1 - cdist(features2_np, features1_np, 'cosine')
    return scores