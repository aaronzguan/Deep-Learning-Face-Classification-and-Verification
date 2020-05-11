import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from common.loader import get_loader
from common.util import save_log, find_best_threshold, eval_acc, get_auc, tensor_pair_cosine_distance


def run(net, args, step=None):
    net.eval()
    dataloader = get_loader(args, 'val_verification', batch_size=256)
    num_faces = dataloader.num_faces
    features11_total = torch.Tensor(np.zeros((num_faces, args.feature_dim), dtype=np.float32)).to(args.device)
    features12_total, features21_total, features22_total = torch.zeros_like(features11_total), \
                                                           torch.zeros_like(features11_total), \
                                                           torch.zeros_like(features11_total)
    labels = torch.Tensor(np.zeros((num_faces, 1), dtype=np.float32)).to(args.device)
    dataloader = dataloader.dataloader
    with torch.no_grad():
        bs_total = 0
        for index, (img1, img2, img1_flip, img2_flip, targets) in enumerate(dataloader):
            bs = len(targets)
            img1, img1_flip = img1.to(args.device), img1_flip.to(args.device)
            img2, img2_flip = img2.to(args.device), img2_flip.to(args.device)
            features11 = net(img1)
            features12 = net(img1_flip)
            features21 = net(img2)
            features22 = net(img2_flip)
            features11_total[bs_total:bs_total + bs] = features11
            features12_total[bs_total:bs_total + bs] = features12
            features21_total[bs_total:bs_total + bs] = features21
            features22_total[bs_total:bs_total + bs] = features22
            labels[bs_total:bs_total + bs] = targets
            bs_total += bs
        assert bs_total == num_faces, print('Verification pairs should be {}!'.format(num_faces))
    labels = labels.cpu().numpy()
    for cal_type in ['concat']:  # cal_type: concat/sum/normal
        scores = tensor_pair_cosine_distance(features11_total, features12_total, features21_total, features22_total, type=cal_type)
        # fpr, tpr, thre = roc_curve(labels, scores)
        # auc = auc(fpr, tpr)
        auc = get_auc(labels, scores)
        thresholds = np.linspace(-10000, 10000, 10000 + 1)
        thresholds = thresholds / 10000
        predicts = np.hstack((scores, labels))
        best_thresh = find_best_threshold(thresholds, predicts)
        acc = eval_acc(best_thresh, predicts)

        message = 'Verification validation auc={:.4f}, acc={:.4f}, thre={:.4f} at {} epoch.'.format(auc, acc, best_thresh, step)
        print(message)
        save_log(message, args)

        #
        # fpr, tpr, _ = roc_curve(labels, scores)
        # roc_auc = auc(fpr, tpr)
        # if step == args.iterations:
        #     np_name = os.path.join(args.checkpoints_dir, args.name, 'cnn_roc_{}_{}.npz'.format(args.loss_type, 'lfw'))
        #     np.savez(np_name, name1=fpr, name2=tpr)
        #     message = 'The fpr and tpr is saved to {}'.format(np_name)
        #     print(message)
    return auc



