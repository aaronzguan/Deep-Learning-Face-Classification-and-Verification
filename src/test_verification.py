import torch
from common.loader import get_loader
import os


def run(net, args):
    net.eval()
    dataloader = get_loader(args, 'test_verification', batch_size=256).dataloader
    score_predict = []
    with torch.no_grad():
        for index, (img1, img2, img1_flip, img2_flip, _) in enumerate(dataloader):
            img1, img1_flip = img1.to(args.device), img1_flip.to(args.device)
            img2, img2_flip = img2.to(args.device), img2_flip.to(args.device)
            features11, features12 = net(img1), net(img1_flip)
            features21, features22 = net(img2), net(img2_flip)
            features1 = torch.cat((features11, features12), dim=1)
            features2 = torch.cat((features21, features22), dim=1)

            cosineSimilarity = torch.nn.CosineSimilarity()(features1, features2)
            score_predict.extend(cosineSimilarity[:].cpu().numpy())

    with open(os.path.join(args._data_root, args._test_verification_pairs)) as f:
        pair_lines = f.readlines()
    trial = [line.replace('\n', '') for line in pair_lines]

    return trial, score_predict

