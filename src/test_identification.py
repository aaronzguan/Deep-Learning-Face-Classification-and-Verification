import torch
from common.loader import get_loader
import torch.nn.functional as F
import os


def run(net, args):
    net.eval()
    dataloader = get_loader(args, 'test_identification', batch_size=256).dataloader
    label = []
    with torch.no_grad():
        for index, (face, target) in enumerate(dataloader):
            face, target = face.to(args.device), target.to(args.device)
            score, loss = net.forward(face, target)
            _, pred_labels = torch.max(F.softmax(score, dim=1), 1)
            pred_labels = pred_labels.view(-1)
            label.extend(pred_labels.cpu().numpy())

    with open(os.path.join(args._data_root, args._test_identification_order)) as f:
        lines = f.readlines()
    id = [line.replace('\n', '') for line in lines]

    return id, label