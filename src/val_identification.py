import torch
import torch.nn.functional as F
from common.loader import get_loader
from common.util import save_log


def run(net, args, step=None):
    net.eval()
    dataloader = get_loader(args, 'val_identification', batch_size=256).dataloader
    running_batch = 0
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for index, (face, target) in enumerate(dataloader):
            face, target = face.to(args.device), target.to(args.device)
            score, loss = net.forward(face, target)

            _, pred_labels = torch.max(F.softmax(score, dim=1), 1)
            pred_labels = pred_labels.view(-1)

            running_batch += len(target)
            running_loss += loss.item() * face.size(0)
            running_correct += torch.sum(torch.eq(pred_labels, target.view(-1))).item()

        running_loss /= running_batch
        running_correct /= running_batch
        message = 'Identification validation acc={:.4f} loss={:.4f} at {} epoch.'.format(running_correct, running_loss, step)
        print(message)
        save_log(message, args)

    return running_correct


