from models.models import CreateModel
from common.loader import get_loader
from common.util import save_log
from arguments import train_args
from tqdm import tqdm
import val_verification
import val_identification

if __name__ == '__main__':
    args = train_args.get_args()
    dataloader = get_loader(args, 'training', args.bs)
    model = CreateModel(args, class_num=dataloader.num_class)
    model.train_setup()   # setup the optimizer, scheduler and initialization for training

    # model.eval()
    # import hiddenlayer as hl
    # import torch
    # input = torch.randn(1, 3, 64, 64)
    # graph = hl.build_graph(model.backbone, input)
    # graph = graph.build_dot()
    # graph.render('/Users/aaron/Desktop/spherenet.png', view=True, format='png')

    # from torchsummary import summary
    # summary(model.backbone, (3, 96, 96))      # summary(your_model, input_size=(channels, H, W))

    pbar = tqdm(range(1, args.epochs + 1), ncols=0)
    best_acc = 0
    best_auc = 0.50
    for epoch in pbar:
        model.train()
        model.update_learning_rate()  # update learning rate
        for i, (data, label) in enumerate(dataloader.dataloader):
            model.optimize_parameters(data, label)    # forward and backprop
            if (i+1) % args.check_freq == 0:
                states = model.get_current_states()
                # display
                description = '[{}|{}] '.format(epoch, i+1)
                for name, value in states.items():
                    description += '{}: {:.4f} '.format(name, value)
                pbar.set_description(desc=description)
                save_log(description, args)

        if epoch % args.eval_freq == 0:
            model.eval()
            val_acc = val_identification.run(model, args, epoch)
            val_auc = val_verification.run(model.backbone, args, epoch)
            if val_acc > best_acc:
                model.save_networks(epoch)
                best_acc = val_acc
            elif val_auc > best_auc:
                model.save_networks(epoch)
                best_auc = val_auc

    model.save_networks(epoch)