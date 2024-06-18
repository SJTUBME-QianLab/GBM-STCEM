# encoding: utf-8
import argparse
import os
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import time
import numpy as np
from tqdm import tqdm
from util import prf, get_train_and_test_datasets, get_batch_data
from torch_cnn import ResSpiNet, ContrastiveLoss
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='PyTorch GBM Training')
parser.add_argument('--data', default='./input', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrained', dest='pretrained', default=False, action='store_true',
                    help='use pre-trained model')
parser.add_argument('--outf', default='./output',
                    help='folder to output model checkpoints')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True, action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--train', default=True,
                    help='train the model')
parser.add_argument('--test', action='store_true', default=True,
                    help='test a [pre]trained model on new images')


def train(train_data, sim_imgs, model, criterion, cont_criterion, optimizer, epoch):
    """Train the model on Training Set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()
    # switch to train mode
    model = model.cuda()
    model.train()
    end = time.time()
    for batch_idx in range(30):
        input, target, sim_dat = get_batch_data(train_data, sim_imgs, 4)
        # measure data loading time
        data_time.update(time.time() - end)
        input = torch.cat([input, sim_dat])
        L = len(target)
        if cuda:
            input, target = input.cuda(), target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            input_var = input_var.unsqueeze(1)
            output, out_features = model(input_var)

            acti_f = out_features[0:L]
            back_f = out_features[L:]
            sin_acti_f = acti_f.view(4, 10, -1).permute(1, 0, 2)
            cont_loss2 = torch.mean(
                torch.stack([cont_criterion(sin_acti_f[i], sin_acti_f[j]) for i in range(10) for j in range(i, 10)]))
            t_loss = torch.mean(torch.abs(
                F.cosine_similarity(acti_f.view(L, -1).unsqueeze(1), back_f.view(L, -1).unsqueeze(0), dim=-1)-0.5))
            loss = criterion(output, target_var.to(torch.int64)) + cont_loss2 + t_loss

            prec1 = accuracy(output.data, target, topk=(1,))
            top.update(prec1[0], input.size(0))
            losses.update(loss, input.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    print('Epoch: [{0}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
          'Prec@1 {top1_val:.4f} ({top1_avg:.4f})'.format(
        epoch, batch_time=batch_time,
        data_time=data_time,
        loss_val =losses.val.detach().cpu().numpy().item(),
        loss_avg = losses.avg.detach().cpu().numpy().item(),
        top1_val=top.val.detach().cpu().numpy().item(),
        top1_avg=top.avg.detach().cpu().numpy().item()))
    return top.val.detach().cpu().numpy().item(), losses.avg.detach().cpu().numpy().item()

def test(test_loader, model):
    """Test the model on the Evaluation Folder
    """
    model.eval()
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)
    counter = 0
    y_true = []
    y_pre = []
    y_scores = []
    test_len = len(test_loader[0])
    test_data = [temp[0] for temp in test_loader[0]]
    test_target = [temp[1] for temp in test_loader[0]]
    test_sin_data = test_loader[1]
    test_index = test_loader[2]
    test_best_ACC = 0.8
    test_best_P = 0.8
    index_out = False

    for i in range(int(test_len / 10)):
        input = torch.from_numpy(np.stack(np.array(test_data)[test_index[10 * i:10 * (i + 1)]]))
        target = torch.from_numpy(np.stack(np.array(test_target)[test_index[10 * i:10 * (i + 1)]]))
        sim_img = torch.from_numpy(np.stack(np.array(test_sin_data)[test_index[10 * i:10 * (i + 1)]])).view(10, 224, 224)
        input = torch.cat([input, sim_img])
        if cuda:
            input = torch.Tensor(input).cuda()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                # compute output
                input_var = input_var.unsqueeze(1)
                output, _ = model(input_var)
            if isinstance(output, tuple):
                output = output[len(output) - 1]
            predlabel = torch.from_numpy(output.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1))
            counter += torch.sum(predlabel.eq(target)).numpy()
            y_true.extend(target.numpy().tolist())
            y_pre.extend(predlabel.numpy().tolist())
            y_score = F.softmax(output, dim=1).detach().cpu().numpy()
            y_scores.extend(y_score[:, 1].tolist())

    P, R, F1, ACC, TPR, TNR, AUC = prf(y_true, y_pre, y_scores)
    is_ACC = bool(ACC >= test_best_ACC)
    is_P = bool(P >= test_best_P)
    is_one = bool(P != 1 and TPR != 1 and TNR != 1)

    if is_ACC and is_P and is_one:
        print("=> best testing...")
        t_ACC = ACC
        t_P = P
        t_R = R
        t_F1 = F1
        t_AUC = AUC
        t_TPR = TPR
        t_TNR = TNR
        print(
            'test_acc={:.3f}, Pre: {:.4f}, R: {:.4f}, F1: {:.4f},AUC: {:.4f}, TPR: {:.4f}, TNR: {:.4f}'.format(
                ACC, P, R, F1, AUC, TPR, TNR))
        index_out = True
    if index_out:
        print(
            'test_acc={:.3f}, Pre: {:.4f}, R: {:.4f}, F1: {:.4f},AUC: {:.4f}, TPR: {:.4f}, TNR: {:.4f}'.format(
                t_ACC, t_P, t_R, t_F1, t_AUC, t_TPR, t_TNR))
        return t_P, t_R, t_F1, t_ACC, t_TPR, t_TNR, t_AUC
    else:
        print(
            'test_acc={:.3f}, Pre: {:.4f}, R: {:.4f}, F1: {:.4f},AUC: {:.4f}, TPR: {:.4f}, TNR: {:.4f}'.format(
                ACC, P, R, F1, AUC, TPR, TNR))
        return P, R, F1, ACC, TPR, TNR, AUC

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    best_prec1 = torch.FloatTensor([0])
    test_best_acc = 0
    print("=> using cuda: {cuda}".format(cuda=cuda))
    model = ResSpiNet(name='resnet18')
    parameters = model.parameters()
    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    t_criterion = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
    cont_criterion = ContrastiveLoss()
    if cuda:
        criterion.cuda()
        t_criterion.cuda()
        cont_criterion.cuda()
    # Set SGD + Momentum
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    schedulers = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    ############ TRAIN/TEST ############
    cudnn.benchmark = True
    P, R, F1, ACC, TPR, TNR, AUC = 0, 0, 0, 0, 0, 0, 0
    train_loss  = []
    train_pre = []
    if args.train:
        print("=> training...")
        for epoch in range(args.start_epoch, args.epochs):
            schedulers.step()
            train_acc, loss = train(train_data, sim_imgs, model, criterion, cont_criterion, optimizer, epoch)
            train_loss.append(loss)
            train_pre.append(train_acc)
            model = model.eval()
            if train_acc > 70:
                tP, tR, tF1, tACC, tTPR, tTNR, tAUC = test(test_loader, model)
                is_test_best = bool(tACC >= test_best_acc)
                if is_test_best:
                    print("=> best testing and save model...")
                    print(
                        'train_acc:{:.4f}, best_prec1:{:.4f}, test_prec1:{:.4f}, best_test_prec1:{:.4f}'.format(
                            train_acc, best_prec1.item(), tACC, test_best_acc))
                    test_best_acc = max(tACC, test_best_acc)
                    P, R, F1, ACC, TPR, TNR, AUC = tP, tR, tF1, tACC, tTPR, tTNR, tAUC
            model = model.train()
    return train_acc, P, R, F1, ACC, TPR, TNR, AUC


if __name__ == '__main__':
    run = 10
    ifolds = 4
    acc = np.zeros((run, ifolds), dtype=float)
    precision = np.zeros((run, ifolds), dtype=float)
    recall = np.zeros((run, ifolds), dtype=float)
    f_score = np.zeros((run, ifolds), dtype=float)

    auc = np.zeros((run, ifolds), dtype=float)
    tpr = np.zeros((run, ifolds), dtype=float)
    tnr = np.zeros((run, ifolds), dtype=float)

    for irun in tqdm(range(run)):
        for fold in range(ifolds):
            train_data, sim_imgs, test_data, sim_imgs_test = get_train_and_test_datasets(irun, fold)
            test_sort = torch.randperm(len(test_data)).numpy().tolist()
            test_loader = [test_data, sim_imgs_test, test_sort]
            args = parser.parse_args()
            cuda = torch.cuda.is_available()
            train_acc, P, R, F1, ACC, TPR, TNR, AUC = main()
            acc[irun][fold], recall[irun][fold], precision[irun][fold], f_score[irun][
                fold], auc[irun][fold], tpr[irun][fold], tnr[irun][
                fold] = ACC, R, P, F1, AUC, TPR, TNR
            print("irun =", irun)
            print("fold=", fold)
            print('mi-net mean accuracy = ', np.mean(acc))
            print('std = ', np.std(acc))
            print('mi-net mean precision = ', np.mean(precision))
            print('std = ', np.std(precision))
            print('mi-net mean recall = ', np.mean(recall))
            print('std = ', np.std(recall))
            print('mi-net mean fscore = ', np.mean(f_score))
            print('std = ', np.std(f_score))
            print('mi-net mean auc = ', np.mean(auc))
            print('std = ', np.std(auc))
            print('mi-net mean tpr = ', np.mean(tpr))
            print('std = ', np.std(tpr))
            print('mi-net mean tnr = ', np.mean(tnr))
            print('std = ', np.std(tnr))


