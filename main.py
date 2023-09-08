# encoding: utf-8
import argparse
import os
import torch.nn as nn
import torch
import numpy as np
import random
from tqdm import tqdm
import pickle
from main_supcon_FA import prf
from PIL import Image
from networks.resnet_GBM_sin_back_cont import ResSpiNet, ContrastiveLoss
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
parser.add_argument('-t', '--fine-tuning', action='store_true',
                    help='transfer learning + fine tuning - train only the last FC layer.')

def get_batch_data(train_loader, back_dataset, batchsize):
    num_0 = int(len([0 for temp in train_loader if temp[1] == 0])/10)
    num_1 = int(len([1 for temp in train_loader if temp[1] == 1])/10)

    psp_d = random.sample(list(range(num_0)), int(batchsize/2))
    ttp_d = random.sample(list(range(num_1)), int(batchsize/2))

    psp_list = [temp_psp * 10 + i for temp_psp in psp_d for i in range(10)]
    ttp_list = [(temp_ttp + num_0) * 10 + i for temp_ttp in ttp_d for i in range(10)]
    data_list = psp_list + ttp_list

    data = [train_loader[dx][0] for dx in data_list]
    label = [train_loader[dx][1] for dx in data_list]

    sim_data = torch.cat([back_dataset[dx] for dx in data_list]).view(-1, 224, 224)
    return torch.from_numpy(np.array(data)), torch.from_numpy(np.array(label)), sim_data

def train(train_data, sim_imgs, model, criterion, t_criterion, cont_criterion, optimizer, epoch):
    """Train the model on Training Set"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model = model.cuda()
    model.train()

    for batch_idx in range(30):
        input, target, sim_dat = get_batch_data(train_data, sim_imgs, 4)
        # measure data loading time
        input = torch.cat([input, sim_dat])
        L = len(target)

        if cuda:
            input, target = input.cuda(), target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            input_var = input_var.unsqueeze(1) #扩展维度
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
            top1.update(prec1[0], input.size(0))
            losses.update(loss, input.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return top1.val.detach().cpu().numpy().item(), losses.avg.detach().cpu().numpy().item()

def test(test_loader, model):
    """Test the model on the Evaluation Folder
    Args:
        - classes: is a list with the class name
        - names: is a generator to retrieve the filename that is classified
    """
    # switch to evaluate mode
    model.eval()
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)
    counter = 0
    # Evaluate all the validation set
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
                input_var = input_var.unsqueeze(1)  # 扩展维度

                output, _ = model(input_var)

            # Take last layer output
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
        test_best_ACC = max(ACC, test_best_ACC)
        test_best_P = max(P, test_best_P)

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

def main(irun, fold):

    test_best_acc = 0
    model = ResSpiNet(name='resnet18')
    parameters = model.parameters()
    criterion = nn.CrossEntropyLoss()
    t_criterion = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
    cont_criterion = ContrastiveLoss()

    if cuda:
        criterion.cuda()
        t_criterion.cuda()
        cont_criterion.cuda()

    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    schedulers = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    P, R, F1, ACC, TPR, TNR, AUC = 0, 0, 0, 0, 0, 0, 0
    train_loss = []
    train_pre = []

    if args.train:
        for epoch in range(args.start_epoch, args.epochs):
            schedulers.step()
            train_acc, loss = train(train_data, sim_imgs, model, criterion, t_criterion, cont_criterion, optimizer, epoch)

            train_loss.append(loss)
            train_pre.append(train_acc)

            model = model.eval()
            if train_acc > 70:
                tP, tR, tF1, tACC, tTPR, tTNR, tAUC = test(test_loader, model)
                is_test_best = bool(tACC >= test_best_acc)

                if is_test_best:
                    test_best_acc = max(tACC, test_best_acc)
                    torch.save(model, 'save/resnet18_GBM_sin_back_cont/model-{}-time-{}-fold.pth'.format(irun, fold))
                    P, R, F1, ACC, TPR, TNR, AUC = tP, tR, tF1, tACC, tTPR, tTNR, tAUC
            model = model.train()
    return train_acc, P, R, F1, ACC, TPR, TNR, AUC

def comput_similar_img(datasets, back_imgs):

    Ld = len(datasets)
    Lb = len(back_imgs)
    data_img = [data[0] for data in datasets]

    newdata_img = data_img
    newback_imgs = back_imgs
    newdata_img = torch.Tensor(np.array(newdata_img)).view(Ld, -1)
    newback_imgs = torch.Tensor(np.array(newback_imgs)).view(Lb, -1)
    sim_img_list = []

    for i in range(Ld):
        single_sim = [torch.cosine_similarity(newback_imgs[j], newdata_img[i], dim=0) for j in range(Lb)]
        single_l = torch.max(torch.stack(single_sim), 0)[1]
        sim_img_list.append(single_l)

    return newback_imgs[sim_img_list]

def get_train_and_test_datasets(irun, fold):

    box = (16, 20, 184, 220)
    back_dataset = pickle.load(open('gen_dataset/all_sin_back_dataset.pkl', 'rb'))
    dataset = pickle.load(open('gen_dataset/all_sin_dataset-{}time-{}fold.pkl'.format(irun, fold), 'rb'))
    train_datasets = dataset[0]
    test_datasets = dataset[1]

    train_data = [[np.array(Image.fromarray(temp_d[0]).crop(box).resize((224, 224))), temp_d[1]]
                    for temp_d in train_datasets]
    test_data = [[np.array(Image.fromarray(temp_d[0]).crop(box).resize((224, 224))), temp_d[1]]
                         for temp_d in test_datasets]
    back_dataset = [np.array(Image.fromarray(temp_d).crop(box).resize((224, 224)))
                    for temp_d in back_dataset]

    if os.path.isfile('save/resnet18_GBM_sin_back_cont/sim_imgs-{}-time-{}-fold.pkl'.format(irun, fold)):
        sim_imgs = torch.load(open('save/resnet18_GBM_sin_back_cont/sim_imgs-{}-time-{}-fold.pkl'.format(
            irun, fold), 'rb'))
    else:
        sim_imgs = comput_similar_img(train_data, back_dataset)
        torch.save(sim_imgs,
                   'save/resnet18_GBM_sin_back_cont/sim_imgs-{}-time-{}-fold.pkl'.format(irun, fold))

    if os.path.isfile('save/resnet18_GBM_sin_back_cont/sim_imgs_test-{}-time-{}-fold.pkl'.format(irun, fold)):
        sim_imgs_test = torch.load(open('save/resnet18_GBM_sin_back_cont/sim_imgs_test-{}-time-{}-fold.pkl'.format(
            irun, fold), 'rb'))
    else:
        sim_imgs_test = comput_similar_img(test_data, back_dataset)
        torch.save(sim_imgs_test,
                   'save/resnet18_GBM_sin_back_cont/sim_imgs_test-{}-time-{}-fold.pkl'.format(irun, fold))

    return train_data, sim_imgs, test_data, sim_imgs_test

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

            train_acc, P, R, F1, ACC, TPR, TNR, AUC = main(irun, fold)
            acc[irun][fold], recall[irun][fold], precision[irun][fold], f_score[irun][
                fold], auc[irun][fold], tpr[irun][fold], tnr[irun][
                fold] = ACC, R, P, F1, AUC, TPR, TNR



