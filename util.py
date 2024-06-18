# encoding: utf-8
import os
import numpy as np
from sklearn import metrics
import torch
import random
import pickle
from PIL import Image

def prf(y_true, y_pred, y_score):

    # true positive
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
    # false positive
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    # true negative
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
    # false negative
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)

    TPR = Recall  # 灵敏度
    TNR = TN / (FP + TN) if (FP + TN) != 0. else TN
    AUC = metrics.roc_auc_score(y_true, y_score)

    return Precision, Recall, F1_score, Accuracy, TPR, TNR, AUC

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
    back_dataset = pickle.load(open('/home/data2/liqiang/code/GBM/SupContrast/SupContrast/gen_dataset/all_sin_back_dataset.pkl', 'rb'))
    dataset = pickle.load(open('/home/data2/liqiang/code/GBM/SupContrast/SupContrast/gen_dataset/all_sin_dataset-{}time-{}fold.pkl'.format(irun, fold), 'rb'))
    train_datasets = dataset[0]
    test_datasets = dataset[1]
    train_data = [[np.array(Image.fromarray(temp_d[0]).crop(box).resize((224, 224))), temp_d[1]]
                    for temp_d in train_datasets]
    test_data = [[np.array(Image.fromarray(temp_d[0]).crop(box).resize((224, 224))), temp_d[1]]
                         for temp_d in test_datasets]
    back_dataset = [np.array(Image.fromarray(temp_d).crop(box).resize((224, 224)))
                    for temp_d in back_dataset]

    if os.path.isfile('/home/data2/liqiang/code/GBM/SupContrast/SupContrast/save/resnet18_GBM_sin_back_cont/sim_imgs-{}-time-{}-fold.pkl'.format(irun, fold)):
        sim_imgs = torch.load(open('/home/data2/liqiang/code/GBM/SupContrast/SupContrast/save/resnet18_GBM_sin_back_cont/sim_imgs-{}-time-{}-fold.pkl'.format(
            irun, fold), 'rb'))
    else:
        sim_imgs = comput_similar_img(train_data, back_dataset)
        torch.save(sim_imgs,
                   '/home/data2/liqiang/code/GBM/SupContrast/SupContrast/save/resnet18_GBM_sin_back_cont/sim_imgs-{}-time-{}-fold.pkl'.format(irun, fold))

    if os.path.isfile('/home/data2/liqiang/code/GBM/SupContrast/SupContrast/save/resnet18_GBM_sin_back_cont/sim_imgs_test-{}-time-{}-fold.pkl'.format(irun, fold)):
        sim_imgs_test = torch.load(open('/home/data2/liqiang/code/GBM/SupContrast/SupContrast/save/resnet18_GBM_sin_back_cont/sim_imgs_test-{}-time-{}-fold.pkl'.format(
            irun, fold), 'rb'))
    else:
        sim_imgs_test = comput_similar_img(test_data, back_dataset)
        torch.save(sim_imgs_test,
                   '/home/data2/liqiang/code/GBM/SupContrast/SupContrast/save/resnet18_GBM_sin_back_cont/sim_imgs_test-{}-time-{}-fold.pkl'.format(irun, fold))
    return train_data, sim_imgs, test_data, sim_imgs_test

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