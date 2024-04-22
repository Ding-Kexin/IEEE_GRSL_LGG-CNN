import os
import sys
import time
import numpy as np
import random
import torch.nn.functional as F
import torch
import torch.utils.data as dataf
import torch.nn as nn
from scipy import io
import HyperX
import losses
import metric
from network import GlobalLocalGatedConvNet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # 设置随机数种子


def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = torch.sum(predict==label)*1.0/n
    correct_sum = torch.zeros((max(label)+1))
    reali = torch.zeros((max(label)+1))
    predicti = torch.zeros((max(label)+1))
    CA = torch.zeros((max(label)+1))
    for i in range(0, max(label) + 1):
        correct_sum[i] = torch.sum(label[np.where(predict == i)] == i)
        reali[i] = torch.sum(label == i)
        predicti[i] = torch.sum(predict == i)
        CA[i] = correct_sum[i] / reali[i]

    Kappa = (n * torch.sum(correct_sum) - torch.sum(reali * predicti)) * 1.0 / (n * n - torch.sum(reali * predicti))
    AA = torch.mean(CA)
    return OA, Kappa, CA, AA


def show_calaError(val_predict_labels, val_true_labels):
   val_predict_labels = torch.squeeze(val_predict_labels)
   val_true_labels = torch.squeeze(val_true_labels)
   OA, Kappa, CA, AA = CalAccuracy(val_predict_labels, val_true_labels)
   # ic(OA, Kappa, CA, AA)
   print("OA: %f, Kappa: %f,  AA: %f" % (OA, Kappa, AA))
   print("CA: ",)
   print(CA)
   return OA, Kappa, CA, AA


def text_create(name):
  desktop_path = r"/home/fuwei/wxt/dkx/records/PaviaU//"
  # 新创建的txt文件的存放路径
  full_path = desktop_path + name + '.txt' # 也可以创建一个.doc的word文档
  file = open(full_path, 'w')


def train_10times():
    net_name = "GlobalLocalGatedConvNet"
    DataPath = './Data/PaviaU/PaviaU.mat'
    Data = io.loadmat(DataPath)
    Data = Data['paviaU']
    Data = Data.astype(np.float32)
    num_train_1 = [5, 10, 15, 20, 25]
    for k in range(len(num_train_1)):
        OA_ALL = []
        AA_ALL = []
        CA_ALL = []
        KAPPA_ALL = []
        for number in range(0, 10):
            patchsize = 24  # 30  例如该值取24，则patch块最后的形状是25×25
            batchsize = 16  # 64  200  # select from [16, 32, 64, 128], the best is 64
            EPOCH = 200
            LR = 0.0001
            setup_seed(20)
            # -------------------------------------------------------------------------------
            # prepare data
            TRPath = './Data/PaviaU/PaviaU_eachClass_%d_train_%d.mat' % (num_train_1[k], number)
            TSPath = './Data/PaviaU/PaviaU_eachClass_%d_test_%d.mat' % (num_train_1[k], number)
            TrLabel = io.loadmat(TRPath)
            TsLabel = io.loadmat(TSPath)
            TrLabel = TrLabel['data']
            TsLabel = TsLabel['data']
            pad_width = np.floor(patchsize / 2)
            pad_width = np.int(pad_width)
            [m, n, l] = np.shape(Data)  # m=610 n=340 l=103
            class_number = np.max(TsLabel)
            # 数据归一化
            for i in range(l):
                Data[:, :, i] = (Data[:, :, i] - Data[:, :, i].min()) / (Data[:, :, i].max() - Data[:, :, i].min())
            x = Data
            # 数据边界填充，准备分割数据块
            temp = x[:, :, 0]
            pad_width = np.floor(patchsize / 2)
            pad_width = np.int(pad_width)
            temp2 = np.pad(temp, pad_width, 'symmetric')
            [m2, n2] = temp2.shape
            x2 = np.empty((m2, n2, l), dtype='float32')  # 待填充
            for i in range(l):
                temp = x[:, :, i]
                pad_width = np.floor(patchsize / 2)
                pad_width = np.int(pad_width)
                temp2 = np.pad(temp, pad_width, 'symmetric')
                x2[:, :, i] = temp2
            [ind1, ind2] = np.where(TsLabel != 0)  # 得到不为0的值的的坐标
            TestNum = len(ind1)
            TestPatch = np.empty((TestNum, l, patchsize + 1, patchsize + 1), dtype='float32')  # (42596,103,31,31)
            TestLabel = np.empty(TestNum)  # (42596)
            ind3 = ind1 + pad_width
            ind4 = ind2 + pad_width
            for i in range(len(ind1)):
                patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                        (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
                patch = np.reshape(patch, ((patchsize + 1) * (patchsize + 1), l))
                patch = np.transpose(patch)
                patch = np.reshape(patch, (l, (patchsize + 1), (patchsize + 1)))
                TestPatch[i, :, :, :] = patch
                patchlabel = TsLabel[ind1[i], ind2[i]]
                TestLabel[i] = patchlabel
            train_dataset = HyperX.dataLoad(x2, TrLabel, patch_size=patchsize, center_pixel=True,
                                            flip_augmentation=True)
            train_loader = dataf.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
            TestPatch = torch.from_numpy(TestPatch)
            TestLabel = torch.from_numpy(TestLabel) - 1
            TestLabel = TestLabel.long()
            Classes = len(np.unique(TrLabel)) - 1
            filename = 'PaviaU_seed20_patchsize25_EachClass%d_%d' % (num_train_1[k], number)
            text_create(filename)
            output = sys.stdout
            outputfile = open(r"/home/fuwei/wxt/dkx/records/PaviaU//" + filename + '.txt', 'a')
            sys.stdout = outputfile
            cnn = GlobalLocalGatedConvNet(classes=class_number, HSI_Data_Shape_H=m, HSI_Data_Shape_W=n,
                                          HSI_Data_Shape_C=l,
                                          patch_size=patchsize + 1)
            print('net_name:', net_name)
            cnn.cuda()
            total = sum([param.nelement() for param in cnn.parameters()])
            print("Number of parameter: %.2fM" % (total / 1e6))
            optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
            loss_fun1 = losses.ConstractiveLoss()  # the target label is not one-hotted
            loss_fun2 = nn.CrossEntropyLoss()
            show_feature_map = []
            BestAcc = 0
            tic1 = time.time()
            # train and test the designed model
            for epoch in range(EPOCH):
                for step, (images, points, labels) in enumerate(
                        train_loader):  # gives batch data, normalize x when iterate train_loader

                    # move train data to GPU
                    images = images.cuda()  # 输入
                    points = points.cuda()
                    labels = labels.cuda()  # 标签
                    bsz = labels.shape[0]

                    features3, output = cnn(images, points)  # fake_img:重构结果，output:分类结果

                    # 对比损失
                    # contrastive_loss = loss_fun1(features3, labels)  # 将其注释
                    # contrastive_loss = loss_fun1(s_cat, s_label_cat)
                    # 分类损失
                    classifier_loss = loss_fun2(output, labels)  # 交叉熵，分类误差
                    # 总损失
                    total_loss = classifier_loss
                    # print("epoch",epoch)
                    # print("step",step)
                    # print("classifier_loss", classifier_loss.data.cpu().numpy())
                    # print("contrastive_loss", contrastive_loss.data.cpu().numpy())

                    # total_loss = classifier_loss + contrastive_loss

                    cnn.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    if step % 50 == 0:  # 迭代50次，测试
                        cnn.eval()
                        pred_y = np.empty((len(TestLabel)), dtype='float32')
                        number = len(TestLabel) // 100
                        # 测试为100个bach
                        for i in range(number):
                            temp = TestPatch[i * 100:(i + 1) * 100, :, :, :]
                            temp_points = temp[:, :, pad_width, pad_width]
                            temp = temp.cuda()
                            temp_points = temp_points.cuda()
                            _, temp2 = cnn(temp, temp_points)
                            # _,_, temp2 = cnn(temp, temp_points)
                            temp3 = torch.max(temp2, 1)[1].squeeze()
                            pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
                            del temp, temp2, temp3, _, temp_points
                        # 不足100个的情况
                        if (i + 1) * 100 < len(TestLabel):
                            temp = TestPatch[(i + 1) * 100:len(TestLabel), :, :, :]
                            temp_points = temp[:, :, pad_width, pad_width]
                            temp_points = temp_points.cuda()
                            temp = temp.cuda()
                            _, temp2 = cnn(temp, temp_points)
                            # _, _, temp2 = cnn(temp, temp_points)
                            temp3 = torch.max(temp2, 1)[1].squeeze()
                            pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
                            del temp, temp2, temp3, _, temp_points

                        pred_y = torch.from_numpy(pred_y).long()
                        accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
                        # print('Epoch: ', epoch, '| classify loss: %.6f' % classifier_loss.data.cpu().numpy(),
                        #       '| contrastive loss: %.6f' % contrastive_loss.data.cpu().numpy(),
                        #       '| test accuracy（OA）: %.6f' % accuracy)
                        print('Epoch: ', epoch, '| classify loss: %.6f' % classifier_loss.data.cpu().numpy(),
                              '| test accuracy（OA）: %.6f' % accuracy)
                        # save the parameters in network
                        if accuracy > BestAcc:
                            torch.save(cnn.state_dict(), 'net_params_myNet_HS.pkl')
                            BestAcc = accuracy
                        cnn.train()

            # # test each class accuracy
            cnn.load_state_dict(torch.load('net_params_myNet_HS.pkl'))
            cnn.eval()

            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 100
            for i in range(number):
                temp = TestPatch[i * 100:(i + 1) * 100, :, :, :]
                temp_points = temp[:, :, pad_width, pad_width]
                temp = temp.cuda()
                temp_points = temp_points.cuda()
                _, temp2 = cnn(temp, temp_points)
                # _, _, temp2 = cnn(temp, temp_points)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
                del temp, temp2, temp3, _, temp_points
            # 不足100个的情况
            if (i + 1) * 100 < len(TestLabel):
                temp = TestPatch[(i + 1) * 100:len(TestLabel), :, :, :]
                temp_points = temp[:, :, pad_width, pad_width]
                temp_points = temp_points.cuda()
                temp = temp.cuda()
                _, temp2 = cnn(temp, temp_points)
                # _, _, temp2 = cnn(temp, temp_points)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
                del temp, temp2, temp3, _, temp_points

            # 评价指标
            pred_y = torch.from_numpy(pred_y).long()
            Classes = np.unique(TestLabel)
            EachAcc = np.empty(len(Classes))
            AA = 0.0
            for i in range(len(Classes)):
                cla = Classes[i]
                right = 0
                sum_new = 0
                for j in range(len(TestLabel)):
                    if TestLabel[j] == cla:
                        sum_new += 1
                    if TestLabel[j] == cla and pred_y[j] == cla:
                        right += 1
                EachAcc[i] = right.__float__() / sum_new.__float__()
                AA += EachAcc[i]

            print('-------------------')
            for i in range(len(EachAcc)):
                # print('|第%d类精度：' % (i + 1), '%.2f|' % (EachAcc[i] * 100))
                print('%.2f' % (EachAcc[i] * 100))
                # print('-------------------')
            AA *= 100 / len(Classes)

            results = metric.metrics(pred_y, TestLabel, n_classes=len(Classes))
            # print('test accuracy（OA）: %.2f ' % results["Accuracy"], 'AA : %.2f ' % AA, 'Kappa : %.2f ' % results["Kappa"])
            print('%.2f' % results["Accuracy"])
            print('%.2f' % AA)
            print('%.2f' % results["Kappa"])
            # print('confusion matrix :')
            # print(results["Confusion matrix"])
            pred_y.type(torch.FloatTensor)
            TestLabel.type(torch.FloatTensor)
            print("***********************Train and test result record***************************")
            OA, Kappa, CA, AA = show_calaError(pred_y, TestLabel)
            print("***********************Train and test result record***************************")
            toc = time.time()
            time_all = toc - tic1
            print('All process complete in {:.0f}m {:.0f}s'.format(time_all / 60, time_all % 60))
            print("**************************************************")
            outputfile.close()
            OA_ALL.append(OA)
            AA_ALL.append(AA)
            CA_ALL.append(CA)
            KAPPA_ALL.append(Kappa)
        filename = 'PaviaU_seed20_patchsize25_EachClass%d_mean' % (num_train_1[k])
        text_create(filename)
        output = sys.stdout
        outputfile = open(r"/home/fuwei/wxt/dkx/records/PaviaU//" + filename + '.txt', 'a')
        sys.stdout = outputfile
        print("***********************Train raw***************************")
        print('OA_10times:', OA_ALL)
        print('OA_10times_mean:', np.mean(OA_ALL))
        print('AA_10times:', AA_ALL)
        print('AA_10times_mean:', np.mean(AA_ALL))
        print('KAPPA_10times:', KAPPA_ALL)
        print('KAPPA_10times_mean:', np.mean(KAPPA_ALL))
        print(CA_ALL)
        outputfile.close()


if __name__ == '__main__':

    train_10times()