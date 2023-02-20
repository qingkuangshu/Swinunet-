import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, BinaryDiceLoss
from torchvision import transforms
from utils import test_single_volume
from pytorch_toolbelt import losses as L
from datasets.dataset_synapse import ImageFolder
from metrics.iou import iou_pytorch
from time import time
from framework import MyFrame
from dice_bce_loss import Dice_bce_loss
from torch.autograd import Variable as V
from metrics.iou import IOUMetric

from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


def trainer_synapse(args, model, snapshot_path):
    # from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    # 换成自己的
    db_train = ImageFolder(args.root_path, mode='train')
    db_train1 = ImageFolder(args.root_path, mode='val')
    db_train2 = ImageFolder(args.root_path, mode='test')

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
    #                          worker_init_fn=worker_init_fn)

    # 换成自己的
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # ce_loss = CrossEntropyLoss()
    # bce_loss = nn.BCELoss()
    # dice_loss = DiceLoss(num_classes)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = BinaryDiceLoss(mode='binary')

    loss_fn = L.JointLoss(first=dice_loss, second=bce_loss, first_weight=0.5, second_weight=0.5).cuda()

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-3)
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2,  # T_0就是初始restart的epoch数目
        T_mult=2,  # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
        eta_min=1e-6  # 最低学习率
    )
    """
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    tic = time()
    device = torch.device('cuda:0')
    solver = MyFrame(model, Dice_bce_loss, 0.001)       #Model=model

    valloader = DataLoader(
        db_train1,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn)
    testloader = DataLoader(
        db_train2,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn)
    
    model = SwinTransformerSys()
    print(model.count_parameters())

    for epoch_num in iterator:
        valloader_num = iter(valloader)
        test_mean_iou = 0
        for val_img, val_mask in tqdm(valloader_num, ncols=70, total=len(valloader_num)):
            val_img, val_mask = val_img.to(device), val_mask.cpu()
            val_mask[np.where(val_mask > 0)] = 1
            val_mask = val_mask.squeeze(0)
            predict = solver.test_one_img(val_img)
     
            predict_temp = torch.from_numpy(predict).unsqueeze(0)
            predict_use = V(predict_temp.type(torch.FloatTensor), volatile=True)
            val_use = V(val_mask.type(torch.FloatTensor), volatile=True)
          

            predict_use = predict_use.squeeze(0)
            predict_use = predict_use.unsqueeze(1)
            predict_use[predict_use >= 0.5] = 1
            predict_use[predict_use < 0.5] = 0
            predict_use = predict_use.type(torch.LongTensor)
            val_use = val_use.squeeze(1).type(torch.LongTensor)
            test_mean_iou += iou_pytorch(predict_use, val_use)
        batch_iou = test_mean_iou / len(valloader_num)
     
        label_list = []
        pre_list = []
        for img, mask in tqdm(valloader,ncols=70,total=len(valloader)):
            img, mask = img.to(device), mask.cpu()
            mask[mask>0] = 1
            mask = mask.squeeze(0)
            mask = mask.cpu().numpy()
            # mask = mask.astype(np.int)
            mask = np.uint8(mask)
            label_list.append(mask)

            #img = img.squeeze(0)
            #img = img.cpu().numpy()
            pre = solver.test_one_img(img)
            pre[pre>=4.0] = 255
            pre[pre<4.0] = 0

            # pre = pre.astype(np.int)
            pre = np.uint8(pre)
            pre[pre>0] = 1
            pre_list.append(pre)
        el = IOUMetric(2)
        acc = el.evaluate(pre_list, label_list)
    
        trainloader_num = iter(trainloader)
        train_mean_iou = 0
        for train_img, train_mask in tqdm(trainloader_num, ncols=70, total=len(trainloader_num)):
            train_img, train_mask = train_img.to(device), train_mask.cpu()
            train_mask[np.where(train_mask > 0)] = 1
            train_mask = train_mask.squeeze(0)
            predict = solver.test_one_img(train_img)

            predict_temp = torch.from_numpy(predict).unsqueeze(0)
            predict_use = V(predict_temp.type(torch.FloatTensor), volatile=True)
            train_use = V(train_mask.type(torch.FloatTensor), volatile=True)

            predict_use = predict_use.squeeze(0)
            predict_use = predict_use.unsqueeze(1)
            predict_use[predict_use >= 0.5] = 1
            predict_use[predict_use < 0.5] = 0
            predict_use = predict_use.type(torch.LongTensor)
            train_use = train_use.squeeze(1).type(torch.LongTensor)
            train_mean_iou += iou_pytorch(predict_use, train_use)
        batch_iou1 = train_mean_iou / len(trainloader_num)

        label_list1 = []
        pre_list1 = []
        for img, mask in tqdm(trainloader,ncols=70,total=len(trainloader)):
            img, mask = img.to(device), mask.cpu()
            mask[mask>0] = 1
            mask = mask.squeeze(0)
            mask = mask.cpu().numpy()
            # mask = mask.astype(np.int)
            mask = np.uint8(mask)
            label_list1.append(mask)

            #img = img.squeeze(0)
            #img = img.cpu().numpy()
            pre = solver.test_one_img(img)
            pre[pre>=4.0] = 255
            pre[pre<4.0] = 0

            # pre = pre.astype(np.int)
            pre = np.uint8(pre)
            pre[pre>0] = 1
            pre_list1.append(pre)
        el1 = IOUMetric(2)
        acc1 = el1.evaluate(pre_list1, label_list1)

        

        for image_batch, label_batch in trainloader:
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            outputs = torch.squeeze(outputs)
            label_batch = torch.squeeze(label_batch)
            loss = loss_fn(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item())

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                #print(image.shape)
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)

                # outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                outputs = torch.sigmoid(outputs)
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                temp = torch.unsqueeze(outputs[0], 0)
                writer.add_image('train/Prediction', temp * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50


        logging.info('iteration : %d  train_loss : %f train_iou : %f time : %d  val_iou : %f val_acc : %f train_acc : %f' % (
        iter_num, loss.item(),batch_iou1.item(),int(time() - tic), batch_iou.item(),acc,acc1))
        #print('train_acc: ',train_acc)
        #print('val_acc: ',acc)
        #print('train_acc: ',acc1)
        save_interval = 10  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

        #scheduler.step()

    writer.close()

    return "Training Finished!"
