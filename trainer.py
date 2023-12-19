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
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
import matplotlib.pyplot as plt

def save_results(train_losses, val_losses, save_path):
    plt.figure()
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses,   label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

def trainer_synapse(args, model, output_dir):
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    from datasets.dataset_synapse import Synapse_dataset
    logging.basicConfig(filename=output_dir + "/log_without_stopping_criteria_1st_try.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    if args.data_aug == '1':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            # transforms.Normalize(mean=[0.09], std=[0.4]),
            transforms.ToTensor()
        ])
    else:
        transform = None

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    for fold, (train_index, val_index) in enumerate(kf.split(range(len(os.listdir(args.root_path))))):

        iter_num_train = 0
        iter_num_valid = 0
    
        train_dataset  = Synapse_dataset(train_index, base_dir=args.root_path, label_dir=args.label_path, list_dir=args.list_dir, split="train", transform=transform)
        valid_dataset  = Synapse_dataset(val_index  , base_dir=args.root_path, label_dir=args.label_path, list_dir=args.list_dir, split="validation", transform=None)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
        valloader   = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

        train_losses = []
        val_losses = []

        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        writer = SummaryWriter(output_dir + '/log_fold_' + str(fold))

        # outputs = model(image_batch)

        for epoch in range(args.max_epochs):
            model.train()
            running_loss = 0.0

            # Training loop
            for _, sampled_batch in enumerate(trainloader):
                iter_num_train += 1
                image_batch, label_batch = sampled_batch[0], sampled_batch[1]
                print("cnt is ===========================================================", sampled_batch[-1])
                logging.info('number of non_zero elements in label_batch[0, 0, :, :] = %d' % (torch.count_nonzero(label_batch[0, 0, :, :])))
                logging.info('number of non_zero elements in label_batch[1, 0, :, :] = %d' % (torch.count_nonzero(label_batch[1, 0, :, :])))
                logging.info('number of non_zero elements in label_batch[2, 0, :, :] = %d' % (torch.count_nonzero(label_batch[2, 0, :, :])))
                logging.info('number of non_zero elements in label_batch[3, 0, :, :] = %d' % (torch.count_nonzero(label_batch[3, 0, :, :])))
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                image_batch = image_batch.to(torch.float32)
                label_batch = label_batch.to(torch.float32)
                image_batch = image_batch.resize_(batch_size, 448, 448)
                label_batch = label_batch.resize_(batch_size, 448, 448)
                for j in range(batch_size):
                    image_batch[j, :, :] = image_batch[j, :, :]/image_batch.view(image_batch.size(0), -1).max(dim=-1).values[j]
                image_batch = image_batch.unsqueeze(1)
                label_batch = label_batch.unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:, 0, :, :].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice
                # print("TRAIN Dice loss is: ", loss_dice)
                # print("TRAIN CE Loss is: ", loss_ce)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # logging.info('iteration on TRAIN %d : loss : %f' % (iter_num_train, loss.item()))
                # if (iter_num_train == 10):
                #     plt.figure()
                #     for index in range(batch_size):
                #         plt.imshow(label_batch[index, 0, :, :].cpu().detach().numpy())
                #         plt.savefig(os.path.join(output_dir, 'label_batch[index, 0, :, :]_iter10_' + str(index) + '.png'))
                #         logging.info('number of non_zero pixels: %f' % torch.count_nonzero(label_batch[index, 0, :, :]))
                # if (iter_num_train == 400):
                #     plt.figure()
                #     for index in range(batch_size):
                #         plt.imshow(label_batch[index, 0, :, :].cpu().detach().numpy())
                #         plt.savefig(os.path.join(output_dir, 'label_batch[index, 0, :, :]_iter400_' + str(index) + '.png'))
                #         logging.info('number of non_zero pixels: %f' % torch.count_nonzero(label_batch[index, 0, :, :]))
                        # plt.imshow(outputs[index, 1, :, :].cpu().detach().numpy())
                    # plt.plot(epochs, val_losses,   label='Validation Loss')
                    # plt.title('Training and Validation Losses')
                    # plt.xlabel('Epochs')
                    # plt.ylabel('Loss')
                    # plt.legend()
                    # plt.grid(True)
                    # plt.savefig(os.path.join(save_path, 'loss_plot.png'))
                    # plt.close()

            # Calculate average training loss for the epoch
            train_loss = running_loss / len(trainloader)
            train_losses.append(train_loss)

            # Validation loop
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for _, sampled_batch in enumerate(valloader):
                    iter_num_valid += 1
                    image_batch, label_batch = sampled_batch[0], sampled_batch[1]
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    image_batch = image_batch.to(torch.float32)
                    label_batch = label_batch.to(torch.float32)
                    image_batch = image_batch.resize_(batch_size, 448, 448)
                    label_batch = label_batch.resize_(batch_size, 448, 448)
                    for j in range(batch_size):
                        image_batch[j, :, :] = image_batch[j, :, :]/image_batch.view(image_batch.size(0), -1).max(dim=-1).values[j]
                    image_batch = image_batch.unsqueeze(1)
                    label_batch = label_batch.unsqueeze(1)

                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    loss = 0.4 * loss_ce + 0.6 * loss_dice
                    print("VALIDATION Dice loss is: ", loss_dice)
                    print("VALIDATION CE Loss is: ", loss_ce)
                    val_running_loss += loss.item()
                    logging.info('iteration on VALID %d : loss : %f' % (iter_num_valid, loss.item()))

            # Calculate average validation loss for the epoch
            val_loss = val_running_loss / len(valloader)
            val_losses.append(val_loss)

        save_mode_path = os.path.join(output_dir, 'epoch_' + str(epoch) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training/Validation Finished!"


# def trainer_synapse(args, model, snapshot_path):
#     k_folds = 5
#     kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
#     from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
#     logging.basicConfig(filename=snapshot_path + "/log_without_stopping_criteria_1st_try.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu
#     # max_iterations = args.max_iterations
#     if args.data_aug:
#         transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(degrees=15),
#             transforms.Normalize(mean=[0.09], std=[0.4]),
#             transforms.ToTensor()
#         ])
#     else:
#         transform = None

#     for fold, (train_index, val_index) in enumerate(kf.split(range(len(os.listdir(args.root_path))))):
    
#         train_dataset  = Synapse_dataset(train_index, base_dir=args.root_path, label_dir=args.label_dir, list_dir=args.list_dir, split="train", transform=transform)
#         valid_dataset  = Synapse_dataset(val_index  , base_dir=args.root_path, label_dir=args.label_dir, list_dir=args.list_dir, split="validation", transform=None)

#         trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
#         valloader   = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

#         def worker_init_fn(worker_id):
#             random.seed(args.seed + worker_id)

#         if args.n_gpu > 1:
#             model = nn.DataParallel(model)

#         # data_indices = list(range(len(trainloader)))
#         model.train()
#         ce_loss = CrossEntropyLoss()
#         dice_loss = DiceLoss(num_classes)
#         optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#         writer = SummaryWriter(snapshot_path + '/log')
#         iter_num = 0
#         max_epoch = args.max_epochs
#         max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
#         logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
#         best_performance = 0.0
#         iterator = tqdm(range(max_epoch), ncols=70)
#         for epoch_num in iterator:
#             train_losses = []
#             for _, sampled_batch in enumerate(trainloader):
#                 image_batch, label_batch = sampled_batch[0], sampled_batch[1]
#                 image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#                 image_batch = image_batch.to(torch.float32)
#                 label_batch = label_batch.to(torch.float32)
#                 image_batch = image_batch.resize_(batch_size, 448, 448)
#                 label_batch = label_batch.resize_(batch_size, 448, 448)
#                 for j in range(batch_size):
#                     image_batch[j, :, :] = image_batch[j, :, :]/image_batch.view(image_batch.size(0), -1).max(dim=-1).values[j]
#                 image_batch = image_batch.unsqueeze(1)
#                 label_batch = label_batch.unsqueeze(1)

#                 outputs = model(image_batch)
#                 loss_ce = ce_loss(outputs, label_batch[:].long())
#                 loss_dice = dice_loss(outputs, label_batch, softmax=True)
#                 loss = 0.4 * loss_ce + 0.6 * loss_dice
#                 print("Dice loss is: ", loss_dice)
#                 print("CE Loss is: ", loss_ce)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = lr_

#                 train_losses.append(loss.item())
#                 iter_num = iter_num + 1
#                 writer.add_scalar('info/lr', lr_, iter_num)
#                 writer.add_scalar('info/total_loss', loss, iter_num)
#                 writer.add_scalar('info/loss_ce', loss_ce, iter_num)

#                 logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

#                 if iter_num % 20 == 0:
#                     image = image_batch[1, 0, :, :]
#                     image = (image - image.min()) / (image.max() - image.min())
#                     writer.add_image('train/Image', image, iter_num)
#                     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                     labs = label_batch[1, ...].unsqueeze(0) * 50
#                     writer.add_image('train/GroundTruth', labs[0, :, :, :], iter_num)

#             save_interval = 50  # int(max_epoch/6)
#             if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
#                 save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#                 torch.save(model.state_dict(), save_mode_path)
#                 logging.info("save model to {}".format(save_mode_path))

#             if epoch_num >= max_epoch - 1:
#                 save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#                 torch.save(model.state_dict(), save_mode_path)
#                 logging.info("save model to {}".format(save_mode_path))
#                 iterator.close()
#                 break

#     writer.close()
#     return "Training Finished!"
############################################################################################################################################################################
# def trainer_synapse(args, model, snapshot_path):
#     from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
#     logging.basicConfig(filename=snapshot_path + "/log_without_stopping_criteria_1st_try.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu
#     # max_iterations = args.max_iterations

#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(degrees=15),
#         transforms.Normalize(mean=[0.09], std=[0.4]),
#         transforms.ToTensor()
#     ])

#     db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
#                                transform=transform)
#     print("The length of train set is: {}".format(len(db_train)))

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
#                              worker_init_fn=worker_init_fn)
#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)
#     data_indices = list(range(len(trainloader)))
#     model.train()
#     ce_loss = CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     writer = SummaryWriter(snapshot_path + '/log')
#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
#     logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
#     best_performance = 0.0
#     iterator = tqdm(range(max_epoch), ncols=70)
#     for epoch_num in iterator:
#         train_losses = []
#         for _, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch[0], sampled_batch[1]
#             image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#             image_batch = image_batch.to(torch.float32)
#             label_batch = label_batch.to(torch.float32)
#             image_batch = image_batch.resize_(batch_size, 448, 448)
#             label_batch = label_batch.resize_(batch_size, 448, 448)
#             for j in range(batch_size):
#                 image_batch[j, :, :] = image_batch[j, :, :]/image_batch.view(image_batch.size(0), -1).max(dim=-1).values[j]
#             image_batch = image_batch.unsqueeze(1)
#             label_batch = label_batch.unsqueeze(1)

#             outputs = model(image_batch)
#             loss_ce = ce_loss(outputs, label_batch[:].long())
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#             loss = 0.4 * loss_ce + 0.6 * loss_dice
#             print("Dice loss is: ", loss_dice)
#             print("CE Loss is: ", loss_ce)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             train_losses.append(loss.item())
#             iter_num = iter_num + 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)

#             logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs[0, :, :, :], iter_num)

#         save_interval = 50  # int(max_epoch/6)
#         if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
#             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))

#         if epoch_num >= max_epoch - 1:
#             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#             iterator.close()
#             break

#     writer.close()
#     return "Training Finished!"