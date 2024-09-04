import logging
import os
import argparse
import random

import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from models import *
from train import *
from utils import *
from data_loader import *

logger = logging.getLogger(__name__)
best_acc = 0

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def fix_interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def new_main():
    parser = argparse.ArgumentParser(description='SSL model Training')

    parser.add_argument('--gpu-id', default='0', type=int)
    parser.add_argument('--num-workers', default=0, type=int)

    parser.add_argument('--method', default='fixmatch', type=str)
    parser.add_argument('--model', default='wrn', type=str)

    parser.add_argument('--ex-epochs', default=10, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--wdecay', default=5e-4, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)

    parser.add_argument('--num-labeled', type=int, default=5)
    parser.add_argument('--train-batch-size', default=16)
    parser.add_argument('--batch-size', default=128)

    parser.add_argument('--train-iteration', default=4, type=int)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=75, type=float)
    parser.add_argument('--threshold', default=0.8, type=float)
    parser.add_argument('--patience-limit', '--pl', default=20, type=int)

    parser.add_argument('--max-alpha', default=3)
    parser.add_argument('--T1', default=10, type=int)
    parser.add_argument('--T2', default=60, type=int)
    parser.add_argument('--nesterov', action='store_true', default=True)

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--use-ema', action='store_true', default=True)

    parser.add_argument('--mode', default='client')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=65361)

    args = parser.parse_args()
    Test_acc = np.zeros((args.ex_epochs, 1))
    data_path = 'D:/SLRA_Bearing_data/data_stft_64'
    path_list = [os.path.join(data_path, f_name) for f_name in os.listdir(data_path)]

    for i in range(args.ex_epochs):
        global best_acc

        if args.local_rank == -1:
            device = torch.device('cuda', args.gpu_id)
            args.world_size = 1
            args.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda', args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            args.world_size = torch.distributed.get_world_size()
            args.n_gpu = 1

        args.device = device

        if args.seed is not None:
            set_seed(args)

        print(f'repeat experiment : {i + 1}')
        print('data loading...')

        # 1. Load Dataset
        transforms_w = transforms.Compose([Noise_w(), ToTensor()])
        transforms_s = transforms.Compose([Noise_s(), ToTensor()])
        trainset, testset, valset, unlabelset = Data_Loader_FixMatch(path_list, transforms_w, transforms_s)

        # 2. Data Loader
        train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        unlabel_loader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        if args.local_rank == 0:
            torch.distributed.barrier()

        # 3. Create Model
        print('creating model...')
        model = create_model(args)

        if args.local_rank == 0:
            torch.distributed.barrier()

        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if args.use_ema:
            ema_model = ModelEMA(args, model, args.ema_decay)

        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                output_device=args.local_rank, find_unused_parameters=True)

        patience_check = 0

        FixMatch_train(args, train_loader, unlabel_loader,val_loader,  model, optimizer, ema_model, scheduler, patience_check)

            # val_loss, val_acc = evaluation(val_loader, model, criterion)

            # if val_acc <= best_acc:
            #     patience_check += 1
            #     if patience_check >= args.patience_limit:
            #         break
            # else:
            #     best_acc = deepcopy(val_acc)
            #     patience_check = 0
            #     torch.save(model.state_dict(), f'D:\\PycharmProjects\\SSL_Bearing\\result\\{args.model}_{args.method}.pth')

            # print(f'Train loss: {train_loss}, Train acc: {train_acc}, Val acc: {val_acc} {patience_check}')

        model.load_state_dict(torch.load(f'./result/{args.model}_{args.method}.pth'))
        test_loss, test_acc = test(args, test_loader, model)

        print(f'Test acc: {test_acc}')

        Test_acc[i][0] = test_acc

        tsne(args, test_loader, model, i)

        if args.seed is not None:
            args.seed = args.seed + 1

    Test_acc = pd.DataFrame(Test_acc)
    Test_acc.to_csv(f'./result/{args.model}_{args.method}_{args.num_labeled}_result.csv')
    print('end')

def FixMatch_train(args, train_loader, unlabeled_loader, test_loader, model, optimizer, ema_model, scheduler, patience):
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        train_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(train_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.train_iteration),
                         disable=args.local_rank not in [-1,0])
        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    train_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(train_loader)
                inputs_x, targets_x = next(labeled_iter)

            targets_x = targets_x.long()

            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.size[0]
            inputs = fix_interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            losses.update(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Mask: {mask:.2f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.train_iteration,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

            if test_acc <= best_acc:
                patience += 1
                if patience >= args.patience_limit:
                    break
            else:
                best_acc = deepcopy(test_acc)
                patience = 0
                torch.save(model.state_dict(), f'./result/{args.model}_{args.method}.pth')

def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader =  tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1,5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))

    return losses.avg, top1.avg

def tsne(args, test_loader, model, repeat):
    features = []
    targets_all = []

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1,0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)

            features.append(outputs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

        features = np.concatenate(features)
        targets_all = np.concatenate(targets_all)

        tsne = TSNE(n_components=2, random_state=args.seed)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(10,10))
        bearing_classes = ['N', 'SB', 'SI', 'SO', 'WB', 'WI', 'WO']
        colors = plt.cm.tab10(np.linspace(0,1,len(bearing_classes)))

        for i, class_name in enumerate(bearing_classes):
            indices = targets_all == i
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                        color=colors[i], label=class_name, alpha=0.6)

        plt.legend(loc='best')
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimention 2")
        plt.title(f"TSNE of {args.method}")
        plt.savefig(f"./plot/tsne_{args.method}_{args.num_labeled}_{repeat}.png")
        plt.close()

if __name__ == '__main__':
    new_main()
