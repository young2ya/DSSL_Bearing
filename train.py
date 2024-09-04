import time
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *



def MixMatch_train(train_loader, unlabel_loader, model, optimizer, ema_optimizer, criterion, epoch, args):
    losses = AverageMeter()

    labeled_train_iter = iter(train_loader)
    unlabeled_train_iter = iter(unlabel_loader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(train_loader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2),_ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            (inputs_u, inputs_u2),_ = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)

        targets_x = torch.zeros(batch_size, 7).scatter_(1, targets_x.view(-1,1).long(),1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            outputs = model(inputs_u)
            outputs_2 = model(inputs_u2)

            p = (torch.softmax(outputs, dim=1) + torch.softmax(outputs_2, dim=1)) / 2
            pt = p**(1/args.T)

            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)


        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1-l) * input_b
        mixed_target = l * target_a + (1-l) * target_b

        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch, batch_idx, args)
        loss = Lx + w * Lu

        losses.update(loss.item(), inputs_x.size(0))

        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        optimizer.zero_grad()

        return losses.avg

def FixMatch_train(train_loader, unlabel_loader, model, optimizer, ema_optimizer, args):
    losses = AverageMeter()
    mask_probs = AverageMeter()

    labeled_train_iter = iter(train_loader)
    unlabeled_train_iter = iter(unlabel_loader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(train_loader)
            inputs_x, targets_x = next(labeled_train_iter)
        try:
            (inputs_w, inputs_s), _ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            (inputs_w, inputs_s),_ = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)
        inputs = torch.cat((inputs_x, inputs_w, inputs_s)).cuda()
        targets_x = targets_x.cuda()

        logits = model(inputs)

        logits_x = logits[:batch_size]
        logits_w, logits_s = logits[batch_size:].chunk(2)

        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

        pseudo_label = torch.softmax(logits_w.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        Lu = (F.cross_entropy(logits_s, targets_u, reduction='none')*mask).mean()

        loss = Lx + Lu

        losses.update(loss.item(), inputs_x.size(0))

        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        optimizer.zero_grad()

        return losses.avg

def Pseudo_train(train_loader, unlabel_loader, model, optimizer, ema_optimizer, args):
    losses = AverageMeter()
    acces = AverageMeter()
    correct = 0
    total = 0

    labeled_train_iter = iter(train_loader)
    unlabeled_train_iter = iter(unlabel_loader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(train_loader)
            inputs_x, targets_x = next(labeled_train_iter)
        try:
            inputs_u,_ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            inputs_u,_ = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)

        inputs = torch.cat((inputs_x, inputs_u)).cuda()
        targets_x = targets_x.cuda()

        outputs = model(inputs)

        outputs_x = outputs[:batch_size]
        outputs_u = outputs[batch_size:]

        outputs_x = torch.softmax(outputs_x.detach(), dim=-1)
        pseudo_label = torch.softmax(outputs_u.detach(), dim=-1)

        _, pred_x = torch.max(outputs_x, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)

        mask = max_probs.ge(args.threshold).float()

        Lx = F.cross_entropy(outputs_x, targets_x, reduction='mean')
        Lu = (F.cross_entropy(outputs_u, targets_u, reduction='none')*mask).mean()

        loss = Lx + Lu

        total += targets_x.size(0)
        correct += (pred_x == targets_x).sum().item()

        losses.update(loss.item(), inputs.size(0))
        acces.update(100*correct/total, inputs.size(0))

        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        optimizer.zero_grad()

    return losses.avg, acces.avg

def Pseudo_train_new(train_loader, unlabel_loader, model, optimizer, ema_optimizer, criterion, epoch, args):
    losses = AverageMeter()
    acces = AverageMeter()
    correct = 0
    total = 0

    if (epoch > args.T1) and (epoch < args.T2):
        alpha = args.max_alpha * (epoch - args.T1) / (args.T2 - args.T1)
    elif epoch >= args.T2:
        alpha = args.max_alpha
    else:
        alpha = 0

    labeled_train_iter = iter(train_loader)
    unlabeled_train_iter = iter(unlabel_loader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(train_loader)
            inputs_x, targets_x = next(labeled_train_iter)
        try:
            inputs_u,_ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            inputs_u,_ = next(unlabeled_train_iter)

        inputs = torch.cat((inputs_x, inputs_u)).cuda()

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
        inputs_u = inputs_u.cuda()

        outputs_x = model(inputs_x)
        if alpha > 0:
            outputs_u = model(inputs_u)

            outputs_x = torch.softmax(outputs_x.detach(), dim=-1)
            outputs_u = torch.softmax(outputs_u.detach(), dim=-1)

            _, pred_u = torch.max(outputs_u.detach(), 1)

            loss = criterion(outputs_x, targets_x) + alpha * criterion(outputs_u, pred_u)
        else:
            outputs_x = torch.softmax(outputs_x.detach(), dim=-1)

            loss = criterion(outputs_x, targets_x)

        _, pred_x = torch.max(outputs_x, dim=-1)

        total += targets_x.size(0)
        correct += (pred_x == targets_x).sum().item()

        losses.update(loss.item(), inputs.size(0))
        acces.update(100*correct/total, inputs.size(0))

        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        optimizer.zero_grad()

    return losses.avg, acces.avg

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim = 1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

# def linear_rampup(current, rampup_length):
#     if rampup_length == 0:
#         return 1.0
#     else:
#         current = np.clip(current / rampup_length, 0.0, 1.0)
#         return float(current)
#
# class SemiLoss(object):
#     def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, batch_idx, args):
#         probs_u = torch.softmax(outputs_u, dim = 1)
#         epochs = epoch + batch_idx/args.train_iteration
#
#         Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
#         Lu = torch.mean((probs_u - targets_u)**2)
#
#         return Lx, Lu, args.lambda_u * linear_rampup(epochs, args.epochs)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

