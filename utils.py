import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.manifold import TSNE

class AverageMeter(object):
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

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed(args.seed)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def evaluation(loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec,_ = accuracy(outputs, targets, topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            acc.update(prec.item(), inputs.size(0))
    return losses.avg, acc.avg

