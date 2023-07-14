import os
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

from deeplearning.networks import nn_registry

def init_all(config, dataset, logger):
    network = nn_registry[config.model](dataset.channel, dataset.num_classes, dataset.im_size)
    network = network.to(config.device)

    criterion = nn.CrossEntropyLoss().to(config.device)

    # load checkpoint if the path exists
    if os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path)
        network.load_state_dict(checkpoint['state_dict'])
        start_round = checkpoint['comm_round']
        logger.info("=> loaded checkpoint '{}' (comm. round {})"
                    .format(config.checkpoint_path, start_round))
    else:
        start_round = 0

    if os.path.exists(config.user_data_mapping):
        logger.info("Non-IID data distribution")
        with open(config.user_data_mapping, "rb") as fp:
            user_data_mapping = pickle.load(fp)

    # initialize user ids
    num_normal_users = config.total_users - config.num_attackers
    user_ids = np.arange(num_normal_users)
    attacker_ids = np.arange(num_normal_users, config.total_users)

    return network, criterion, user_ids, attacker_ids, user_data_mapping, start_round


def init_optimizer(config, network):
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), 0.1*config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=config.nesterov)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.__dict__[config.optimizer](network.parameters(), config.lr, momentum=config.momentum,
                                                            weight_decay=config.weight_decay, nesterov=config.nesterov)
    return optimizer

def save_checkpoint(state, path):
    torch.save(state, path)


def test(test_loader, network, criterion, config):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy_meter = AverageMeter('Accs', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, data in enumerate(test_loader):
        input, target = data[0], data[1]

        target = target.to(config.device)
        input = input.to(config.device)

        # Compute output
        with torch.no_grad():
            output = network(input)

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        accuracy_meter.update(acc.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    network.no_grad = False
    network.train()

    return accuracy_meter.avg, losses.avg


def validate_and_log(config, model, train_loader, test_loader, criterion, comm_round, best_testacc, logger, record):
    logger.info("-"*50)
    logger.info("Communication Round {:d}".format(comm_round))

    trainacc, train_loss = test(train_loader, model, criterion, config)
    testacc, test_loss = test(test_loader, model, criterion, config)

    # remember best prec@1 and save checkpoint
    is_best = (testacc > best_testacc)
    if is_best:
        best_testacc = testacc
        save_checkpoint({"comm_round": comm_round + 1,
            "state_dict": model.state_dict()},
            # config.output_dir + "/checkpoint_epoch{:d}.pth".format(epoch))
            config.output_dir + "/checkpoint.pth")

    logger.info("Train Acc: {:.2f}, Test acc: {:.2f}".format(trainacc, testacc))

    record["train_acc"].append(trainacc)
    record["test_loss"].append(test_loss)
    record["test_acc"].append(testacc)

    return best_testacc  

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


