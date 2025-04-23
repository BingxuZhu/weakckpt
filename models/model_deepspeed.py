import argparse
import os
import shutil
import time
import math
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import logging


import deepspeed
from weakckpt.config import WeakCkptConfig
from weakckpt.deepspeed_plugin import WeakCkptDeepSpeedPlugin
from torchvision import datasets, transforms
from mymodel import MyModel

# 添加检查点相关模块路径
sys.path.append('/ssd/home/scw6fet/Dan_1/dataset/')
from cf_checkpoint import CFCheckpoint
from cf_manager import CFManager, CFMode
from cf_iterator import CFIterator


#def train():
 # 
 #
 #   # 数据准备
 #   transform = transforms.Compose([...])
 #   dataset = datasets.ImageNet('/data/imagenet', transform=transform)
 #   loader = engine.train_dataloader(dataset)
 #
 #   for step, batch in enumerate(loader):
 #       inputs, labels = batch
 #       loss = engine(inputs, labels)
 #       engine.backward(loss)
 #       engine.step()




model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="/ssd/home/scw6fet/Dan_1/dataset", type=str,
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--nopin', action='store_false', help='Use this argument to disable memory pinning')
parser.add_argument('--resume', default=False, action='store_true',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--steps_per_run", default=-1, type=int)
parser.add_argument("--classes", default=2, type=int)
parser.add_argument("--cache_size", default=0, type=int)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--noeval', action='store_true')
parser.add_argument('--channels-last', type=bool, default=False)
parser.add_argument('--iters', default=-1, type=int, metavar='N', help='Num iters (default: 50)')
parser.add_argument('--chk-freq', default=0, type=int, metavar='N', help='checkpoint frequency')
parser.add_argument('--barrier', action='store_true', default=False)
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--synchronous', action='store_true', default=False)
parser.add_argument('--tic-tac', action='store_true', default=False)
parser.add_argument('--rename', action='store_true', default=False)
parser.add_argument('--tic-tac-len', default=2, type=int)
parser.add_argument('--chk-prefix', type=str, default="./")
parser.add_argument('--checkfreq', action='store_true', default=False)
parser.add_argument('--cf_iterator', action='store_true', default=False)
parser.add_argument('--chk_mode_baseline', action='store_true', default=False)

cudnn.benchmark = True

must_chk = False
compute_time_list = []
data_time_list = []
chk_time_list = []
best_prec1 = 0
args = parser.parse_args()

# 设置模型架构
args.arch = 'resnet18'

if args.test:
    args.epochs = 1
    args.start_epoch = 0
    args.arch = 'resnet50'
    args.batch_size = 64
    args.data = []
    args.prof = True
    args.data.append('/data/imagenet/train-jpeg/')
    args.data.append('/data/imagenet/val-jpeg/')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.local_rank)
    torch.set_printoptions(precision=10)

if not len(args.data):
    raise Exception("error: too few arguments")

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
else:
    args.world_size = 1

args.total_batch_size = args.world_size * args.batch_size

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

# 定义保存检查点函数
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():
    logging.basicConfig(format='%(module)s - %(funcName)s - %(levelname)s - %(message)s', level=logging.INFO)

    start_full = time.time()
    global best_prec1, args

    time_stat = []
    chk_stat = []
    start = time.time()

    args.gpu = 0
    torch.cuda.set_device(args.gpu)

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        
    # DeepSpeed 配置文件路径
    ds_config = 'ds_config.json'
    # 弱一致性检查点配置
    wc_config = WeakCkptConfig(stride_steps=4, max_version_diff=2, base_interval=200)

    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 初始化 DeepSpeed
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config_params=ds_config
    )

    # 创建 WeakCkpt 插件并绑定
    wc_plugin = WeakCkptDeepSpeedPlugin(wc_config, './weakckpt_ds')
    wc_plugin.attach_to_engine(engine)    
    # 创建模型
    print("=> creating model 'resnet18'")
    model = models.resnet18(num_classes=2)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss().cuda()

    args.lr = args.lr * float(args.batch_size * args.world_size) / 256.

    if args.chk_mode_baseline:
        args.chk_mode = CFMode.MANUAL
    else:
        args.chk_mode = CFMode.AUTO

    chk = CFCheckpoint(model=model, optimizer=optimizer)
    cf_manager = CFManager(args.chk_prefix, chk, mode=args.chk_mode)

    args.start_index = 0
    args.steps_so_far = 0
    extra_state = None
    if args.resume:
        extra_state = cf_manager.restore(gpu=args.gpu)
        if extra_state is not None:
            args.start_epoch = extra_state['epoch']
            args.start_index = extra_state['start_index']
            args.steps_so_far = extra_state['steps_so_far']
            print("Populated: epoch :{}, start_idx:{}, steps_so_far:{}".format(args.start_epoch, args.start_index,
                                                                               args.steps_so_far))

    # 数据加载
    traindir = os.path.join(args.data, 'mini_train')
    valdir = os.path.join(args.data, 'mini_val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=not args.nopin, sampler=train_sampler)

    if args.cf_iterator:
        train_loader = CFIterator(
            dataloader=train_loader,
            worker_id=args.local_rank,
            bs=args.batch_size,
            steps_this_epoch=int(args.start_index / args.batch_size),
            epoch=args.start_epoch,
            cf_manager=cf_manager,
            chk_freq=args.chk_freq,
            steps_to_run=args.steps_per_run,
            persist=True,
            dynamic=args.dynamic
        )
        if args.resume:
            train_loader.load_state_dict(extra_state)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=not args.nopin)

    if args.evaluate and not args.noeval:
        validate(val_loader, model, criterion)
        return

    total_time = AverageMeter()
    dur_setup = time.time() - start
    time_stat.append(dur_setup)
    print("Batch size for GPU {} is {}, workers={}".format(args.gpu, args.batch_size, args.workers))

    fname = 'time-split' + str(args.local_rank) + '.csv'
    df = open(fname, 'w+')
    df.write("epoch, iter, dtime, mtime, ftime, ctime, ttime, chktime, tottime\n")

    for epoch in range(args.start_epoch, args.epochs):
        if args.local_rank == 0 and epoch == 0:
            os.system("swapoff -a")
            os.system("free -g")

        start_ep = time.time()
        df.write("\n")

        avg_train_time = train(train_loader, model, criterion, optimizer, epoch, df, cf_manager)
        total_time.update(avg_train_time)
        if args.prof:
            break

        if args.noeval:
            prec1 = 0
        else:
            prec1 = validate(val_loader, model, criterion)

        filename = 'acc-progress-' + str(args.gpu) + '.csv'
        with open(filename, 'a+') as fw:
            fw.write("{},{},{}\n".format(epoch, time.time() - start_ep, prec1))

        # 修改后的检查点保存部分
        chk_st = time.time()
        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n'
                      '##Perf  {1}'.format(prec1, args.total_batch_size / total_time.avg))
        dur_chk = time.time() - chk_st

        if args.cf_iterator and train_loader.exit:
            break

        dur_ep = time.time() - start_ep
        print("EPOCH DURATION = {}".format(dur_ep))
        time_stat.append(dur_ep)
        chk_stat.append(dur_chk)

    if args.local_rank == 0:
        for i in time_stat:
            print("Time_stat : {}".format(i))
        for i in range(0, len(data_time_list)):
            print("Data time : {}\t Compute time : {}\t Chk time : {}".format(data_time_list[i], compute_time_list[i],
                                                                              chk_time_list[i]))

    dur_full = time.time() - start_full
    if args.local_rank == 0:
        print("Total time for all epochs = {}".format(dur_full))
        if cf_manager.chk_process is not None:
            cf_manager.chk_process.join()

def train(train_loader, model, criterion, optimizer, epoch, df, cf_manager):
    batch_time = AverageMeter()
    total_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    global must_chk

    model.train()

    end = time.time()
    dataset_time = compute_time = checkpoint_time = 0
    chk_per_epoch = 0

    for i, (images, target) in enumerate(train_loader):
        target = target.squeeze().cuda().long()
        input_var = Variable(images).cuda(args.gpu, non_blocking=True)
        target_var = Variable(target).cuda(args.gpu, non_blocking=True)
        train_loader_len = len(train_loader)

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        if args.prof:
            if i > 10:
                break

        dtime = time.time() - end
        start_copy = time.time()
        mtime = time.time() - start_copy
        data_time.update(time.time() - end)
        dataset_time += (time.time() - end)
        compute_start = time.time()

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1 = accuracy(output.data, target, topk=(1,))[0]

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), images.size(0))
        top1.update(to_python_float(prec1), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        compute_time += (time.time() - compute_start)
        ctime = time.time() - compute_start

        ttime = time.time() - end
        ch_st = time.time()
        chktime = time.time() - ch_st
        checkpoint_time += chktime

        if args.barrier:
            dist.barrier()
        tottime = time.time() - end
        total_time.update(time.time() - end)
        df.write("{},{},{}\n".format(epoch, i, tottime))

        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, train_loader_len,
                args.total_batch_size / batch_time.val,
                args.total_batch_size / batch_time.avg,
                batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

        if args.iters > 0 and args.iters == i:
            must_chk = False
            break

    data_time_list.append(dataset_time)
    compute_time_list.append(compute_time)
    chk_time_list.append(checkpoint_time)
    return batch_time.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()

    for i, (images, target) in enumerate(val_loader):
        target = target.squeeze().cuda().long()
        input_var = Variable(images).cuda(args.gpu, non_blocking=True)
        target_var = Variable(target).cuda(args.gpu, non_blocking=True)
        val_loader_len = len(val_loader)

        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        prec1 = accuracy(output.data, target, topk=(1,))[0]

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), images.size(0))
        top1.update(to_python_float(prec1), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, val_loader_len,
                args.total_batch_size / batch_time.val,
                args.total_batch_size / batch_time.avg,
                batch_time=batch_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

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

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    factor = epoch // 30
    if epoch >= 80:
        factor = factor + 1
    lr = args.lr * (0.1 ** factor)

    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if args.local_rank == 0 and step % args.print_freq == 0 and step > 1:
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
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

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
