from __future__ import print_function
import argparse
import torch.nn as nn
from models.resnet import ResNet20, ResNet50
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.utils.data.distributed
import horovod.torch as hvd
import os
import wandb
import time

hvd.init()
global_step = 0

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


criterion = nn.CrossEntropyLoss()


def train(log_interval, model, device, train_loader, optimizer, epoch, train_sampler):
    global global_step
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        global_step += 1
        start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        end = time.time()
        if batch_idx % log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                       100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"epoch": epoch, "loss": loss.item(), "total_throughput": hvd.size()*len(data)/(end-start)}, step=global_step)


def test(model, device, test_loader, test_sampler, epoch):
    global global_step
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).float().sum().item()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, test_accuracy))

    wandb.log({
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss,
        "epoch": epoch}, step=global_step)


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--model', default='resnet20')
    parser.add_argument('--data_name', default='imagenet')
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--log-dir", default="log", help='not used, for compatibility')

    parser.add_argument('--optimizer', default='momentum')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform_train)
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)

    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)

    if args.model == 'resnet20':
        model = ResNet20()
    elif args.model == 'resnet50':
        model = ResNet50()
    else:
        raise NotImplemented('model not implemented')

    # Move model to GPU.
    model.to(device)

    # Horovod: scale learning rate by the number of GPUs.
    if args.optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(), weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(), momentum=args.momentum, weight_decay=args.weight_decay)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    if hvd.rank()!=0:
        os.environ['WANDB_MODE'] = 'dryrun'
    wandb_id = os.environ.get('WANDB_ID', None)
    if wandb_id is None:
        wandb.init(config=args)
    else:
        wandb.init(config=args, id=f"{wandb_id}{hvd.rank()}")
    wandb.config.update({'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID', None)})
    # wandb.watch(model)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters())


    for epoch in range(1, args.epochs + 1):
        train(args.log_interval, model, device, train_loader, optimizer, epoch, train_sampler)
        test(model, device, test_loader, test_sampler, epoch)


if __name__ == '__main__':
    main()
