from __future__ import print_function
import argparse
import torch.nn as nn
from models.resnet import ResNet20, ResNet50
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.utils.data.distributed
import horovod.torch as hvd

hvd.init()
if hvd.rank() == 0:
    import wandb


def get_compressor(args):
    tags = [args.compression_method]
    if args.memory:
        tags.append('mem')
        memory = hvd.compression.ResidualMemory()
    else:
        memory = hvd.compression.NoneMemory()
    tags.append(args.communication_method)

    if args.compression_method == 'none':
        compressor = hvd.compression.NoneCompressor()
    elif args.compression_method == 'fp16':
        compressor = hvd.compression.FP16Compressor(memory=memory)
    elif args.compression_method == 'randomk':
        tags.append(f'ratio={args.compress_ratio}')
        compressor = hvd.compression.RandomKCompressor(compress_ratio=args.compress_ratio, memory=memory)
    elif args.compression_method == 'topk':
        tags.append(f'ratio={args.compress_ratio}')
        compressor = hvd.compression.TopKCompressor(compress_ratio=args.compress_ratio, memory=memory)
    elif args.compression_method == 'threshold':
        tags.append(f'threshold={args.threshold}')
        compressor = hvd.compression.ThresholdCompressor(threshold_val=args.threshold, memory=memory)
    elif args.compression_method == 'signsgd':
        compressor = hvd.compression.SignSGDCompressor(memory=memory)
    elif args.compression_method == 'efsignsgd':
        tags.append(f'lr={args.efsgdlr}')
        compressor = hvd.compression.EFSignSGDCompressor(lr=args.efsgdlr)
    elif args.compression_method == 'signum':
        tags.append(f'momentum={args.momentum}')
        compressor = hvd.compression.SignumCompressor(momentum=args.momentum, memory=memory)
    elif args.compression_method == 'qsgd':
        tags.append(f'quantum={args.quantum}')
        compressor = hvd.compression.QSGDCompressor(quantum_num=args.quantum, memory=memory)
    elif args.compression_method == 'onebit':
        compressor = hvd.compression.OneBitCompressor(memory=memory)
    elif args.compression_method == 'terngrad':
        compressor = hvd.compression.TernGradCompressor(memory=memory)
    elif args.compression_method == 'dgc':
        tags.append(f'ratio={args.compress_ratio}')
        tags.append(f'momentum={args.momentum}')
        tags.append(f'clipping={args.clipping}')
        compressor = hvd.compression.DgcCompressor(compress_ratio=args.compress_ratio, momentum=args.momentum, gradient_clipping=args.clipping)
    elif args.compression_method == 'powersgd':
        compressor = hvd.compression.PowerSGDCompressor(use_memory=args.memory)
    else:
        # 'adaq'
        # 'adas'
        # 'u8bit'
        # 'natural'
        # 'sketch'
        raise NotImplemented('compression not selected')
    return compressor, tags


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


criterion = nn.CrossEntropyLoss()


def train(log_interval, model, device, train_loader, optimizer, epoch, train_sampler):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, test_sampler):
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
            "Test Loss": test_loss})


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

    parser.add_argument('--optimizer', default='momentum')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--communication_method', default='allreduce', choices=('allreduce', 'allgather', 'broadcast'))
    parser.add_argument('--compression_method', default='none')
    parser.add_argument('--threshold', type=int, default=256)
    parser.add_argument('--quantum', type=int, default=256)
    parser.add_argument('--efsgdlr', type=float, default=0.1)
    parser.add_argument('--compress_ratio', type=float, default=0.01)
    parser.add_argument('--clipping', action='store_true', default=False)
    parser.add_argument('--memory', action='store_true', default=False)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Horovod: initialize library.
    # hvd.init()

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
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = datasets.CIFAR10('data', train=False, transform=transform_test)

    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)

    model = ResNet20()
    # if model == 'resnet20':
    #     net = ResNet20()
    # elif model == 'resnet50':
    #     net = ResNet50()
    # else:
    #     raise NotImplemented('model not implemented')

    # Move model to GPU.
    model.to(device)

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(), momentum=args.momentum, weight_decay=args.weight_decay)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compressor, tags = get_compressor(args)
    name = '_'.join(tags)

    if hvd.rank() == 0:
        wandb.init(project="cifar10-pytorch", id=name, tags=tags)
        wandb.config.update(args)
        wandb.watch(model)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compressor=compressor,
                                         communication=args.communication_method)


    for epoch in range(1, args.epochs + 1):
        train(args.log_interval, model, device, train_loader, optimizer, epoch, train_sampler)
        test(model, device, test_loader, test_sampler)


if __name__ == '__main__':
    main()
