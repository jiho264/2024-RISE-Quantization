import torch, tqdm
import torchvision.models.quantization.resnet as resnet

from torch.quantization import prepare, convert
from torch.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

from torch.quantization import QConfig


def GetDataset(batch_size=64):
    import torchvision
    import torchvision.transforms as transforms

    train_dataset = torchvision.datasets.ImageNet(
        root="data/ImageNet",
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    test_dataset = torchvision.datasets.ImageNet(
        root="data/ImageNet",
        split="val",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, test_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
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


def evaluate(model, data_loader, neval_batches, device):
    model.eval().to(device)
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    cnt = 0
    with torch.no_grad():
        for image, target in tqdm.tqdm(data_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5

    return top1, top5


def main():
    # prepare the model
    model = resnet.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to("cuda")

    _batch_size = 64

    train_loader, test_loader = GetDataset(batch_size=_batch_size)

    # num_eval_batches = len(test_loader)
    num_eval_batches = 50

    top1, top5 = evaluate(
        model, test_loader, neval_batches=num_eval_batches, device="cuda"
    )
    print("Original model evaluation...")
    print(
        "Evaluation accuracy on %d images, %2.2f"
        % (num_eval_batches * _batch_size, top1.avg)
    )

    # %%###############################################################################
    model.fuse_model()

    # model.qconfig = torch.quantization.get_default_qconfig("x86")

    model.qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8, reduce_range=True),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )
    print(model.qconfig)

    prepare(model, inplace=True)

    # %% calibrate the model ############################################################
    calib_len = 16
    print("Calibrating the model...")
    print(f"ㄴ Complited with {_batch_size * calib_len} images...")
    for i, (data, _) in enumerate(train_loader):
        if i > calib_len:
            break
        with torch.no_grad():
            model(data.to("cuda"))

    # %%convert the model ############################################################
    model.to("cpu")
    convert(model, inplace=True)

    # %%evaluate the model ############################################################

    print("Quantized model evaluation...")
    top1, top5 = evaluate(
        model, test_loader, neval_batches=num_eval_batches, device="cpu"
    )
    print(
        "Evaluation accuracy on %d images, %2.2f"
        % (num_eval_batches * _batch_size, top1.avg)
    )


if __name__ == "__main__":
    print("ResNet50 quantization with pytorch tutorial...")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
