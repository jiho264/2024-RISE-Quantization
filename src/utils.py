import torch, time, os, tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torchvision.models.resnet as resnet


layers_mapping = {
    18: resnet18,
    34: resnet34,
    50: resnet50,
    101: resnet101,
    152: resnet152,
}
pretrained_weights_mapping = {
    18: resnet.ResNet18_Weights.DEFAULT,
    34: resnet.ResNet34_Weights.DEFAULT,
    50: resnet.ResNet50_Weights.DEFAULT,
    101: resnet.ResNet101_Weights.DEFAULT,
    152: resnet.ResNet152_Weights.DEFAULT,
}


def GetDataset(
    dataset_name: str,
    device: str,
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 8,
) -> tuple:
    """Get the dataset and dataloader

    Args:
        dataset_name (str): {"CIFAR10", "CIFAR100", "ImageNet"}
        device (str): {"cpu", "cuda:0", ...}
        root (str, optional): The root folder of datasets. Defaults to "data".
        batch_size (int, optional): Size of Batch. Defaults to 128.
        num_workers (int, optional): Num of Workers about Dataloader. Defaults to 8.

    Returns:
        tuple: trainloader, testloader
    """
    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        if dataset_name == "CIFAR10":
            train_dataset = datasets.CIFAR10(
                root=root, train=True, download=False, transform=transform_train
            )
            test_dataset = datasets.CIFAR10(
                root=root, train=False, download=False, transform=transform_test
            )
        elif dataset_name == "CIFAR100":
            train_dataset = datasets.CIFAR100(
                root=root, train=True, download=False, transform=transform_train
            )
            test_dataset = datasets.CIFAR100(
                root=root, train=False, download=False, transform=transform_test
            )
    # Load the CIFAR-100 dataset
    elif dataset_name == "ImageNet":
        train_dataset = datasets.ImageNet(
            root=root + "/ImageNet",
            split="train",
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        test_dataset = datasets.ImageNet(
            root=root + "/ImageNet",
            split="val",
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
    if device == "cpu":
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=device,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=device,
        )
    return train_loader, test_loader


def CheckPointLoader(
    model, device: str, folder_path: str, file_name: str, only_eval: bool = False
) -> int:
    """Check if there is a savepoint and load it
        - If there is no savepoint, start from pre-trained model
        - otherwise, continue from the latest savepoint in the folder
            - if use wanna select the checkpoint, should be delete or backup the other checkpoints.

    Args:
        model (_type_): _description_
        device (str): _description_
        folder_path (str): _description_
        file_name (str): _description_

    Returns:
        latest_epoch (int): latest epoch number about the checkpoint.
    """
    file_extension = ".pth"
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # Get a list of all pth files in the folder
    pth_files = [
        file for file in os.listdir(folder_path) if file.endswith(file_extension)
    ]

    if pth_files == []:
        # print("No savepoint found. Starting from scratch")
        print("No savepoint found. Starting from Pretrained Model")
        if only_eval == True:
            print("evaluate acc of pretrained model.")
        else:
            print("Let's do transfer learning!")
        latest_epoch = 0
    else:
        # Sort the pth files based on the epoch number
        sorted_files = sorted(
            pth_files, key=lambda x: int(x.split("_epoch")[1].split(".pth")[0])
        )

        # Get the file with the highest epoch number
        latest_file = sorted_files[-1]

        # Extract the epoch number from the file name
        latest_epoch = int(latest_file.split("_epoch")[1].split(".pth")[0])
        model.load_state_dict(
            torch.load(
                f"{folder_path}/{file_name}{latest_epoch}.pth", map_location=device
            )
        )
        print(f"Continuing from latest savepoint {latest_epoch} epochs")

    return latest_epoch


def SingleEpochTrain(
    model,
    trainloader,
    criterion,
    optimizer,
    device: str,
    verb=True,
) -> tuple:
    """Run the training loop for a single epoch

    Args:
        model (_type_): _description_
        trainloader (_type_): _description_
        criterion (_type_): Cost function
        optimizer (_type_): Optimizer
        device (_type_): _description_
        verb (bool, optional): _description_. Defaults to True.

    Returns:
        tuple: train_loss, train_acc
    """
    # Training loop
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    # for i, (images, labels) in enumerate(trainloader, start=0):
    i = 0
    for images, labels in tqdm.tqdm(trainloader):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 100 == 0 and verb == True:
            now_time = time.time()
            print(
                f"Batch {str(i).rjust(len(str(len(trainloader))))}/{len(trainloader)} ({now_time - start_time:.2f}s) | train_loss: {loss.item():.4f} | train_acc: {correct/total*100:.2f}%"
            )
        i += 1
    train_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct / total

    return train_loss, train_acc


def SingleEpochEval(
    model,
    testloader,
    criterion,
    device: str,
) -> tuple:
    """_summary_

    Args:
        model (_type_): _description_
        testloader (torch.utils.data.Dataloader): _description_
        criterion (torch.nn.modules.loss): _description_
        device (str): _description_

    Returns:
        tuple: eval_loss, eval_acc
    """
    # Evaluation loop
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm.tqdm(testloader):

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Print statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    eval_loss = running_loss / len(testloader)
    eval_acc = 100.0 * correct / total

    return eval_loss, eval_acc


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


# %% https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
# class AverageMeter(object):
#     """평균과 현재 값 계산 및 저장"""

#     def __init__(self, name, fmt=":f"):
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __str__(self):
#         fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
#         return fmtstr.format(**self.__dict__)


# def accuracy(output, target, topk=(1,)):
#     """특정 k값을 위해 top k 예측의 정확도 계산"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


# def evaluate(model, criterion, data_loader, neval_batches):
#     model.eval()
#     top1 = AverageMeter("Acc@1", ":6.2f")
#     top5 = AverageMeter("Acc@5", ":6.2f")
#     cnt = 0
#     with torch.no_grad():
#         for image, target in tqdm.tqdm(data_loader):
#             output = model(image)
#             loss = criterion(output, target)
#             cnt += 1
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             print(".", end="")
#             top1.update(acc1[0], image.size(0))
#             top5.update(acc5[0], image.size(0))
#             if cnt >= neval_batches:
#                 return top1, top5

#     return top1, top5
