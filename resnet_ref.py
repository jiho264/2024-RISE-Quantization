import torch, sys, time, os
import torch
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import argparse
import torchvision.models.resnet as resnet
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from src.utils import SingleEpochTrain, SingleEpochEval, GetDataset


def main() -> None:

    # %% Create an argument parser
    parser = argparse.ArgumentParser(description="ResNet Training")

    # Add arguments
    parser.add_argument("--num_layers", type=int, default=18, help="number of layers")
    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="name of the dataset"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--save_every", type=int, default=5, help="save model every n epochs"
    )
    parser.add_argument(
        "--continue_from",
        type=int,
        default=-1,
        help="continue training from a savepoint",
    )
    # parser.add_argument("--verbose", type=bool, default=False, help="verbose mode")

    # Parse the arguments
    args = parser.parse_args()

    assert args.num_layers in [18, 34, 50, 101, 152]
    assert args.dataset in ["CIFAR10", "CIFAR100", "ImageNet2012"]
    assert args.lr > 0
    assert args.momentum > 0
    assert args.batch_size > 0
    assert args.num_epochs > 0
    assert args.save_every > 0
    assert args.continue_from >= -1

    device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Print the parsed arguments

    folder_path = f"resnet{args.num_layers}_{args.dataset}"
    file_name = f"resnet{args.num_layers}_{args.dataset}_epoch"  # resnet18_cifar10_epoch{epoch}.pth
    # %% use save?

    if args.continue_from == 0:
        print("Starting from scratch")
        latest_epoch = 0
    elif args.continue_from == -1:

        file_extension = ".pth"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Get a list of all pth files in the folder
        pth_files = [
            file for file in os.listdir(folder_path) if file.endswith(file_extension)
        ]

        if pth_files == []:
            print("No savepoint found. Starting from scratch")
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

            print(f"Continuing from latest savepoint {latest_epoch} epochs")

    else:
        print(f"Continuing from {args.continue_from} epochs")
        latest_epoch = args.continue_from

    # Load the ResNet-50 model
    layers_mapping = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152,
    }
    weights_mapping = {
        18: resnet.ResNet18_Weights.DEFAULT,
        34: resnet.ResNet34_Weights.DEFAULT,
        50: resnet.ResNet50_Weights.DEFAULT,
        101: resnet.ResNet101_Weights.DEFAULT,
        152: resnet.ResNet152_Weights.DEFAULT,
    }
    if latest_epoch == 0:
        # model = layers_mapping[args.num_layers](weights=weights_mapping[args.num_layers]).to(device)
        model = layers_mapping[args.num_layers]().to(device)  # start from scratch
    else:
        model = layers_mapping[args.num_layers]().to(device)
        model.load_state_dict(
            torch.load(
                f"{folder_path}/{file_name}{latest_epoch}.pth", map_location=device
            )
        )

    # %%Set up training and evaluation processes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    trainloader, testloader = GetDataset(args.dataset, device, args.batch_size)

    # if args.verbose == True:
    # print("")
    # print("Python version :", sys.version)
    # print("Pytorch version : ", torch.__version__)
    # print("cuda available : ", torch.cuda.is_available())
    # print(f"Using device: {device} |", torch.cuda.get_device_name(0))
    # print("cudnn version : ", torch.backends.cudnn.version())
    # print("cudnn enabled:", torch.backends.cudnn.enabled)
    # print("Parsed Arguments:")
    # print("- ResNet", args.num_layers)
    # print("- Learning Rate:", args.lr)
    # print("- Momentum:", args.momentum)
    # print("- Batch Size:", args.batch_size)
    # print("- Number of Epochs:", args.num_epochs)
    # print("- Save Model Every n Epochs:", args.save_every)
    # print(
    #     "- Continue From:",
    #     args.continue_from,
    #     "(Possible values: -1 (latest), 0, n)",
    # )
    # print("")
    # print(criterion)
    # print(optimizer)

    # # Load the CIFAR-10 dataset
    # print("")
    # print(trainloader.dataset)
    # print(testloader.dataset)

    # %% Training loop
    model.train()
    for epoch in range(args.num_epochs):  # Change the number of epochs as needed
        now_epoch = epoch + 1 + latest_epoch
        start_time = time.time()
        train_loss, train_acc = SingleEpochTrain(
            model=model,
            trainloader=trainloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            verb=False,
        )
        eval_loss, eval_acc = SingleEpochEval(
            model=model, testloader=testloader, criterion=criterion, device=device
        )
        end_time = time.time()
        scheduler.step()
        print(
            f"Epoch {str(now_epoch).rjust(len(str(args.num_epochs)))}/{args.num_epochs} ({end_time - start_time:.2f}s) | train_loss: {train_loss:.4f} | train_acc: {train_acc:.2f}% | eval_loss: {eval_loss:.4f} | eval_acc: {eval_acc:.2f}%"
        )
        if (now_epoch) % args.save_every == 0:
            torch.save(
                model.state_dict(),
                f"{folder_path}/{file_name}{now_epoch}.pth",
            )

    print("Finished training")
    return None


if __name__ == "__main__":
    main()
