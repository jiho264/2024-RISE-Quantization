import torch, sys, time, os
import torch
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import argparse
import torchvision.models.resnet as resnet
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

print("")
print("Python version :", sys.version)
print("Pytorch version : ", torch.__version__)
print("cuda available : ", torch.cuda.is_available())
device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device} |", torch.cuda.get_device_name(0))
print("cudnn version : ", torch.backends.cudnn.version())
print("cudnn enabled:", torch.backends.cudnn.enabled)

print("")
# Create an argument parser
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
    "--continue_from", type=int, default=-1, help="continue training from a savepoint"
)

# Parse the arguments
args = parser.parse_args()

assert args.num_layers in [18, 34, 50, 101, 152]
assert args.dataset in ["CIFAR10", "CIFAR100"]
assert args.lr > 0
assert args.momentum > 0
assert args.batch_size > 0
assert args.num_epochs > 0
assert args.save_every > 0
assert args.continue_from >= -1


# Print the parsed arguments
print("Parsed Arguments:")
print("- ResNet", args.num_layers)
print("- Learning Rate:", args.lr)
print("- Momentum:", args.momentum)
print("- Batch Size:", args.batch_size)
print("- Number of Epochs:", args.num_epochs)
print("- Save Model Every n Epochs:", args.save_every)
print("- Continue From:", args.continue_from)

folder_path = f"resnet{args.num_layers}_{args.dataset}"
file_name = (
    f"resnet{args.num_layers}_{args.dataset}_epoch"  # resnet18_cifar10_epoch{epoch}.pth
)
# use save?

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
if latest_epoch == 0:
    model = resnet18(weights=resnet.ResNet18_Weights.DEFAULT).to(device)
else:
    model = resnet18().to(device)
    model.load_state_dict(
        torch.load(f"{folder_path}/{file_name}{latest_epoch}.pth", map_location=device)
    )

# Set up training and evaluation processes
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    pin_memory_device=device,
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    pin_memory_device=device,
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
print("")


def SingleEpochTrain(
    model, trainloader, criterion, optimizer, device, verb=True
) -> tuple:
    # Training loop
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(trainloader, start=0):

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

    train_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct / total

    return train_loss, train_acc


def SingleEpochEval(model, testloader, criterion, device) -> tuple:
    # Evaluation loop
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(trainloader, start=0):

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


# Training loop
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
    print(
        f"Epoch {str(now_epoch).rjust(len(str(args.num_epochs)))}/{args.num_epochs} ({end_time - start_time:.2f}s) | train_loss: {train_loss:.4f} | train_acc: {train_acc:.2f}% | eval_loss: {eval_loss:.4f} | eval_acc: {eval_acc:.2f}%"
    )
    if (now_epoch) % args.save_every == 0:
        torch.save(
            model.state_dict(),
            f"{folder_path}/{file_name}{now_epoch}.pth",
        )

print("Finished training")
