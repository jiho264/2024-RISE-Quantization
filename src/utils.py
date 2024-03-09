import torch, time


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
        for i, (images, labels) in enumerate(testloader, start=0):

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


def GetDataset(dataset_name, device, batch_size, num_workers=8):
    # Load the CIFAR-10 dataset
    if dataset_name == "CIFAR10":
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets

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

        trainset = datasets.CIFAR10(
            root="./data", train=True, download=False, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=device,
        )

        testset = datasets.CIFAR10(
            root="./data", train=False, download=False, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return trainloader, testloader

    # Load the CIFAR-100 dataset
    elif dataset_name == "CIFAR100":
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        trainset = datasets.CIFAR100(
            root="./data", train=True, download=False, transform=transform_train
        )
        testset = datasets.CIFAR100(
            root="./data", train=False, download=False, transform=transform_test
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return trainloader, testloader
