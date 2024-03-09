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
