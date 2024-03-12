import torch, time, argparse
import torch.nn as nn
import torch.optim as optim

from src.utils import (
    SingleEpochTrain,
    SingleEpochEval,
    GetDataset,
    CheckPointLoader,
    layers_mapping,
    pretrained_weights_mapping,
    evaluate,
    print_size_of_model,
)


def main() -> None:
    """Main function for training ResNet model.

    Args:
        --num_layers (int): number of layers
        --dataset (str): name of the dataset
        --lr (float): learning rate
        --momentum (float): momentum
        --batch_size (int): batch size
        --num_epochs (int): number of epochs
        --save_every (int): save model every n epochs
        --qat (bool): Enable Quantization Aware Training
        --only_eval (bool): only evaluation mode
        --verbose (bool): print mini-batch loss and accuracy


    Returns:
        None.
    """
    # %% Create an argument parser
    parser = argparse.ArgumentParser(description="ResNet Training")

    # Add arguments
    parser.add_argument("--num_layers", type=int, default=50, help="number of layers")
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
        "--qat", type=bool, default=False, help="Enable Quantization Aware Training"
    )
    parser.add_argument("--only_eval", type=bool, default=False, help="verbose mode")

    parser.add_argument("--verbose", type=bool, default=False, help="verbose mode")

    # Parse the arguments
    args = parser.parse_args()

    assert args.num_layers in [18, 34, 50, 101, 152]
    assert args.dataset in ["CIFAR10", "CIFAR100", "ImageNet"]
    assert args.lr > 0
    assert args.momentum > 0
    assert args.batch_size > 0
    assert args.num_epochs > 0
    assert args.save_every > 0
    assert args.verbose in [True, False]
    assert args.only_eval in [True, False]
    assert args.qat in [True, False]

    # %%
    device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # Load the ResNet-50 model
    device = "cpu"
    model = layers_mapping[args.num_layers](
        weights=pretrained_weights_mapping[args.num_layers]
    ).to(device)
    ####################################################################################################
    # 여기에 Block, forward 재정의 필요함.
    #
    ####################################################################################################
    if args.qat == True:
        _folder_path = f"resnet{args.num_layers}_{args.dataset}_QAT"
    else:
        _folder_path = f"resnet{args.num_layers}_{args.dataset}"

    _file_name = f"resnet{args.num_layers}_{args.dataset}_epoch"  # resnet18_cifar10_epoch{epoch}.pth

    _latest_epoch = CheckPointLoader(
        model=model,
        device=device,
        folder_path=_folder_path,
        file_name=_file_name,
        only_eval=args.only_eval,
    )

    # %%Set up training and evaluation processes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader, test_loader = GetDataset(
        dataset_name=args.dataset,
        device=device,
        root="data",
        batch_size=args.batch_size,
    )

    # %% Training loop
    if args.only_eval == False:
        for epoch in range(args.num_epochs):  # Change the number of epochs as needed
            now_epoch = epoch + 1 + _latest_epoch
            start_time = time.time()
            train_loss, train_acc = SingleEpochTrain(
                model=model,
                trainloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                verb=args.verbose,
            )
            eval_loss, eval_acc = SingleEpochEval(
                model=model, testloader=test_loader, criterion=criterion, device=device
            )
            end_time = time.time()

            print(
                f"Epoch {str(now_epoch).rjust(len(str(args.num_epochs)))}/{args.num_epochs} ({end_time - start_time:.2f}s) | train_loss: {train_loss:.4f} | train_acc: {train_acc:.2f}% | eval_loss: {eval_loss:.4f} | eval_acc: {eval_acc:.2f}%"
            )
            if (now_epoch) % args.save_every == 0:
                torch.save(
                    model.state_dict(),
                    f"{_folder_path}/{_file_name}{now_epoch}.pth",
                )

        print("Finished training")
    else:
        # eval_loss, eval_acc = SingleEpochEval(
        #     model=model, testloader=test_loader, criterion=criterion, device=device
        # )
        # print(f"eval_loss: {eval_loss:.4f} | eval_acc: {eval_acc:.2f}%")
        num_eval_batches = (
            3125  # batch 16이면 3125개. 9999999로 하면 전체 데이터셋으로 평가
        )
        _, test_loader = GetDataset(
            dataset_name=args.dataset,
            device=device,
            root="data",
            batch_size=16,
            num_workers=4,
        )
        print("eval ref")
        top1, top5 = evaluate(
            model, criterion, test_loader, neval_batches=num_eval_batches
        )

        print(
            "Evaluation accuracy on %d images, %2.2f"
            % (num_eval_batches * args.batch_size, top1.avg)
        )
    print_size_of_model(model)

    return None


if __name__ == "__main__":
    main()
