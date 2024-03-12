import torch, time
import torch.nn as nn
import torch.optim as optim

from src.utils import *
from override_resnet import *


# %% my code
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
    args = parser_args()

    # %% Load the ResNet-50 model
    if args.qat == True:
        if args.qat_method == "dynamic":
            # case 1 : Dynamic Quantization
            device = "cpu"
            model = resnet50_quan(
                weights=pretrained_weights_mapping[args.num_layers]
            ).to(device)
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            model = quantized_model
            print("----------Dynamic Quantization enabled")
        elif args.qat_method == "static":
            # case 2 : Static Quantization
            print("----------Static Quantization enabled")

        elif args.qat_method == "qat":
            # case 3 : Quantization Aware Training
            print("----------Quantization Aware Training enabled")
        else:
            raise ValueError("Invalid QAT method")

    else:
        device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model = layers_mapping[args.num_layers](
            weights=pretrained_weights_mapping[args.num_layers]
        ).to(device)

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
        _, test_loader = GetDataset(
            dataset_name=args.dataset,
            device=device,
            root="data",
            batch_size=16,
            num_workers=8,
        )
        eval_loss, eval_acc = SingleEpochEval(
            model=model, testloader=test_loader, criterion=criterion, device=device
        )
        print(f"eval_loss: {eval_loss:.4f} | eval_acc: {eval_acc:.2f}%")
    print_size_of_model(model)

    return None


if __name__ == "__main__":
    main()
