import torch
from myAdaRound.utils import *


#################################################################################################
## 3. Main function
#################################################################################################
def main():
    # prepare the model
    import torchvision.models.resnet as resnet

    model = resnet.resnet18(weights="IMAGENET1K_V1")
    model.eval().to("cuda")

    _batch_size = 64

    train_loader, test_loader = GetDataset(batch_size=_batch_size)

    # num_eval_batches = len(test_loader)
    _num_eval_batches = 10
    # _top1, _ = evaluate(
    #     model, test_loader, neval_batches=_num_eval_batches, device="cuda"
    # )
    # print(
    #     f" Original model Evaluation accuracy on {_num_eval_batches * _batch_size} images, {_top1.avg:2.2f}"
    # )

    def quant_module_refactor_wo_fuse(
        module: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(
                    module,
                    name,
                    QuantModule(child_module, weight_quant_params, act_quant_params),
                )
            else:
                quant_module_refactor_wo_fuse(
                    child_module, weight_quant_params, act_quant_params
                )

    weight_quant_params = dict(
        active=True,
        # active=False,
        n_bits=8,
        per_channel=False,
    )
    act_quant_params = {}
    quant_module_refactor_wo_fuse(model, weight_quant_params, act_quant_params)

    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            cnt += 1
            print(f"QuantModule: {name}, {module.weight.shape}")

    print(f"Total QuantModule: {cnt}")
    # def calibration():
    #     _calib_len = 16
    #     print(f"Calibration complited with {_batch_size * _calib_len} images...")
    #     for i, (data, _) in enumerate(train_loader):
    #         if i > _calib_len:
    #             break
    #         with torch.no_grad():
    #             model(data.to("cuda"))

    #     model.to("cuda")

    # calibration()
    # convert(model, inplace=True)

    _top1, _ = evaluate(
        model, test_loader, neval_batches=_num_eval_batches, device="cuda"
    )
    print("")
    print(
        f" Quantized model Evaluation accuracy on {_num_eval_batches * _batch_size} images, {_top1.avg:2.2f}"
    )


if __name__ == "__main__":
    print("ResNet18 quantization with myAdaRound...")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
