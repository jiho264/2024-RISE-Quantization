import torch, time
import torch.nn as nn
from myAdaRound.utils import QuantModule, GetDataset, evaluate
from myAdaRound.data_utils import _save_inp_oup_data
import torchvision.models.resnet as resnet


#################################################################################################
## (option) 4. Compute AdaRound values
#################################################################################################
# calibration data loader code in AdaRound
def _get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


def _computeAdaRoundValues(model, layer, cali_data):
    # to optain the X_hat which is the output of the previous quantized layers
    layer.w_quant_enable = True
    x_hat_inputs, _ = _save_inp_oup_data(model, layer, cali_data)

    print("   ", x_hat_inputs.shape)

    Y = layer.fp_outputs.view(-1)  # Y = W * X / it is constant

    Y_hat = layer.forward(x_hat_inputs).view(-1)  # Y_hat = W_hat * X_hat
    _norm = torch.norm(Y - Y_hat, p=2)
    print(_norm)
    # [ ] implement more details
    return None


def runAdaRound(model, train_loader, num_samples=1024) -> None:

    model.eval()

    cali_data = _get_train_samples(train_loader, num_samples)
    cali_data = _get_train_samples(train_loader, 128)

    # Optain the ORIGIN input and output data of each layer
    def _getFpInputOutput(module: nn.Module):
        for name, module in module.named_children():
            if isinstance(module, QuantModule):
                module.w_quant_enable = False
                _, FP_OUTPUTS = _save_inp_oup_data(model, module, cali_data)
                module.fp_outputs = FP_OUTPUTS
                print("   ", module.fp_outputs.shape)
            else:
                _getFpInputOutput(module)

    _getFpInputOutput(model)

    # Compute the AdaRound values
    def _runAdaRound(module: nn.Module):
        for name, module in module.named_children():
            if isinstance(module, QuantModule):
                _computeAdaRoundValues(model, module, cali_data)
            else:
                _runAdaRound(module)

    _runAdaRound(model)

    return None


#################################################################################################
## 3. Main function
#################################################################################################
def main():

    # resnet18 Acc@1: 69.758%
    # resnet50 Acc@1: 76.130%
    model = resnet.resnet18(weights="IMAGENET1K_V1")
    model.eval().to("cuda")

    _batch_size = 128

    train_loader, test_loader = GetDataset(batch_size=_batch_size)

    _num_eval_batches = len(test_loader)
    _num_eval_batches = 32
    # _top1, _ = evaluate(
    #     model, test_loader, neval_batches=_num_eval_batches, device="cuda"
    # )
    # print(
    #     f" Original model Evaluation accuracy on {_num_eval_batches * _batch_size} images, {_top1.avg:2.2f}"
    # )

    def _quant_module_refactor(
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
                _quant_module_refactor(
                    child_module, weight_quant_params, act_quant_params
                )

    weight_quant_params = dict(
        # scheme="AbsMaxQuantizer",
        # scheme="MinMaxQuantizer",
        # scheme="L2DistanceQuantizer",
        scheme="AdaRoundQuantizer",
        per_channel=True,
        dstDtype="INT4",
    )
    act_quant_params = {}
    _quant_module_refactor(model, weight_quant_params, act_quant_params)
    print("Qparams computing done...")

    # Count the number of QuantModule
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            cnt += 1
            print(f"    QuantModule: {name}, {module.weight.shape}")

    print(f"Total QuantModule: {cnt}")

    if weight_quant_params["scheme"] == "AdaRoundQuantizer":
        runAdaRound(model, train_loader, num_samples=1024)
        print(f"AdaRound values computing done...")

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

    STARTTIME = time.time()
    main()
    print(f"Total time: {time.time() - STARTTIME:.2f} sec")
