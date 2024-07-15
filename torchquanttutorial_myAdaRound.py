import torch, time
import torch.nn as nn
from myAdaRound.utils import QuantModule, GetDataset, evaluate
from myAdaRound.data_utils import save_inp_oup_data
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


def _computeAdaRoundValues(model, layer, cali_data, batch_size):
    model.eval()
    # [1] get X_hat_input, Y_fp
    layer.w_quant_enable = True  # Quantized inference

    Y_fp = layer.fp_outputs
    quantized_act_input, _ = save_inp_oup_data(model, layer, cali_data)
    print("")

    # [2] init values
    optimizer, n_iter = torch.optim.Adam([layer.weight_quantizer._v], lr=0.01), 20000
    for i in range(1, n_iter + 1):
        optimizer.zero_grad()
        model.train()

        # random sampling (32 samples in 1024 samples)
        idx = torch.randperm(quantized_act_input.size(0))[:batch_size]
        Y_hat = layer.forward(quantized_act_input[idx])  # Y_hat = W_hat * X_hat
        _mse = (Y_fp[idx] - Y_hat).abs().pow(2).mean()  # | Y - Y_hat |^2

        # [3] regularization term (optional)
        _warmup = 0.2
        _reg_loss = 0
        _beta = 0

        if i < n_iter * _warmup:
            _reg_loss = 0
            pass
        else:
            # 0 ~ 1 when after 4k iter of 20k len
            decay = (i - n_iter * _warmup) / (n_iter * (1 - _warmup))
            _beta = 18 - decay * 18 + 2
            _reg_loss = layer.weight_quantizer.f_reg(beta=_beta)

        loss = _mse + layer.weight_quantizer.lamda * _reg_loss
        loss.backward()
        optimizer.step()
        if i % 500 == 0 or i == 1:
            print(
                f"Iter {i:5d} | Total loss: {loss:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss:.4f}) beta={_beta:.2f}"
            )

    torch.cuda.empty_cache()
    layer.weight_quantizer.setRoundingValues()
    return None


def runAdaRound(model, train_loader, num_samples=1024, batch_size=32) -> None:

    model.eval()

    cali_data = _get_train_samples(train_loader, num_samples)

    # Optaining the ORIGIN input and output data of each layer
    def _getFpInputOutput(module: nn.Module):
        for name, module in module.named_children():
            if isinstance(module, QuantModule):
                module.w_quant_enable = False  # FP inference
                _, FP_OUTPUTS = save_inp_oup_data(model, module, cali_data)
                module.fp_outputs = FP_OUTPUTS
                print("\n   FP_OUTPUTS shape", module.fp_outputs.shape)
            else:
                _getFpInputOutput(module)

    _getFpInputOutput(model)

    # Compute the AdaRound values
    def _runAdaRound(module: nn.Module, batch_size):
        for name, module in module.named_children():
            if isinstance(module, QuantModule):
                print(f"\nAdaRound computing: {name}")
                _computeAdaRoundValues(model, module, cali_data, batch_size)
                # the len of cali_data = num_samples
                # the GD batch size = batch_size
            else:
                _runAdaRound(module, batch_size)

    _runAdaRound(model, batch_size)

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

    _len_eval_batches = len(test_loader)
    # _len_eval_batches = 32
    # _top1, _ = evaluate(
    #     model, test_loader, neval_batches=_len_eval_batches, device="cuda"
    # )
    # # for benchmarking
    # if _len_eval_batches == len(test_loader):
    #     print(
    #         f"    Original model Evaluation accuracy on 50000 images, {_top1.avg:2.2f}"
    #     )
    # # for debugging
    # else:
    #     print(
    #         f"    Original model Evaluation accuracy on {_len_eval_batches * _batch_size} images, {_top1.avg:2.2f}"
    #     )

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
        # scheme="NormQuantizer",
        # p=2,
        scheme="OrgNormQuantizerCode",
        # scheme="AdaRoundQuantizer",
        per_channel=True,
        dstDtype="INT8",
    )
    print(weight_quant_params)
    act_quant_params = {}
    _quant_module_refactor(model, weight_quant_params, act_quant_params)
    print("Qparams computing done!")

    # Count the number of QuantModule
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            cnt += 1
            print(f"    QuantModule: {name}, {module.weight.shape}")

    print(f"Total QuantModule: {cnt}")

    if weight_quant_params["scheme"] == "AdaRoundQuantizer":
        runAdaRound(model, train_loader, num_samples=1024, batch_size=32)
        print(f"AdaRound values computing done!")

    _top1, _ = evaluate(
        model, test_loader, neval_batches=_len_eval_batches, device="cuda"
    )
    # for benchmarking
    if _len_eval_batches == len(test_loader):
        print(
            f"    Quantized model Evaluation accuracy on 50000 images, {_top1.avg:2.2f}"
        )
    # for debugging
    else:
        print(
            f"    Quantized model Evaluation accuracy on {_len_eval_batches * _batch_size} images, {_top1.avg:2.2f}"
        )


if __name__ == "__main__":
    print("ResNet18 quantization with myAdaRound!")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    STARTTIME = time.time()
    main()
    print(f"Total time: {time.time() - STARTTIME:.2f} sec")
