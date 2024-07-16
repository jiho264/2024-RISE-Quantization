import torch, time, argparse
import torch.nn as nn
from myAdaRound.utils import QuantModule, GetDataset, evaluate
from myAdaRound.data_utils import save_inp_oup_data
import torchvision.models.resnet as resnet


#################################################################################################
## (option) 4. Compute AdaRound values
#################################################################################################
def _get_train_samples(train_loader, num_samples):
    # calibration data loader code in AdaRound
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


def _computeAdaRoundValues(model, layer, cali_data, batch_size, lr):
    model.eval()
    # [1] get X_hat_input, Y_fp
    layer.w_quant_enable = True  # Quantized inference

    Y_fp = layer.fp_outputs
    quantized_act_input, _ = save_inp_oup_data(model, layer, cali_data)
    print(" <- Commas indicate the INT inference.")

    # [2] init values
    optimizer, n_iter = torch.optim.Adam([layer.weight_quantizer._v], lr=lr), 20000
    print(optimizer, n_iter)
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
        if i % 2000 == 0 or i == 1:
            print(
                f"Iter {i:5d} | Total loss: {loss:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss:.4f}) beta={_beta:.2f}"
            )

    torch.cuda.empty_cache()
    layer.weight_quantizer.setRoundingValues()
    return None


def runAdaRound(
    model, train_loader, num_samples=1024, batch_size=32, lr=0.01, num_layers=None
):

    model.eval()

    cali_data = _get_train_samples(train_loader, num_samples)

    # Optaining the ORIGIN input and output data of each layer
    def _getFpInputOutput(module: nn.Module):
        for name, module in module.named_children():
            if isinstance(module, QuantModule):
                module.w_quant_enable = False  # FP inference
                _, FP_OUTPUTS = save_inp_oup_data(model, module, cali_data)
                module.fp_outputs = FP_OUTPUTS
                print(" <- Dots indicate the Original FP inference.")
                print("   FP_OUTPUTS shape", module.fp_outputs.shape)
            else:
                _getFpInputOutput(module)

    _getFpInputOutput(model)

    _layer_cnt = 0

    # Compute the AdaRound values
    def _runAdaRound(module: nn.Module, batch_size):
        nonlocal _layer_cnt
        for name, module in module.named_children():
            if isinstance(module, QuantModule):
                _layer_cnt += 1
                print(f"\n[{_layer_cnt}/{num_layers}] AdaRound computing: {name}")
                _computeAdaRoundValues(model, module, cali_data, batch_size, lr)
                # the len of cali_data = num_samples
                # the GD batch size = batch_size
            else:
                _runAdaRound(module, batch_size)

    _runAdaRound(model, batch_size)

    return None


#################################################################################################
## 3. Main function
#################################################################################################
def seed_all(seed=0):
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(weight_quant_params, act_quant_params, args):

    # resnet18 Acc@1: 69.758%
    # resnet50 Acc@1: 76.130%
    if args["arch"] == "resnet18":
        model = resnet.resnet18(weights="IMAGENET1K_V1")
    else:
        raise NotImplementedError
    model.eval().to("cuda")

    _batch_size = args["batch_size"]

    train_loader, test_loader = GetDataset(batch_size=_batch_size)

    _len_eval_batches = len(test_loader)
    # _top1, _ = evaluate(
    #     model, test_loader, neval_batches=_len_eval_batches, device="cuda"
    # )
    # # for benchmarking
    # if _len_eval_batches == len(test_loader):
    #     print(
    #         f"    Original model Evaluation accuracy on 50000 images, {_top1.avg:2.2f}%"
    #     )
    # # for debugging
    # else:
    #     print(
    #         f"    Original model Evaluation accuracy on {_len_eval_batches * _batch_size} images, {_top1.avg:2.2f}%"
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

    _quant_module_refactor(model, weight_quant_params, act_quant_params)
    print("Qparams computing done!")

    # Count the number of QuantModule
    num_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            num_layers += 1
            print(f"    QuantModule: {name}, {module.weight.shape}")

    print(f"Total QuantModule: {num_layers}")

    if weight_quant_params["scheme"] == "AdaRoundQuantizer":
        runAdaRound(
            model,
            train_loader,
            num_samples=args["num_samples"],
            batch_size=args["batch_size_AdaRound"],
            num_layers=num_layers,
            lr=args["lr"],
        )
        print(f"AdaRound values computing done!")

    _top1, _ = evaluate(
        model, test_loader, neval_batches=_len_eval_batches, device="cuda"
    )
    # for benchmarking
    if _len_eval_batches == len(test_loader):
        print(
            f"    Quantized model Evaluation accuracy on 50000 images, {_top1.avg:2.2f}%"
        )
    # for debugging
    else:
        print(
            f"    Quantized model Evaluation accuracy on {_len_eval_batches * _batch_size} images, {_top1.avg:2.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="myAdaRound", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--arch", default="resnet18", type=str, help="architecture")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size for evaluation"
    )
    parser.add_argument(
        "--batch_size_AdaRound", default=32, type=int, help="batch size for AdaRound"
    )

    parser.add_argument(
        "--num_samples",
        default=1024,
        type=int,
        help="number of samples for calibration",
    )

    parser.add_argument(
        "--scheme",
        default="AdaRoundQuantizer",
        type=str,
        help="quantization scheme",
        choices=[
            "AbsMaxQuantizer",
            "MinMaxQuantizer",
            "NormQuantizer",
            "OrgNormQuantizerCode",
            "AdaRoundQuantizer",
        ],
    )
    parser.add_argument(
        "--BaseScheme",
        default="AbsMaxQuantizer",
        type=str,
        help="quantization scheme for init v in AdaRound",
        choices=[
            "AbsMaxQuantizer",
            "MinMaxQuantizer",
            "NormQuantizer",
            "OrgNormQuantizerCode",
        ],
    )
    parser.add_argument(
        "--per_channel", action="store_true", help="per channel quantization"
    )
    parser.add_argument(
        "--dstDtypeW",
        default="INT4",
        type=str,
        help="destination data type",
        choices=["INT4", "INT8"],
    )
    parser.add_argument(
        "--dstDtypeA",
        default="FP32",
        type=str,
        help="destination data type",
        choices=["INT4", "INT8", "FP32"],
    )
    parser.add_argument(
        "--p", default=2.4, type=float, help="L_p norm for NormQuantizer"
    )
    parser.add_argument(
        "--lr", default=0.01, type=float, help="learning rate for AdaRound"
    )

    ##### Setup
    args = parser.parse_args()

    weight_quant_params = dict(
        scheme=args.scheme,
        per_channel=args.per_channel,
        dstDtype=args.dstDtypeW,
    )
    act_quant_params = {
        # Not implemented yet
    }
    main_args = dict(
        arch=args.arch,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    if args.scheme == "NormQuantizer":
        weight_quant_params.update(dict(p=args.p))
    elif args.scheme == "AdaRoundQuantizer":
        main_args.update(dict(batch_size_AdaRound=args.batch_size_AdaRound))
        main_args.update(dict(lr=args.lr))
        weight_quant_params["per_channel"] = True  # always True when using AdaRound
        args.per_channel = True
        weight_quant_params.update(dict(BaseScheme=args.BaseScheme))

    _case_name = f"{args.arch}_{args.scheme}"
    _case_name += "_CH" if args.per_channel else "_Layer"
    _case_name += "_W" + args.dstDtypeW[-1]
    _case_name += "A" + "32" if args.dstDtypeA == "FP32" else args.dstDtypeA[-1]

    print(f"Case: [ {_case_name} ]")
    print(f"    - {main_args}")
    print(f"    - weight params: {weight_quant_params}")
    print(f"    - activation params: {act_quant_params}")
    # exit()
    seed_all(args.seed)

    STARTTIME = time.time()
    main(weight_quant_params, act_quant_params, main_args)
    print(f"Total time: {time.time() - STARTTIME:.2f} sec")
