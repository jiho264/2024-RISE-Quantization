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
    quantized_act_input, fp_act_output = save_inp_oup_data(model, layer, cali_data)
    print(" <- Commas indicate the INT inference.")
    if layer.a_quant_inited == False:
        idx = torch.randperm(fp_act_output.size(0))[: max(batch_size, 256)]
        layer.init_act_quantizer(fp_act_output[idx])
        print("activation quantizer initialized")
        layer.a_quant_inited = True

    # [2] init values
    optimizer_w, n_iter = torch.optim.Adam([layer.weight_quantizer._v], lr=lr), 20000
    optimizer_a = torch.optim.Adam([layer.act_quantizer._scaler])
    print(optimizer_w, n_iter)
    for i in range(1, n_iter + 1):
        optimizer_w.zero_grad()
        optimizer_a.zero_grad()
        model.train()

        # random sampling (32 samples in 1024 samples)
        idx = torch.randperm(quantized_act_input.size(0))[:batch_size]
        # init act quantizer

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
        optimizer_w.step()
        optimizer_a.step()
        if i % 1000 == 0 or i == 1:
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


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input


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
    #         f"    Original model Evaluation accuracy on 50000 images, {_top1.avg:2.3f}%"
    #     )
    # # for debugging
    # else:
    #     print(
    #         f"    Original model Evaluation accuracy on {_len_eval_batches * _batch_size} images, {_top1.avg:2.3f}%"
    #     )

    def _quant_module_refactor_with_bn_folding(
        module: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        """
        prev_module = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d)):
                prev_module = child_module
                prev_name = name
            elif isinstance(child_module, nn.BatchNorm2d):
                print(
                    f"    {prev_module._get_name()} <- {child_module._get_name()}",
                    end="",
                )
                setattr(
                    module,
                    prev_name,
                    QuantModule(
                        prev_module,  # prev == conv2d or linear
                        weight_quant_params,
                        act_quant_params,
                        child_module,  # child == BN
                    ),
                )
                setattr(module, name, StraightThrough())  # remove bn layer
            elif isinstance(child_module, nn.Linear):
                # FC layer does not have BN
                setattr(
                    module,
                    name,
                    QuantModule(child_module, weight_quant_params, act_quant_params),
                )
            else:
                _quant_module_refactor_with_bn_folding(
                    child_module, weight_quant_params, act_quant_params
                )

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

    print("Replace to QuantModule")
    with torch.no_grad():
        if args["fold"] == True:
            _quant_module_refactor_with_bn_folding(
                model, weight_quant_params, act_quant_params
            )
        else:
            _quant_module_refactor(model, weight_quant_params, act_quant_params)

    print("Qparams computing done!")

    # Count the number of QuantModule
    num_layers = 0
    num_bn = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            num_layers += 1
            print(f"    QuantModule: {name}, {module.weight.shape}")
        elif isinstance(module, StraightThrough):
            num_bn += 1
    print(f"Total QuantModule: {num_layers}, Folded BN layers : {num_bn}")

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
            f"    Quantized model Evaluation accuracy on 50000 images, {_top1.avg:2.3f}%"
        )
    # for debugging
    else:
        print(
            f"    Quantized model Evaluation accuracy on {_len_eval_batches * _batch_size} images, {_top1.avg:2.3f}%"
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
        default="AbsMaxQuantizer",
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
        default="NormQuantizer",
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
        default="INT8",
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
    parser.add_argument("--fold", action="store_true", help="BN folding")

    ##### Setup
    args = parser.parse_args()

    weight_quant_params = dict(
        scheme=args.scheme,
        per_channel=args.per_channel,
        dstDtype=args.dstDtypeW,
    )
    act_quant_params = dict(
        # Not implemented yet
        # scheme=args.BaseScheme,
        scheme="AbsMaxQuantizer",
        dstDtype=args.dstDtypeA,
        per_channel=False,  # activation quantization is always per layer
    )
    main_args = dict(
        arch=args.arch,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        fold=args.fold,
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
    _case_name += "A" + "32" if args.dstDtypeA == "FP32" else "A" + args.dstDtypeA[-1]

    print(f"Case: [ {_case_name} ]")
    print(f"    - {main_args}")
    print(f"    - weight params: {weight_quant_params}")
    print(f"    - activation params: {act_quant_params}")
    print("")
    seed_all(args.seed)

    STARTTIME = time.time()
    main(weight_quant_params, act_quant_params, main_args)
    print(f"Total time: {time.time() - STARTTIME:.2f} sec")
