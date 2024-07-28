import torch, time, argparse
import torch.nn as nn
from myAdaRound.utils import (
    QuantModule,
    GetDataset,
    evaluate,
    quantizerDict,
    StraightThrough,
)
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
    # [1] get {Origin FP output(A_fp_lth), Quantized input and output(X_q_lth, A_q_lth)}
    layer.w_quant_enable = True
    A_fp_lth = layer.fp_outputs
    X_q_lth, A_q_lth = save_inp_oup_data(model, layer, cali_data)
    print(" <- Commas indicate the INT inference.")

    # [2] Define the optimizer and loss function
    optimizer_w, n_iter = torch.optim.Adam([layer.weight_quantizer._v], lr=lr), 20000
    optimizer_a = None
    if layer.a_quant_inited == False and layer.a_quant_enable == True:
        idx = torch.randperm(A_q_lth.size(0))[: max(256, batch_size)]
        layer.init_act_quantizer(A_q_lth[idx])
        layer.a_quant_inited = True
        optimizer_a = torch.optim.Adam([layer.act_quantizer._scaler])
        print("Activation quantizer initialized")

    print(optimizer_w, n_iter)
    if optimizer_a != None:
        print(optimizer_a)

    model.train()

    def _get_f_reg_with_decayed_beta(_i, n_iter, _warmup=0.2):
        if i < n_iter * _warmup:
            return torch.tensor(0.0), torch.tensor(0.0)
        else:
            # 0 ~ 1 when after 4k iter of 20k len
            decay = (i - n_iter * _warmup) / (n_iter * (1 - _warmup))
            _beta = 18 - decay * 18 + 2
            return layer.weight_quantizer.f_reg(beta=_beta), _beta

    for i in range(1, n_iter + 1):

        idx = torch.randperm(X_q_lth.size(0))[:batch_size]

        optimizer_w.zero_grad()
        if optimizer_a != None:
            optimizer_a.zero_grad()

        _tmp_A_q_lth = layer.forward(X_q_lth[idx])
        _mse = (A_fp_lth[idx] - _tmp_A_q_lth).abs().pow(2).mean()
        _reg_loss, _beta = _get_f_reg_with_decayed_beta(i, n_iter)

        loss = _mse + layer.weight_quantizer.lamda * _reg_loss
        loss.backward()
        optimizer_w.step()
        if optimizer_a != None:
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
                # the GD batch size = batch_size.
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
    #         f"\n    Original model Evaluation accuracy on 50000 images, {_top1.avg:2.3f}%"
    #     )
    # # for debugging
    # else:
    #     print(
    #         f"\n    Original model Evaluation accuracy on {_len_eval_batches * _batch_size} images, {_top1.avg:2.3f}%"
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
                if args["folding"] == True:
                    print(
                        f"    {prev_module._get_name()} <- {child_module._get_name()}",
                        end="",
                    )
                _quant_module = QuantModule(
                    org_module=prev_module,  # prev == conv2d or linear
                    w_quant_args=weight_quant_params,
                    a_quant_args=act_quant_params,
                    bn_module=child_module,  # child == BN
                    folding=args["folding"],
                )
                setattr(
                    module,
                    prev_name,
                    _quant_module,
                )
                # remove bn layer
                setattr(module, name, StraightThrough())
                prev_module = _quant_module
            elif isinstance(child_module, nn.Linear):
                setattr(
                    module,
                    name,
                    QuantModule(
                        org_module=child_module,
                        w_quant_args=weight_quant_params,
                        a_quant_args=act_quant_params,
                        bn_module=None,  # FC layer does not have BN
                        folding=False,  # FC layer does not have BN
                    ),
                )
            elif isinstance(child_module, nn.ReLU):
                prev_module.act_func = child_module
                # 9 ConvBNRelu layers in resnet18
                # 8 ConvBn
                # 1 FC (Linear)
                print(f"    {child_module._get_name()} merged")
                # for the first conv layer which first block of each stage.
                # run only 9 times.
            else:
                _quant_module_refactor_with_bn_folding(
                    child_module, weight_quant_params, act_quant_params
                )

    print("Replace to QuantModule")
    with torch.no_grad():
        _quant_module_refactor_with_bn_folding(
            model, weight_quant_params, act_quant_params
        )
    print("Qparams computing done!")

    # Count the number of QuantModule
    num_layers = 0
    num_bn = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            num_layers += 1
            print(f"    QuantModule: {name}, {module.weight.shape}")
            if module.folding == True:
                num_bn += 1
            # ### for BN folding effect viewer
            # _str = None
            # if name == "conv1":
            #     if args["folding"] == True:
            #         _str = (
            #             f"weights_firstconv_org_folded_{weight_quant_params["dstDtype"]}.pt"
            #         )
            #     else:
            #         _str = (
            #             f"weights_firstconv_org_nonfold_{weight_quant_params["dstDtype"]}.pt"
            #         )

            # if name == "layer4.1.conv2":
            #     if args["folding"] == True:
            #         _str = (
            #             f"weights_lastconv_org_folded_{weight_quant_params["dstDtype"]}.pt"
            #         )
            #     else:
            #         _str = (
            #             f"weights_lastconv_org_nonfold_{weight_quant_params["dstDtype"]}.pt"
            #         )
            # if _str != None:
            #     torch.save(
            #         [
            #             module.weight,
            #             module.weight_quantizer._scaler,
            #             module.weight_quantizer._zero_point,
            #             module.weight_quantizer._n_bits,
            #         ],
            #         _str,
            #     )
    print(f"Total QuantModule: {num_layers}, Folded BN layers : {num_bn}")

    if "AdaRound" in weight_quant_params:
        if weight_quant_params["AdaRound"] == True:
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
            f"\n    Quantized model Evaluation accuracy on 50000 images, {_top1.avg:2.3f}%"
        )
    # for debugging.
    else:
        print(
            f"\n    Quantized model Evaluation accuracy on {_len_eval_batches * _batch_size} images, {_top1.avg:2.3f}%"
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
        "--num_samples",
        default=1024,
        type=int,
        help="number of samples for calibration",
    )
    parser.add_argument("--folding", action="store_true", help="BN folding")

    """ weight quantization"""
    parser.add_argument(
        "--scheme_w",
        default="MinMaxQuantizer",
        type=str,
        help="quantization scheme for weights",
        choices=quantizerDict,
    )
    parser.add_argument(
        "--dstDtypeW",
        default="INT4",
        type=str,
        help="destination data type",
        choices=["INT4", "INT8"],
    )
    parser.add_argument(
        "--AdaRound", default=False, type=bool, help="AdaRound for weights"
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="per channel quantization for weights",
    )
    parser.add_argument(
        "--batch_size_AdaRound", default=32, type=int, help="batch size for AdaRound"
    )
    parser.add_argument(
        "--p", default=2.4, type=float, help="L_p norm for NormQuantizer"
    )
    parser.add_argument(
        "--lr", default=0.01, type=float, help="learning rate for AdaRound"
    )

    """ Activation quantization """
    parser.add_argument(
        "--scheme_a",
        default="MinMaxQuantizer",
        type=str,
        help="quantization scheme for activations",
        choices=quantizerDict,
    )
    parser.add_argument(
        "--dstDtypeA",
        default="FP32",
        type=str,
        help="destination data type",
        choices=["INT4", "INT8", "FP32"],
    )

    ##### Setup
    args = parser.parse_args()

    weight_quant_params = dict(
        scheme=args.scheme_w,
        dstDtype=args.dstDtypeW,
        per_channel=args.per_channel,
    )
    act_quant_params = {}
    if args.dstDtypeA != "FP32" and args.AdaRound == True:
        act_quant_params = dict(
            scheme=args.scheme_a,
            dstDtype=args.dstDtypeA,
            per_channel=False,  # activation quantization is always per layer
        )
        if args.scheme_a == "NormQuantizer":
            act_quant_params.update(dict(p=args.p))

    main_args = dict(
        arch=args.arch,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        folding=args.folding,
    )
    if args.scheme_w == "NormQuantizer":
        weight_quant_params.update(dict(p=args.p))

    if args.AdaRound == True:
        main_args.update(dict(batch_size_AdaRound=args.batch_size_AdaRound))
        main_args.update(dict(lr=args.lr))
        weight_quant_params.update(dict(AdaRound=True))
        weight_quant_params["per_channel"] = True  # always Per-CH when using AdaRound
        args.per_channel = True  # always Per-CH when using AdaRound for weights

    _case_name = f"{args.arch}_"
    if args.AdaRound:
        _case_name += "AdaRound_"
    _case_name += args.scheme_w
    _case_name += "_CH" if args.per_channel else "_Layer"
    _case_name += "_W" + args.dstDtypeW[-1]
    _case_name += "A" + "32" if args.dstDtypeA == "FP32" else "A" + args.dstDtypeA[-1]
    _case_name += "_BNFold" if args.folding else ""
    if args.scheme_w == "NormQuantizer":
        _case_name += "_p" + str(args.p)
    if args.AdaRound:
        _case_name += "_AdaRoundLR" + str(args.lr)

    print(f"\nCase: [ {_case_name} ]")
    for k, v in main_args.items():
        print(f"    - {k}: {v}")
    print(f"\n- weight params:")
    for k, v in weight_quant_params.items():

        print(f"    - {k}: {v}")
    print(f"\n- activation params:")
    for k, v in act_quant_params.items():
        print(f"    - {k}: {v}")
    print("")
    seed_all(args.seed)

    STARTTIME = time.time()
    main(weight_quant_params, act_quant_params, main_args)
    print(f"Total time: {time.time() - STARTTIME:.2f} sec")
