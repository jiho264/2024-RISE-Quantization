import torch, time, argparse
import torch.nn as nn
from myAdaRound.quant_layer import QuantLayer
from myAdaRound.quant_block import QuantBasicBlock
from myAdaRound.utils import (
    GetDataset,
    evaluate,
    quantizerDict,
    StraightThrough,
)
from myAdaRound.data_utils import save_inp_oup_data, _get_train_samples
import torchvision.models.resnet as resnet


#################################################################################################
## (option) 4. Compute AdaRound values
#################################################################################################


def _decayed_beta(i, n_iter, _warmup=0.2):
    if i < n_iter * _warmup:
        return torch.tensor(0.0)
    else:
        # 0 ~ 1 when after 4k iter of 20k len
        decay = (i - n_iter * _warmup) / (n_iter * (1 - _warmup))
        _beta = 18 - decay * 18 + 2
        return _beta


def _computeAdaRoundValues(
    model, layer, cali_data, batch_size, lr, _fp_prediction=None
):
    model.eval()
    # [1] get {Origin FP output(A_fp_lth), Quantized input and output(X_q_lth, A_q_lth)}
    layer.w_quant_enable = True
    A_fp_lth = layer.fp_outputs.to("cuda")
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

    print(optimizer_w, n_iter)
    if optimizer_a != None:
        print(optimizer_a)

    model.train()

    _pd_loss_func = None
    if _fp_prediction != None:
        _pd_loss_func = torch.nn.KLDivLoss(reduction="batchmean")

    for i in range(1, n_iter + 1):

        idx = torch.randperm(X_q_lth.size(0))[:batch_size]

        optimizer_w.zero_grad()
        if optimizer_a != None:
            optimizer_a.zero_grad()

        """ Layer reconstruction loss"""
        _tmp_A_q_lth = layer.forward(X_q_lth[idx])
        _mse = (A_fp_lth[idx] - _tmp_A_q_lth).abs().pow(2).mean()
        _beta = _decayed_beta(i, n_iter)
        _reg_loss = layer.weight_quantizer.f_reg(beta=_beta)

        loss = _mse + layer.weight_quantizer.lamda * _reg_loss

        """ PD loss """
        if _fp_prediction != None:
            _fp_prediction_idx = _fp_prediction[idx]
            _int_prediction_idx = model.forward(X_q_lth[idx])
            _pd_loss = _pd_loss_func(
                nn.functional.log_softmax(_int_prediction_idx, dim=1),
                nn.functional.softmax(_fp_prediction_idx, dim=1),
            )
            loss += _pd_loss * 0.001

        loss.backward()
        optimizer_w.step()
        if optimizer_a != None:
            optimizer_a.step()
        if i % 1000 == 0 or i == 1:
            if _fp_prediction == None:
                print(
                    f"Iter {i:5d} | Total loss: {loss:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss:.4f}) beta={_beta:.2f}"
                )
            else:
                print(
                    f"Iter {i:5d} | Total loss: {loss:.4f} (MSE:{_mse:.4f}, PD: {_pd_loss}, Reg:{_reg_loss:.4f}) beta={_beta:.2f}"
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
            if isinstance(module, QuantLayer):
                print(name, end="")
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
            if isinstance(module, QuantLayer):
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
## (option) 5. Compute BRECQ values
#################################################################################################
def _computeBRECQValues(model, block, cali_data, batch_size, lr, _fp_prediction=None):
    model.eval()
    # [1] get {Origin FP output(A_fp_lth), Quantized input and output(X_q_lth, A_q_lth)}
    block.w_quant_enable = True
    A_fp_lth = block.fp_outputs.to("cuda")
    X_q_lth, A_q_lth = save_inp_oup_data(model, block, cali_data)
    print(" <- Commas indicate the INT inference.")

    # [2] Define the optimizer and loss function
    # optimizer_w, n_iter = torch.optim.Adam([layer.weight_quantizer._v], lr=lr), 20000
    optimizer_w, n_iter = torch.optim.Adam(block.get_rounding_parameter(), lr=lr), 20000
    optimizer_a = None
    if block.a_quant_inited == False and block.a_quant_enable == True:
        idx = torch.randperm(A_q_lth.size(0))[: max(256, batch_size)]
        block.init_act_quantizer(A_q_lth[idx])
        block.a_quant_inited = True
        # optimizer_a = torch.optim.Adam([layer.act_quantizer._scaler])
        optimizer_a = torch.optim.Adam([block.get_scaler_parameter()])

    print(optimizer_w, n_iter)
    if optimizer_a != None:
        print(optimizer_a)

    model.train()

    _pd_loss_func = None
    if _fp_prediction != None:
        _pd_loss_func = torch.nn.KLDivLoss(reduction="batchmean")

    for i in range(1, n_iter + 1):

        idx = torch.randperm(X_q_lth.size(0))[:batch_size]

        optimizer_w.zero_grad()
        if optimizer_a != None:
            optimizer_a.zero_grad()

        """ Block reconstruction loss """
        _tmp_A_q_lth = block.forward(X_q_lth[idx])
        _mse = (A_fp_lth[idx] - _tmp_A_q_lth).abs().pow(2).mean()
        _beta = _decayed_beta(i, n_iter)
        # _reg_loss = block.weight_quantizer.f_reg(beta=_beta)
        _reg_loss = block.get_sum_of_f_reg_with_lambda(beta=_beta)

        # loss = _mse + block.weight_quantizer.lamda * _reg_loss
        loss = _mse + _reg_loss  # already computed lambda * f_reg

        """ PD loss """
        if _fp_prediction != None:
            _fp_prediction_idx = _fp_prediction[idx]
            _int_prediction_idx = model.forward(X_q_lth[idx])
            _pd_loss = _pd_loss_func(
                nn.functional.log_softmax(_int_prediction_idx, dim=1),
                nn.functional.softmax(_fp_prediction_idx, dim=1),
            )
            loss += _pd_loss * 1

        loss.backward()
        optimizer_w.step()
        if optimizer_a != None:
            optimizer_a.step()
        if i % 1000 == 0 or i == 1:
            if _fp_prediction == None:
                print(
                    f"Iter {i:5d} | Total loss: {loss:.4f} (MSE:{_mse:.4f}, Reg:{_reg_loss:.4f}) beta={_beta:.2f}"
                )
            else:
                print(
                    f"Iter {i:5d} | Total loss: {loss:.4f} (MSE:{_mse:.4f}, PD: {_pd_loss}, Reg:{_reg_loss:.4f}) beta={_beta:.2f}"
                )

    torch.cuda.empty_cache()
    # layer.weight_quantizer.setRoundingValues()
    block.setRoundingValues()
    return None


def runBRECQ(
    model,
    train_loader,
    num_samples=1024,
    batch_size=32,
    lr=0.01,
    num_layers=None,
    PDquant=False,
):
    model.eval()
    _fp_prediction = None
    cali_data = _get_train_samples(train_loader, num_samples)

    # Optaining the ORIGIN input and output data of each layer
    def _getFpInputOutput(module: nn.Module):
        for name, module in module.named_children():
            if isinstance(module, QuantLayer):
                print(name, end="")
                module.w_quant_enable = False  # FP inference
                _, FP_OUTPUTS = save_inp_oup_data(model, module, cali_data)
                module.fp_outputs = FP_OUTPUTS.to("cpu")
                print(" <- Dots indicate the Original FP inference.")
                print("   FP_OUTPUTS shape", module.fp_outputs.shape)
                if name == "fc" and PDquant == True:
                    nonlocal _fp_prediction
                    _fp_prediction = FP_OUTPUTS.to("cuda")
            else:
                _getFpInputOutput(module)

    _getFpInputOutput(model)

    # Optaining the ORIGIN input and output data of each layer
    # 이건 Id addition 이후에 있는 진짜 블록의 output이라, relu뭍은거여서 따로 구해야함.
    def _getFpInputOutput_block(module: nn.Module):
        for name, module in module.named_children():
            if isinstance(module, QuantBasicBlock):
                print(name, end="")
                module.w_quant_enable = False  # FP inference
                _, FP_OUTPUTS = save_inp_oup_data(model, module, cali_data)
                module.fp_outputs = FP_OUTPUTS.to("cpu")
                print(" <- Dots indicate the Original FP inference.")
                print("   FP_OUTPUTS shape", module.fp_outputs.shape)
            else:
                _getFpInputOutput_block(module)

    _getFpInputOutput_block(model)

    if PDquant == True:
        return model, cali_data, _fp_prediction

    _layer_cnt = 0

    # Compute the AdaRound values
    def _runBRECQ(module: nn.Module, batch_size):
        nonlocal _layer_cnt
        for name, module in module.named_children():
            if isinstance(module, QuantLayer):
                _layer_cnt += 1
                print(f"\n[{_layer_cnt}/{num_layers}] AdaRound computing: {name}")
                _computeAdaRoundValues(model, module, cali_data, batch_size, lr)
                # the len of cali_data = num_samples
                # the GD batch size = batch_size
            elif isinstance(module, QuantBasicBlock):
                _layer_cnt += 1
                print(f"\n[{_layer_cnt}/{num_layers}] BRECQ computing: {name}")
                _computeBRECQValues(model, module, cali_data, batch_size, lr)
                # the len of cali_data = num_samples
                # the GD batch size = batch_size
            else:
                _runBRECQ(module, batch_size)

    _runBRECQ(model, batch_size)


#################################################################################################
## (option) 5. Compute PDquant values
#################################################################################################
def runPDquant(
    model, train_loader, num_samples=1024, batch_size=32, lr=0.01, num_layers=None
):
    model, cali_data, _fp_prediction = runBRECQ(
        model, train_loader, num_samples, batch_size, lr, num_layers, PDquant=True
    )

    _layer_cnt = 0

    # Compute the AdaRound values
    def _runPDquant(module: nn.Module, batch_size):
        nonlocal _layer_cnt
        for name, module in module.named_children():
            if isinstance(module, QuantLayer):
                _layer_cnt += 1
                print(f"\n[{_layer_cnt}/{num_layers}] PDquant computing: {name}")
                _computeAdaRoundValues(
                    model, module, cali_data, batch_size, lr, _fp_prediction
                )
                # the len of cali_data = num_samples
                # the GD batch size = batch_size
            elif isinstance(module, QuantBasicBlock):
                _layer_cnt += 1
                print(f"\n[{_layer_cnt}/{num_layers}] PDquant computing: {name}")
                _computeBRECQValues(
                    model, module, cali_data, batch_size, lr, _fp_prediction
                )
                # the len of cali_data = num_samples
                # the GD batch size = batch_size
            else:
                _runPDquant(module, batch_size)

    _runPDquant(model, batch_size)


#################################################################################################
## 3. Main function
#################################################################################################
def seed_all(seed=0):
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU..
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

    def block_refactor(module, weight_quant_params, act_quant_params):
        first_conv, first_bn = None, None
        # if W4A4 with head_stem_8bit
        head_stem_weight_quant_params = weight_quant_params.copy()
        head_stem_act_quant_params = act_quant_params.copy()

        if args.get("head_stem_8bit") == True:
            head_stem_weight_quant_params.update(dict(dstDtype="INT8"))
            head_stem_act_quant_params.update(dict(dstDtype="INT8"))

        for name, child_module in module.named_children():
            if isinstance(child_module, nn.Conv2d):
                """For first ConvBnRelu"""
                if name == "conv1":
                    first_conv = child_module
                    setattr(module, name, StraightThrough())
            elif isinstance(child_module, (nn.BatchNorm2d)):
                """For first ConvBnRelu"""
                if name == "bn1":
                    first_bn = child_module
                    setattr(module, name, StraightThrough())
            elif isinstance(child_module, (nn.ReLU)):
                """For first ConvBnRelu"""
                setattr(
                    module,
                    "conv1",
                    QuantLayer(
                        conv_module=first_conv,
                        bn_module=first_bn,
                        act_module=child_module,
                        w_quant_args=head_stem_weight_quant_params,
                        a_quant_args=head_stem_act_quant_params,
                        folding=args["folding"],
                    ),
                )
                setattr(module, name, StraightThrough())
            elif isinstance(child_module, (nn.Sequential)):
                """For each Blocks"""
                for blockname, blockmodule in child_module.named_children():
                    if isinstance(blockmodule, (resnet.BasicBlock)):
                        setattr(
                            child_module,
                            blockname,
                            QuantBasicBlock(
                                org_basicblock=blockmodule,
                                w_quant_args=weight_quant_params,
                                a_quant_args=act_quant_params,
                                folding=args["folding"],
                            ),
                        )
                        print(f"- Quant Block {blockname} making done !")
            elif isinstance(child_module, (nn.Linear)):
                # only for FC layer
                if name == "fc":
                    setattr(
                        module,
                        name,
                        QuantLayer(
                            conv_module=child_module,
                            w_quant_args=head_stem_weight_quant_params,
                            a_quant_args=head_stem_act_quant_params,
                        ),
                    )

    print("Replace to QuantLayer")
    with torch.no_grad():
        block_refactor(model, weight_quant_params, act_quant_params)

    print("Qparams computing done!")

    # Count the number of QuantMoQuantLayerdule
    num_layers = 0
    num_bn = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantLayer):
            num_layers += 1
            print(f"    QuantLayer: {name}, {module.weight.shape}")
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
    if "BRECQ" in weight_quant_params:
        if weight_quant_params["BRECQ"] == True:
            runBRECQ(
                model,
                train_loader,
                num_samples=args["num_samples"],
                batch_size=args["batch_size_AdaRound"],
                num_layers=num_layers,
                lr=args["lr"],
            )
            print(f"BRECQ values computing done!")
    if "PDquant" in weight_quant_params:
        if weight_quant_params["PDquant"] == True:
            runPDquant(
                model,
                train_loader,
                num_samples=args["num_samples"],
                batch_size=args["batch_size_AdaRound"],
                num_layers=num_layers,
                lr=args["lr"],
            )
            print("PDQuant computing done!")

    _top1, _ = evaluate(
        model, test_loader, neval_batches=_len_eval_batches, device="cuda"
    )
    # for benchmarking
    if _len_eval_batches == len(test_loader):
        print(
            f"\n    Quantized model Evaluation accuracy on 50000 images, {_top1.avg:2.3f}%"
        )
    # for debugging
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
        "--paper",
        default="PDquant",
        type=str,
        help="paper name",
        choices=["AdaRound", "BRECQ", "PDquant"],
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
        default="INT4",
        type=str,
        help="destination data type",
        choices=["INT4", "INT8", "FP32"],
    )
    parser.add_argument(
        "--head_stem_8bit",
        action="store_true",
        help="Head and stem are 8 bit. default is 4 bit (fully quantized)",
    )

    """ Setup """
    args = parser.parse_args()

    def _get_args_dict(args):
        """weight"""
        weight_quant_params = dict(
            scheme=args.scheme_w,
            dstDtype=args.dstDtypeW,
            per_channel=args.per_channel,
        )
        if args.scheme_w == "NormQuantizer":
            weight_quant_params.update(dict(p=args.p))

        """ activation """
        act_quant_params = {}
        if args.dstDtypeA != "FP32":
            if args.paper:
                act_quant_params = dict(
                    scheme=args.scheme_a,
                    dstDtype=args.dstDtypeA,
                    per_channel=False,  # activation quantization is always per layer
                )
                if args.scheme_a == "NormQuantizer":
                    act_quant_params.update(dict(p=args.p))

        """ Main """
        main_args = dict(
            arch=args.arch,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            folding=args.folding,
        )

        if args.paper:
            main_args.update(dict(batch_size_AdaRound=args.batch_size_AdaRound))
            main_args.update(dict(lr=args.lr))
            if args.paper == "AdaRound":
                weight_quant_params.update(dict(AdaRound=True))
            elif args.paper == "BRECQ":
                weight_quant_params.update(dict(BRECQ=True))
            elif args.paper == "PDquant":
                weight_quant_params.update(dict(PDquant=True))
            else:
                print("error")
                exit()
            weight_quant_params["per_channel"] = True
            args.per_channel = True  # always Per-CH when using AdaRound for weights
        if (
            args.head_stem_8bit
            and args.dstDtypeW == "INT4"
            and args.dstDtypeA == "INT4"
        ):
            main_args.update(dict(head_stem_8bit=True))

        return weight_quant_params, act_quant_params, main_args

    weight_quant_params, act_quant_params, main_args = _get_args_dict(args=args)

    """ case naming """
    _case_name = f"{args.arch}_"
    if args.paper:
        _case_name += args.paper + "_"
    _case_name += args.scheme_w + "_"
    if args.head_stem_8bit:
        _case_name += "head_stem_8bit_"
    _case_name += "CH_" if args.per_channel else "Layer_"
    _case_name += "W" + args.dstDtypeW[-1]
    _case_name += "A" + "32" if args.dstDtypeA == "FP32" else "A" + args.dstDtypeA[-1]
    _case_name += "_BNFold" if args.folding else ""
    if args.scheme_w == "NormQuantizer":
        _case_name += "_p" + str(args.p)
    if args.paper:
        _case_name += "_RoundingLR" + str(args.lr)

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
