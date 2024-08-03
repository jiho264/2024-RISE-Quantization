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
from myAdaRound.quant_encoderblock import QuantEncoder
import torchvision.models.vision_transformer as vision_transformer
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock


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


# def main(weight_quant_params, act_quant_params, args):
def main(weight_quant_params={}, act_quant_params={}, args={}):
    args = {
        "arch": "ViT_B_16",
        "batch_size": 128,
        "num_samples": 1024,
    }
    print(args)
    if args["arch"] == "ViT_B_16":
        model = vision_transformer.vit_b_16(
            weights=vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        )
    else:
        raise NotImplementedError
    model.eval().to("cuda")

    _batch_size = args["batch_size"]

    train_loader, test_loader = GetDataset(batch_size=_batch_size)
    """
    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )   
    """
    _model_dict = dict(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1,
        progress=True,
    )
    for name0, module0 in model.named_children():
        # for name0, module0 in model.named_modules():
        _encoderBlockDict = None
        if isinstance(module0, nn.Conv2d):
            # name0 == conv_proj
            print("conv_proj", name0)
        elif isinstance(module0, Encoder):
            # dropout -> {0~11} -> LN
            setattr(model, name0, QuantEncoder(orgEncoder=module0))

            # for name1, module1 in module0.named_children():
            #     if isinstance(module1, nn.Dropout):
            #         print("encoder.dropout", name1)
            #     elif isinstance(module1, nn.Sequential):
            #         for encodername, encodermodule in module1.named_children():
            #             ## each EncoderBlocks

            #             if isinstance(encodermodule, EncoderBlock):
            #                 # print("encoder.block", encodername)
            #                 setattr(
            #                     module1,
            #                     encodername,
            #                     QuantEncoderBlock(orgEncoderBlock=encodermodule),
            #                 )

            #     elif isinstance(module1, nn.LayerNorm):
            #         print("encoder.LN", name1)

    # dd = model(torch.randn(4, 3, 224, 224).to("cuda"))
    # exit()
    _len_eval_batches = len(test_loader)

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

    #     ...
    # elif isinstance(module0, nn.MultiheadAttention):
    #     print(f"MSA : {name0}")
    # elif isinstance(module0, MLPBlock):
    #     print(f"MLP : {name0}")
    # elif isinstance(module0, nn.LayerNorm):
    #     print(f"LN : {name0}")
    # elif isinstance(module0, nn.Linear):
    #     print(f"Linear : {name0}")

    # def block_refactor(module, weight_quant_params, act_quant_params):
    #     first_conv, first_bn = None, None
    #     # if W4A4 with head_stem_8bit
    #     head_stem_weight_quant_params = weight_quant_params.copy()
    #     head_stem_act_quant_params = act_quant_params.copy()

    #     if args.get("head_stem_8bit") == True:
    #         head_stem_weight_quant_params.update(dict(dstDtype="INT8"))
    #         head_stem_act_quant_params.update(dict(dstDtype="INT8"))

    #     for name, child_module in module.named_children():
    #         if isinstance(child_module, nn.Conv2d):
    #             """For first ConvBnRelu"""
    #             if name == "conv1":
    #                 first_conv = child_module
    #                 setattr(module, name, StraightThrough())
    #         elif isinstance(child_module, (nn.BatchNorm2d)):
    #             """For first ConvBnRelu"""
    #             if name == "bn1":
    #                 first_bn = child_module
    #                 setattr(module, name, StraightThrough())
    #         elif isinstance(child_module, (nn.ReLU)):
    #             """For first ConvBnRelu"""
    #             setattr(
    #                 module,
    #                 "conv1",
    #                 QuantLayer(
    #                     conv_module=first_conv,
    #                     bn_module=first_bn,
    #                     act_module=child_module,
    #                     w_quant_args=head_stem_weight_quant_params,
    #                     a_quant_args=head_stem_act_quant_params,
    #                     folding=args["folding"],
    #                 ),
    #             )
    #             setattr(module, name, StraightThrough())
    #         elif isinstance(child_module, (nn.Sequential)):
    #             """For each Blocks"""
    #             for blockname, blockmodule in child_module.named_children():
    #                 if isinstance(blockmodule, (resnet.BasicBlock)):
    #                     setattr(
    #                         child_module,
    #                         blockname,
    #                         QuantBasicBlock(
    #                             org_basicblock=blockmodule,
    #                             w_quant_args=weight_quant_params,
    #                             a_quant_args=act_quant_params,
    #                             folding=args["folding"],
    #                         ),
    #                     )
    #                     print(f"- Quant Block {blockname} making done !")
    #         elif isinstance(child_module, (nn.Linear)):
    #             # only for FC layer
    #             if name == "fc":
    #                 setattr(
    #                     module,
    #                     name,
    #                     QuantLayer(
    #                         conv_module=child_module,
    #                         w_quant_args=head_stem_weight_quant_params,
    #                         a_quant_args=head_stem_act_quant_params,
    #                     ),
    #                 )

    # print("Replace to QuantLayer")
    # with torch.no_grad():
    #     block_refactor(model, weight_quant_params, act_quant_params)

    # print("Qparams computing done!")

    # # Count the number of QuantMoQuantLayerdule
    # num_layers = 0
    # num_bn = 0
    # for name, module in model.named_modules():
    #     if isinstance(module, QuantLayer):
    #         num_layers += 1
    #         print(f"    QuantLayer: {name}, {module.weight.shape}")
    #         if module.folding == True:
    #             num_bn += 1
    #         # ### for BN folding effect viewer
    #         # _str = None
    #         # if name == "conv1":
    #         #     if args["folding"] == True:
    #         #         _str = (
    #         #             f"weights_firstconv_org_folded_{weight_quant_params["dstDtype"]}.pt"
    #         #         )
    #         #     else:
    #         #         _str = (
    #         #             f"weights_firstconv_org_nonfold_{weight_quant_params["dstDtype"]}.pt"
    #         #         )

    #         # if name == "layer4.1.conv2":
    #         #     if args["folding"] == True:
    #         #         _str = (
    #         #             f"weights_lastconv_org_folded_{weight_quant_params["dstDtype"]}.pt"
    #         #         )
    #         #     else:
    #         #         _str = (
    #         #             f"weights_lastconv_org_nonfold_{weight_quant_params["dstDtype"]}.pt"
    #         #         )
    #         # if _str != None:
    #         #     torch.save(
    #         #         [
    #         #             module.weight,
    #         #             module.weight_quantizer._scaler,
    #         #             module.weight_quantizer._zero_point,
    #         #             module.weight_quantizer._n_bits,
    #         #         ],
    #         #         _str,
    #         #     )
    # print(f"Total QuantModule: {num_layers}, Folded BN layers : {num_bn}")

    # if "AdaRound" in weight_quant_params:
    #     if weight_quant_params["AdaRound"] == True:
    #         runAdaRound(
    #             model,
    #             train_loader,
    #             num_samples=args["num_samples"],
    #             batch_size=args["batch_size_AdaRound"],
    #             num_layers=num_layers,
    #             lr=args["lr"],
    #         )
    #         print(f"AdaRound values computing done!")
    # if "BRECQ" in weight_quant_params:
    #     if weight_quant_params["BRECQ"] == True:
    #         runBRECQ(
    #             model,
    #             train_loader,
    #             num_samples=args["num_samples"],
    #             batch_size=args["batch_size_AdaRound"],
    #             num_layers=num_layers,
    #             lr=args["lr"],
    #         )
    #         print(f"BRECQ values computing done!")
    # if "PDquant" in weight_quant_params:
    #     if weight_quant_params["PDquant"] == True:
    #         runPDquant(
    #             model,
    #             train_loader,
    #             num_samples=args["num_samples"],
    #             batch_size=args["batch_size_AdaRound"],
    #             num_layers=num_layers,
    #             lr=args["lr"],
    #         )
    #         print("PDQuant computing done!")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(
    #     description="myAdaRound", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument("--arch", default="ViT_B_16", type=str, help="architecture")
    # parser.add_argument("--seed", default=0, type=int, help="seed")
    # parser.add_argument(
    #     "--batch_size", default=128, type=int, help="batch size for evaluation"
    # )

    # parser.add_argument(
    #     "--num_samples",
    #     default=1024,
    #     type=int,
    #     help="number of samples for calibration",
    # )

    # """ weight quantization"""
    # parser.add_argument(
    #     "--scheme_w",
    #     default="MinMaxQuantizer",
    #     type=str,
    #     help="quantization scheme for weights",
    #     choices=quantizerDict,
    # )
    # parser.add_argument(
    #     "--dstDtypeW",
    #     default="INT8",
    #     type=str,
    #     help="destination data type",
    #     choices=["INT4", "INT8"],
    # )

    # parser.add_argument(
    #     "--paper",
    #     default="",
    #     type=str,
    #     help="paper name",
    #     choices=["AdaRound", "BRECQ", "PDquant"],
    # )

    # parser.add_argument(
    #     "--per_channel",
    #     action="store_false",
    #     help="per channel quantization for weights",
    # )
    # parser.add_argument(
    #     "--batch_size_AdaRound", default=32, type=int, help="batch size for AdaRound"
    # )
    # parser.add_argument(
    #     "--p", default=2.4, type=float, help="L_p norm for NormQuantizer"
    # )
    # parser.add_argument(
    #     "--lr", default=0.001, type=float, help="learning rate for AdaRound"
    # )

    # """ Activation quantization """
    # parser.add_argument(
    #     "--scheme_a",
    #     default="MinMaxQuantizer",
    #     type=str,
    #     help="quantization scheme for activations",
    #     choices=quantizerDict,
    # )
    # parser.add_argument(
    #     "--dstDtypeA",
    #     default="INT8",
    #     type=str,
    #     help="destination data type",
    #     choices=["INT4", "INT8", "FP32"],
    # )
    # parser.add_argument(
    #     "--head_stem_8bit",
    #     action="store_true",
    #     help="Head and stem are 8 bit. default is 4 bit (fully quantized)",
    # )

    # """ Setup """
    # args = parser.parse_args()

    # def _get_args_dict(args):
    #     """weight"""
    #     weight_quant_params = dict(
    #         scheme=args.scheme_w,
    #         dstDtype=args.dstDtypeW,
    #         per_channel=args.per_channel,
    #     )
    #     if args.scheme_w == "NormQuantizer":
    #         weight_quant_params.update(dict(p=args.p))

    #     """ activation """
    #     act_quant_params = {}
    #     if args.dstDtypeA != "FP32":
    #         if args.paper:
    #             act_quant_params = dict(
    #                 scheme=args.scheme_a,
    #                 dstDtype=args.dstDtypeA,
    #                 per_channel=False,  # activation quantization is always per layer
    #             )
    #             if args.scheme_a == "NormQuantizer":
    #                 act_quant_params.update(dict(p=args.p))

    #     """ Main """
    #     main_args = dict(
    #         arch=args.arch,
    #         batch_size=args.batch_size,
    #         num_samples=args.num_samples,
    #     )

    #     if args.paper:
    #         main_args.update(dict(batch_size_AdaRound=args.batch_size_AdaRound))
    #         main_args.update(dict(lr=args.lr))
    #         if args.paper == "AdaRound":
    #             weight_quant_params.update(dict(AdaRound=True))
    #         elif args.paper == "BRECQ":
    #             weight_quant_params.update(dict(BRECQ=True))
    #         elif args.paper == "PDquant":
    #             weight_quant_params.update(dict(PDquant=True))
    #         else:
    #             print("error")
    #             exit()
    #         weight_quant_params["per_channel"] = True
    #         args.per_channel = True  # always Per-CH when using AdaRound for weights
    #     if (
    #         args.head_stem_8bit
    #         and args.dstDtypeW == "INT4"
    #         and args.dstDtypeA == "INT4"
    #     ):
    #         main_args.update(dict(head_stem_8bit=True))

    #     return weight_quant_params, act_quant_params, main_args

    # weight_quant_params, act_quant_params, main_args = _get_args_dict(args=args)

    # """ case naming """
    # _case_name = f"{args.arch}_"
    # if args.paper:
    #     _case_name += args.paper + "_"
    # _case_name += args.scheme_w + "_"
    # if args.head_stem_8bit:
    #     _case_name += "head_stem_8bit_"
    # _case_name += "CH_" if args.per_channel else "Layer_"
    # _case_name += "W" + args.dstDtypeW[-1]
    # _case_name += "A" + "32" if args.dstDtypeA == "FP32" else "A" + args.dstDtypeA[-1]
    # if args.scheme_w == "NormQuantizer":
    #     _case_name += "_p" + str(args.p)
    # if args.paper:
    #     _case_name += "_RoundingLR" + str(args.lr)

    # print(f"\nCase: [ {_case_name} ]")
    # for k, v in main_args.items():
    #     print(f"    - {k}: {v}")
    # print(f"\n- weight params:")
    # for k, v in weight_quant_params.items():

    #     print(f"    - {k}: {v}")
    # print(f"\n- activation params:")
    # for k, v in act_quant_params.items():
    #     print(f"    - {k}: {v}")
    # print("")
    # seed_all(args.seed)

    STARTTIME = time.time()
    main()
    # main(weight_quant_params, act_quant_params, main_args)
    print(f"Total time: {time.time() - STARTTIME:.2f} sec")
