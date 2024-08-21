import torch, time, argparse
import torch.nn as nn
from myAdaRound.quant_layer import QuantLayer
from myAdaRound.quant_block import QuantBasicBlock
from myAdaRound.utils import *
from myAdaRound.data_utils import save_inp_oup_data, _get_train_samples
import torchvision.models.resnet as resnet
from myAdaRound.quant_model import QuantResNet


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


def main(weight_quant_params={}, act_quant_params={}, args={}):

    # resnet18 Acc@1: 69.758%
    # resnet50 Acc@1: 76.130%
    # if args["arch"] == "resnet18":
    #     model = resnet.resnet18(weights="IMAGENET1K_V1")
    # else:
    #     raise NotImplementedError
    model = resnet.resnet18(weights="IMAGENET1K_V1")
    model.eval().to("cuda")

    _batch_size = 128

    train_loader, test_loader = GetDataset(batch_size=_batch_size)
    weight_quant_params = dict(
        scheme="AbsMaxQuantizer",
        dstDtype="INT8",
    )
    act_quant_params = dict(
        scheme="NaiveDynnamicMinMaxQuantizer",
        dstDtype="INT8",
    )
    main_args = dict(
        folding=True,
    )
    print(weight_quant_params)
    print(act_quant_params)

    model = QuantResNet(model, weight_quant_params, act_quant_params, main_args)

    # calib
    # _top1, _ = evaluate(model, test_loader, neval_batches=16, device="cuda")

    _len_eval_batches = len(test_loader)
    # _len_eval_batches = 1

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
    seed_all()
    # exit()
    starttime = time.time()
    # main(weight_quant_params, act_quant_params, main_args)
    main()
    print(f"Total time: {time.time() - starttime:.2f} sec")
