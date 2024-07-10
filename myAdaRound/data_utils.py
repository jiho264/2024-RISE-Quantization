import torch


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """

    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(
        self,
        model,
        layer,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(
            store_input=True, store_output=True, stop_forward=True
        )

    def __call__(self, model_input):
        self.model.eval()
        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass

        handle.remove()
        self.model.train()

        return (
            self.data_saver.input_store[0].detach(),
            self.data_saver.output_store.detach(),
        )


def save_inp_oup_data(
    model,
    layer,
    cali_data: torch.Tensor,
    batch_size: int = 128,
    keep_gpu: bool = True,
):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_inp, cur_out = get_inp_out(cali_data[i * batch_size : (i + 1) * batch_size])
        cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
    return cached_inps, cached_outs
