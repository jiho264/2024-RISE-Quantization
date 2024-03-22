# 2024-RISE-QUANTIZATION
-  LEE, JIHO
-  Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
-  jiho264@inu.ac.kr  


# Experiments
## Post Training Dynamic Quantization
## Post Training Static Quantization
- Reference Model : 
  - ```model = torchvision.models.quantization.resnet50(weight=resnet.ResNet50_Weights.DEFAULT)```
- Dynamic Quantization
  - ```model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)```
- Static Quantization - Observer :
    - ```log
        model.qconfig = torch.quantization.get_default_qconfig("x86")
        model.conv1.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.NoopObserver.with_args(dtype=torch.float32), 
            weight=torch.quantization.NoopObserver.with_args(dtype=torch.float32)
        )
        torch.quantization.prepare(model, inplace=True)
        ```
- evaluate : batch size 25, 500 iterations on ImageNet2012 validation set
| Name                                      | 1st Conv Fuse | 1st Conv Quan | Block Fuse | Block Quan | eval_loss | eval_acc | Size  | Inference Time |
| ----------------------------------------- | ------------- | ------------- | ---------- | ---------- | --------- | -------- | ----- | -------------- |
| ResNet50                                  | X             | X             | X          | X          | 1.2494    | 85.64%   | 106MB | 42ms           |
| ----Dynamic-------------------            | ------------- | ------------- | ---------- | ---------- | --------- | -------- | ----- | -----------    |
| PTDQ_baseline                             | Dynamic       | Dynamic       | Dynamic    | Dynamic    | 1.2482    | 85.63%   | 96MB  | 44ms           |
| ----fuse on/off-------------------        | ------------- | ------------- | ---------- | ---------- | --------- | -------- | ----- | -----------    |
| PTSQ_baseline                             | O             | O             | O          | O          | 1.2597    | 85.10%   | 26MB  | 23ms           |
| PTSQ_nofuseall                            | X             | O             | X          | O          | 1.2374    | 84.62%   | 26MB  | 23ms           |
| ----1st conv------------------            | ------------- | ------------- | ---------- | ---------- | --------- | -------- | ----- | -----------    |
| PTSQ_nofuseandquanfirstconv (can't trust) | X             | X             | O          | O          | 1.2836?   | 84.94%?  | 26MB  | 23ms           |
| PTSQ_nofusefirstconv                      | X             | O             | O          | O          | 1.2744    | 85.05%   | 26MB  | 23ms           |
| PTSQ_noquanfirstconv   (can't trust)      | O             | X             | O          | O          | 1.2497?   | 85.41%?  | 26MB  | 23ms           |


