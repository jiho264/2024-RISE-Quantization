# MyAdaRound
##### [Nagel, Markus, et al. "Up or down? adaptive rounding for post-training quantization." International Conference on Machine Learning. PMLR, 2020.]


[ResNet18]
 - Base(IMAGENET1K_V1, W32A32) : **69.758%**
 - AdaRoundPaper(base: 69.68%, W4A32): **68.71** +-0.06%
   - Symmetric quantization (AbsMaxQuantizer)
   - AdaRound GD batch = 32
   - Optimizer : Defalut Adam (lr=0.001 from pytorch)
    
# Results on ResNet18 
- Full precision ResNet18: **69.758%**
- 
 
| Quantization Scheme (Per-Layer) | W8A32      | W4A32      | Folded W8A32 | Folded W4A32 |
| ------------------------------- | ---------- | ---------- | ------------ | ------------ |
| AbsMaxQuantizer_Layer           | 69.54%     | 0.76%      | 69.52%       |              |
| MinMaxQuantizer_Layer           | **69.65%** | 1.92%      | 69.49%       |              |
| NormQuantizer_Layer_p2.0        | **69.65%** | 48.32%     |              |              |
| NormQuantizer_Layer_p2.4        | 69.58%     | **51.41%** |              |              |
| OrgNormQuantizerCode_Layer_p2.4 | 69.62%     | 51.07%     |              |              |

| Quantization Scheme (Per-Channel) | W8A32      | W4A32      |
| --------------------------------- | ---------- | ---------- |
| AbsMaxQuantizer_CH                | 69.64%     | 50.37%     |
| MinMaxQuantizer_CH                | 69.76%     | 58.24%     |
| NormQuantizer_CH_p2.0             | 69.76%     | 58.14%     |
| NormQuantizer_CH_p2.4             | 69.76%     | **60.81%** |
| OrgNormQuantizerCode_CH_p2.4      | **69.79%** | 57.58%     |

| Quantization Scheme (Per-Channel) | W8A32 | W4A32  |
| --------------------------------- | ----- | ------ |
| AdaRoundAbsMax_CH_lr0.01          | -     | 68.56% |
| AdaRoundAbsMax_CH_lr0.001         | -     | 67.14% |
| AdaRoundMinMax_CH_lr0.01          | -     | 69.06% |
| AdaRoundMinMax_CH_lr0.001         | -     | 68.01% |
| AdaRoundNorm_CH_lr0.01_p2.4       | -     | 68.86% |
| AdaRoundNorm_CH_lr0.001_p2.4      | -     | 68.08% |
| AdaRoundOrgNorm_CH_lr0.01_p2.4    | -     | 68.55% |
| AdaRoundOrgNorm_CH_lr0.001_p2.4   | -     | 68.44% |


# ToDo
- [ ] L2Distance quantization for one side distribution (Activation values with ReLU function)
- [x] L2Distance forward code for AdaRound.
- [ ] Activation quantization code.

# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  
