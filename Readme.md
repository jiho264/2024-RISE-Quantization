# MyAdaRound
##### [Nagel, Markus, et al. "Up or down? adaptive rounding for post-training quantization." International Conference on Machine Learning. PMLR, 2020.]


[ResNet18]
 - Base(IMAGENET1K_V1, W32A32) : **69.758%**
 - AdaRoundPaper(base: 69.68%, W4A32): **68.71** +-0.06%
   - Symmetric quantization (AbsMaxQuantizer)
   - AdaRound GD batch = 32
   - Optimizer : Defalut Adam (lr=0.**001** from pytorch)
    
# Results on ResNet18 
- Full precision ResNet18: **69.758%**
- BN folding is referred to https://nathanhubens.github.io/fasterai/misc.bn_folding.html
 
| Quantization Scheme (Per-Layer) | W8A32   | W8A32_Folded | W4A32   | W4A32_Folded |
| ------------------------------- | ------- | ------------ | ------- | ------------ |
| AbsMaxQuantizer_Layer           | 69.534% | 69.516%      | 0.762%  | 0.246%       |
| MinMaxQuantizer_Layer           | 69.636% | 69.494%      | 1.926%  | 0.284%       |
| NormQuantizer_Layer_p2.0        | 69.612% | 69.462%      | 48.322% | 24.248%      |
| NormQuantizer_Layer_p2.4        | 69.614% | 69.504%      | 51.396% | 21.970%      |
| OrgNormQuantizerCode_Layer_p2.4 | 69.584% | 69.406%      | 51.054% | 27.032%      |

| Quantization Scheme (Per-Channel) | W8A32   | W8A32_Folded | W4A32   | W4A32_Folded |
| --------------------------------- | ------- | ------------ | ------- | ------------ |
| AbsMaxQuantizer_CH                | 69.654% | 69.654%      | 50.348% | 51.232%      |
| MinMaxQuantizer_CH                | 69.744% | 69.744%      | 58.242% | 58.236%      |
| NormQuantizer_CH_p2.0             | 69.744% | 69.744%      | 58.114% | 58.296%      |
| NormQuantizer_CH_p2.4             | 69.744% | 69.744%      | 60.836% | 59.864%      |
| OrgNormQuantizerCode_CH_p2.4      | 69.788% | 69.788%      | 57.606% | 57.606%      |

| Quantization Scheme (Per-Channel) | W8A32 | W8A32_Folded | W4A32  | W4A32_Folded |
| --------------------------------- | ----- | ------------ | ------ | ------------ |
| AdaRoundAbsMax_CH_lr0.01          | -     |              | 68.56% |              |
| AdaRoundAbsMax_CH_lr0.001         | -     |              | 67.14% |              |
| AdaRoundMinMax_CH_lr0.01          | -     |              | 69.06% |              |
| AdaRoundMinMax_CH_lr0.001         | -     |              | 68.01% |              |
| AdaRoundNorm_CH_lr0.01_p2.4       | -     |              | 68.86% |              |
| AdaRoundNorm_CH_lr0.001_p2.4      | -     |              | 68.08% |              |
| AdaRoundOrgNorm_CH_lr0.01_p2.4    | -     |              | 68.55% |              |
| AdaRoundOrgNorm_CH_lr0.001_p2.4   | -     |              | 68.44% |              |


# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  
