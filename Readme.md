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
 
 ## Per Layer with different quantization schemes
| Quantization Scheme (Per-Layer) | W8A32   | W8A32_Folded | W4A32   | W4A32_Folded |
| ------------------------------- | ------- | ------------ | ------- | ------------ |
| AbsMaxQuantizer_Layer           | 69.534% | 69.516%      | 0.762%  | 0.246%       |
| MinMaxQuantizer_Layer           | 69.636% | 69.494%      | 1.926%  | 0.284%       |
| NormQuantizer_Layer_p2.0        | 69.612% | 69.462%      | 48.322% | 24.248%      | <BigO설계오류> |
| NormQuantizer_Layer_p2.4        | 69.614% | 69.504%      | 51.396% | 21.970%      | <BigO설계오류> |
| OrgNormQuantizerCode_Layer_p2.4 | 69.584% | 69.406%      | 51.054% | 27.032%      |

## Per Channel with different quantization schemes 
| Quantization Scheme (Per-Channel) | W8A32   | W8A32_Folded | W4A32   | W4A32_Folded |
| --------------------------------- | ------- | ------------ | ------- | ------------ |
| AbsMaxQuantizer_CH                | 69.654% | 69.654%      | 50.348% | 51.232%      |
| MinMaxQuantizer_CH                | 69.744% | 69.744%      | 58.242% | 58.236%      |
| NormQuantizer_CH_p2.0             | 69.744% | 69.744%      | 58.114% | 58.296%      | <BigO설계오류> |
| NormQuantizer_CH_p2.4             | 69.744% | 69.744%      | 60.836% | 59.864%      | <BigO설계오류> |
| OrgNormQuantizerCode_CH_p2.4      | 69.788% | 69.788%      | 57.606% | 57.606%      |

## AdaRound with different BASE quantization schemes (The lr which for AdaRound is 0.01)
| Quantization Scheme (Per-Channel) | W4A32   | W4A32_Folded | W4A8    | W4A8_Folded |
| --------------------------------- | ------- | ------------ | ------- | ----------- |
| AdaRoundAbsMax_CH_lr0.01          | 68.992% | 68.692%      | 68.834% | 68.544%     |
| AdaRoundMinMax_CH_lr0.01          | 69.176% | 69.046%      | 69.212% | 68.994%     |
| AdaRoundNorm_CH_lr0.01_p2.4       | 68.86%  | 68.994%      |         |             |
| AdaRoundOrgNorm_CH_lr0.01_p2.4    | 69.282% | 69.154%      | 69.222% | 69.076%     |

- ReLU인 곳에서 1D Search시, AdaRoundAbsMax_CH_lr0.01에서 68.482%  -> 68.544%로 성능 향상. 
- OrgNorm에서도 ReLU인 부분 1D로 변경시 0.1%p미만의 미미한 성능 향상.



# Not comfirmed yet
## AdaRound with different BASE quantization schemes (The lr which for AdaRound is 0.01)
| Quantization Scheme (Per-Channel) | W4A4    | W4A4_Folded |
| --------------------------------- | ------- | ----------- |
| AdaRoundAbsMax_CH_lr0.01          | 25.476% | 22.928%     |
| AdaRoundMinMax_CH_lr0.01          | 48.010% | 49.414%     |
| AdaRoundNorm_CH_lr0.01_p2.4       |         | 39.374%     |
| AdaRoundOrgNorm_CH_lr0.01_p2.4    |         |             |

## AdaRound with different BASE quantization schemes (The lr which for AdaRound is 0.001)
| Quantization Scheme (Per-Channel) | W4A32   | W4A32_Folded | W4A8    | W4A8_Folded | W4A4    | W4A4_Folded |
| --------------------------------- | ------- | ------------ | ------- | ----------- | ------- | ----------- |
| AdaRoundAbsMax_CH_lr0.001         | 68.124% | 67.128%      | 67.966% | 66.848%     | 24.698% | 25.412%     |
| AdaRoundMinMax_CH_lr0.001         | 68.634% | 68.038%      | 68.438% | 67.764%     | 46.804% | 47.298%     |
| AdaRoundNorm_CH_lr0.001_p2.4      | 68.08%  | 68.210%      |         |             |         |             |
| AdaRoundOrgNorm_CH_lr0.001_p2.4   | 68.44%  | 68.360%      |         |             |         |             |

# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  
