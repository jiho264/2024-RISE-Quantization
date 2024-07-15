
# MyAdaRound
##### [Nagel, Markus, et al. "Up or down? adaptive rounding for post-training quantization." International Conference on Machine Learning. PMLR, 2020.]


[ResNet18]
 - Base(IMAGENET1K_V1, W32A32) : **69.758%**
 - AdaRoundPaper(base: 69.68%, W4A32): 68.71 +-0.06%
   - Symmetric quantization (AbsMaxQuantizer)
   - AdaRound GD batch = 32
   - Optimizer : Defalut Adam (lr=0.001 from pytorch)
    
# Results on ResNet18 (Per-layer or Per-channel scheme)
- Base : **69.76%**
 
| Quantization Scheme             | W8A32                | W4A32      |
| ------------------------------- | -------------------- | ---------- |
| AbsMaxQuantizer_Layer           | 69.54%               | 0.76%      |
| AbsMaxQuantizer_CH              | 69.64%               | 50.37%     |
| MinMaxQuantizer_Layer           | 69.65%               | 1.92%      |
| MinMaxQuantizer_CH              | 69.76%               | 58.24%     |
| OrgNormQuantizerCode_Layer_p2.4 |                      | 51.07%     |
| OrgNormQuantizerCode_CH_p2.4    | 69.79%               | 57.58%     |
| NormQuantizer_Layer_p2.0        | 69.65%               | 48.32%     |
| NormQuantizer_Layer_p2.4        | 69.67%               | 51.41%     |
| NormQuantizer_CH_p2.0           | 69.76%               | 58.14%     |
| NormQuantizer_CH_p2.4           | **69.80%**(overcome) | 55.84%     |
| AdaRoundAbsMax_CH_lr0.01        | -                    | 68.54%     |
| AdaRoundAbsMax_CH_lr0.001       | -                    | 67.17%     |
| AdaRoundMinMax_CH_lr0.01        | -                    | **68.95%** |
| AdaRoundMinMax_CH_lr0.001       | -                    | 67.91%     |

# ToDo
- [ ] L2Distance quantization for one side distribution (Activation values with ReLU function)
- [ ] L2Distance forward code for AdaRound.
- [ ] Activation quantization code.

# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  
