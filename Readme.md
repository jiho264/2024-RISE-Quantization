****# MyAdaRound
> [Nagel, Markus, et al. "Up or down? adaptive rounding for post-training quantization." International Conference on Machine Learning. PMLR, 2020.]

- Pytorch Base 
  - ResNet18 W32A32 (IMAGENET1K_V1, W32A32) : **69.758%**
- AdaRoundPaper
  - ResNet18 W32A32 (base): **69.68%**
  - ResNet18 W4A32 : **68.71%** +- 0.06
  - ResNet18 W4A8 : **68.55%** +- 0.01 (Using Learn Step Size quantization a.k.a. LSQ for activation quantization)
  - Detail : 
    - Symmetric quantization (AbsMaxQuantizer)
    - AdaRound GD batch = 32
    - Optimizer : Defalut Adam (lr=0.001 from pytorch)
    
## Results on ResNet18 
- BN folding is referred to https://nathanhubens.github.io/fasterai/misc.bn_folding.html
 
### Different quantization schemes (Quantize weight only)
| Quantization Scheme (Per-Layer or Per-CH)  | W8A32       | W8A32_Folded | W4A32       | W4A32_Folded |
| ------------------------------------------ | ----------- | ------------ | ----------- | ------------ |
| AbsMaxQuantizer_Layer                      | 69.534%     | 69.516%      | 0.762%      | 0.246%       |
| MinMaxQuantizer_Layer                      | 69.636%     | 69.494%      | 1.926%      | 0.284%       |
| NormQuantizer_Layer_p2.0 (old, deprecated) | 69.612%     | 69.462%      | 48.322%     | 24.248%      |
| NormQuantizer_Layer_p2.4 (old, deprecated) | 69.614%     | 69.504%      | **51.396%** | 21.970%      |
| NormQuantizer_Layer_p2.0                   | **69.642%** | 69.534%      | 41.198%     | 20.402%      |
| NormQuantizer_Layer_p2.4                   | 69.576%     | **69.538%**  | 43.800%     | 15.144%      |
| OrgNormQuantizerCode_Layer_p2.4            | 69.584%     | 69.406%      | 51.054%     | **27.032%**  |
| ---------------------------------------    | -------     | ------------ | -------     | ------------ |
| AbsMaxQuantizer_CH                         | 69.654%     | 69.654%      | 50.348%     | 51.232%      |
| MinMaxQuantizer_CH                         | 69.744%     | 69.744%      | 58.242%     | 58.236%      |
| NormQuantizer_CH_p2.0 (old, deprecated)    | 69.744%     | 69.744%      | 58.114%     | 58.296%      |
| NormQuantizer_CH_p2.4 (old, deprecated)    | 69.744%     | 69.744%      | 60.836%     | 59.864%      |
| NormQuantizer_CH_p2.0                      | 69.744%     | 69.744%      | 55.376%     | 57.444%      |
| NormQuantizer_CH_p2.4                      | 69.744%     | 69.744%      | **61.110%** | **60.980%**  |
| OrgNormQuantizerCode_CH_p2.4               | **69.788%** | **69.788%**  | 57.606%     | 57.606%      |

### AdaRound with different base quantization schemes (The lr for AdaRound is 1e-2 or 1e-3) 
| Quantization Scheme               | W4A32       | W4A32_Folded | W4A8        | W4A8_Folded |
| --------------------------------- | ----------- | ------------ | ----------- | ----------- |
| AdaRoundAbsMax_CH_lr1e-2          | 68.992%     | 68.692%      | 68.824%     | 68.544%     |
| AdaRoundMinMax_CH_lr1e-2          | 69.176%     | 69.046%      | 69.212%     | 68.994%     |
| AdaRoundNorm_CH_lr1e-2_p2.4       | **69.314%** | 69.078%      | 69.202%     | 68.974%     |
| AdaRoundOrgNorm_CH_lr1e-2_p2.4    | 69.282%     | **69.154%**  | **69.222%** | **69.076%** |
| --------------------------------- | -------     | ------------ | -------     | ----------- |
| AdaRoundAbsMax_CH_lr1e-3          | 68.124%     | 67.128%      | 67.976%     | 67.050%     |
| AdaRoundMinMax_CH_lr1e-2          | 68.634%     | 68.038%      | 68.438%     | 67.764%     |
| AdaRoundNorm_CH_lr1e-2_p2.4       | 68.832%     | 68.182%      | 68.610%     | 68.046%     |
| AdaRoundOrgNorm_CH_lr1e-2_p2.4    | **69.050%** | **68.190%**  | **68.878%** | **69.030%** |

### BRECQ with different base quantization schemes (The lr for AdaRound is 1e-2 or 1e-3) 
| Quantization Scheme       | W4A32 | W4A32_Folded | W4A8    | W4A8_Folded | W4A4    | W4A4_Folded | W4A4_8bit |
| ------------------------- | ----- | ------------ | ------- | ----------- | ------- | ----------- | --------- |
| BRECQ_MinMax_CH_lr1e-2    |       |              | 69.116% |             | 47.006% | -           | 69.220%   |
| BRECQ_Norm_CH_lr1e-2_p2.4 |       |              |         | -           | 46.790% |             | 69.332%   |

- BRECQ W4A32 : 70.70%
- BRECQ W3A32 : 69.81%
- BRECQ W2A32 : 66.30%
- BRECQ W4A4 with hean/stem are 8 Bit  : 69.60%
- BRECQ W2A4 with hean/stem are 8 Bit  : 64.80%

### PD-Quant with different base quantization schemes (The lr for AdaRound is 1e-2 or 1e-3) 
| Quantization Scheme       | W4A32 | W4A32_Folded | W4A8 | W4A8_Folded | W4A4 | W4A4_Folded | W4A4_8bit |
| ------------------------- | ----- | ------------ | ---- | ----------- | ---- | ----------- | --------- |
| BRECQ_MinMax_CH_lr1e-2    |       |              |      |             |      | -           |           |
| BRECQ_Norm_CH_lr1e-2_p2.4 |       |              |      | -           |      |             |           |

- Weight : Per-Channel
- Activation : Per-Layer
- The quantization for Weight and Activation are using same quantization scheme.

### 주요 구현 포인트
- Weight들은 양수와 음수가 모두 있는 분포이지만, ReLU이후의 Activation 값들은 0과 양수만 존재하므로 해당 부분에선 one side distibution 전용 quantization 수식 적용함.
  - ReLU인 곳에서 1D Search시, AdaRoundAbsMax_CH_lr1e-2_W4A8_Folded 에서 68.482%  -> 68.544%로 성능 향상. 
  - OrgNorm에서도 ReLU인 부분 1D로 변경시 0.1%p미만의 미미한 성능 향상.
- Weight는 AdaRound에서 항상 Per-Channel로 Qparams를 구함. Activation은 항상 Per-Layer로 Qparams를 구함. 각 Batch에 따라 ch에 대한 정보가 크게 달라질 수 있기 때문임.
- Activation에 대한 Scaler는 1024장의 Calibration set으로부터 256장의 sample로 구함. Qparams를 구하는 수식은 각 scheme을 따르며, AdaRound를 진행하면서 Scaler_A도 Gradient descent를 통해 개선해 나감.

### Todo
- [ ] Add the W4A4 results using only AdaRound.
- [ ] Add the results about BRECQ with many different scheme and bit-witdh cases.
- [ ] Implement the PD-Quant. need adding "prediction diffence loss"

# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  