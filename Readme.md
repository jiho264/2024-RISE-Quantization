# MyAdaRound
- AdaRound, 2020
> [Nagel, Markus, et al. "Up or down? adaptive rounding for post-training quantization." International Conference on Machine Learning. PMLR, 2020.]
- BRECQ, 2021
> [Li, Yuhang, et al. "Brecq: Pushing the limit of post-training quantization by block reconstruction." ICLR 2021.]
- PD-Quant, 2023
> [Liu, Jiawei, et al. "Pd-quant: Post-training quantization based on prediction difference metric." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.]
    
## Paper benckmark
- AdaRound W4A32 : 68.71% (base 69.68%, -0.97%p)
- AdaRound W4A8 : 68.55% (base 69.68%, -1.13%p)
- BRECQ W4A32 : 70.70% (base 71.01%, -0.31%p)
- BRECQ W4A4 with 8 Bit head/stem : 69.60% (base 71.08%, -1.48%p)
- PD-Quant W4A4 with 8 Bit head/stem : 69.23% (base 71.01%, -1.78%p)

## My implementation
| Quantization Scheme           | W4A8             | W4A4              | W4A4_8bit        | W4A4_8bit_paper               |
| ----------------------------- | ---------------- | ----------------- | ---------------- | ----------------------------- |
| AdaRound_Norm (lr=1e-2, 1e-3) | 69.202% (-0.556) | 41.040% (-28.718) | 65.250% (-4.508) | None                          |
| AdaRound_Norm (lr=1e-2, 4e-5) | 64.272% (-5.486) | 42.696% (-27.062) | 65.356% (-4.402) | None                          |
| BRECQ_Norm    (lr=1e-3, 4e-5) | 67.526% (-2.232) | 42.932% (-26.826) | 67.806% (-1.952) | 69.60% (base 71.08%, -1.48%p) |
| PDquant_Norm  (lr=1e-3, 4e-5) | 66.632% (-3.216) | 57.852% (-11.906) | 67.978% (-1.780) | 69.23% (base 71.01%, -1.78%p) |


- Pytorch Base ResNet18 W32A32 (IMAGENET1K_V1, W32A32) : 69.758%
- Weight : Per-Channel
- Activation : Per-Layer
- The quantization for Weight and Activation are using same quantization scheme.
- BN folding is referred to https://nathanhubens.github.io/fasterai/misc.bn_folding.html

## Hyper parameter search
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
 
# Made by
- LEE, JIHO
- Embedded AI LAB, INU 
- Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
- jiho264@inu.ac.kr  