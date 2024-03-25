# 2024-RISE-QUANTIZATION
-  LEE, JIHO
-  Dept. of Embedded Systems Engineering, Incheon National University, Republic of Korea
-  jiho264@inu.ac.kr  



# Post Training Static Quantization Experiments
- Evaluate : Entire of ImageNet2012 validation set
### 0. The reference model
- ```model = torchvision.models.quantization.resnet50(weight=resnet.ResNet50_Weights.DEFAULT)```

| Name     | eval_loss | eval_acc | Size     | Inference Time |
| -------- | --------- | -------- | -------- | -------------- |
| ResNet50 | 1.3998    | 80.852%  | 102.53MB | 32.45ms        |

### 1. Pick the fusion layer
- Quantize ALL + ```torch.quantization.get_default_qconfig("x86")```
  
| Fuse        | eval_loss | eval_acc    | Size    | Inference |
| ----------- | --------- | ----------- | ------- | --------- |
| None        | 1.4529    | 79.372%     | 26.55MB | 14.37ms   |
| Only Blocks | 1.4560    | 79.876%     | 26.15MB | 12.86ms   |
| **ALL**     | 1.4373    | **80.330%** | 26.15MB | 13.61ms   |

### 2. Pick an observer to calculate the quantization parameters (scale, zero_point)
- Fuse ALL + Quantize ALL
  
| Activation                     | Weight                                     | eval_loss | eval_acc    | Size    | Inference Time |
| ------------------------------ | ------------------------------------------ | --------- | ----------- | ------- | -------------- |
| **HistObs(reduce_range=True)** | HistogramObserver                          | 1.4645    | 78.990%     | 25.69MB | 12.44ms        |
| **HistObs(reduce_range=True)** | MinMaxObserver                             | 1.5083    | 79.126%     | 25.69MB | 12.88ms        |
| **HistObs(reduce_range=True)** | MovingAverageMinMaxObserver                | 1.5019    | 79.078%     | 25.69MB | 12.66ms        |
| **HistObs(reduce_range=True)** | **PerChannelMinMaxObserver** (default x86) | 1.4396    | **80.300%** | 26.15MB | 12.69ms        |
| **HistObs(reduce_range=True)** | **MovingAveragePerChannelMinMaxObserver**  | 1.4409    | **80.160%** | 26.15MB | 12.73ms        |
| HistObs(reduce_range=False)    | HistogramObserver                          | 4.0794    | 29.848%     | 25.69MB | 12.70ms        |
| HistObs(reduce_range=False)    | MinMaxObserver                             | 4.3364    | 25.848%     | 25.69MB | 12.90ms        |
| HistObs(reduce_range=False)    | MovingAverageMinMaxObserver                | 4.4176    | 24.836%     | 25.69MB | 12.71ms        |
| HistObs(reduce_range=False)    | PerChannelMinMaxObserver                   | 4.4285    | 24.754%     | 26.15MB | 12.58ms        |
| HistObs(reduce_range=False)    | MovingAveragePerChannelMinMaxObserver      | 4.7509    | 19.266%     | 26.15MB | 12.86ms        |

### 3. Pick the starting layer for quantization
- Fuse ALL + ```torch.quantization.get_default_qconfig("x86")```

| Quantization | eval_loss | eval_acc    | Size        | Inference Time |
| ------------ | --------- | ----------- | ----------- | -------------- |
| **ALL**      | 1.4322    | **80.324%** | 26.151272MB | 13.11ms        |
| **Layer1**   | 1.4242    | **80.324%** | 26.177596MB | 13.54ms        |
| Layer2       | 1.4129    | 80.588%     | 26.783074MB | 21.96ms        |
| Layer3       | 1.4091    | 80.534%     | 30.349262MB | 28.16ms        |
| Layer4       | 1.4217    | 80.652%     | 51.397638MB | 30.67ms        |

### 4. Pick the utilization of the training data set during the calibration phase
- Fuse ALL + ```torch.quantization.get_default_qconfig("x86")``` + Quantize ALL

| Utilization | eval_loss | eval_acc    | Size    | Inference Time |
| ----------- | --------- | ----------- | ------- | -------------- |
| 10%         | 1.4338    | 80.206%     | 26.15MB | 12.62ms        |
| 20%         | 1.4316    | 80.302%     | 26.15MB | 12.98ms        |
| 30%         | 1.4229    | 80.216%     | 26.15MB | 13.07ms        |
| 40%         | 1.4123    | 80.284%     | 26.15MB | 12.93ms        |
| 50%         | 1.4277    | 80.260%     | 26.15MB | 12.43ms        |
| 60%         | 1.4210    | 80.296%     | 26.15MB | 12.98ms        |
| 70%         | 1.4271    | 80.378%     | 26.15MB | 13.13ms        |
| 80%         | 1.4193    | 80.302%     | 26.15MB | 12.38ms        |
| 90%         | 1.4268    | 80.270%     | 26.15MB | 13.52ms        |
| **100%**    | 1.4256    | **80.424%** | 26.15MB | 12.77ms        |

#### Conclusion
- Fusion : All
- Observer : ```torch.quantization.get_default_qconfig("x86")```
  - ```log
    backend == 'x86':
    qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                   weight=default_per_channel_weight_observer) # PerChannelMinMaxObserver
    default_per_channel_weight_observer = PerChannelMinMaxObserver.with_args(
              dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    ```
  - Activation : HistogramObserver(reduce_range=True)
  - Weight : PerChannelMinMaxObserver
- Quantization : All
- Calibration : 100%
- eval_acc : 
  - Experiment01 (ALL) : 80.330%
  - Experiment02 (PerChannelMinMaxObserver) : 80.300%
  - Experiment03 (ALL) : 80.324%
  - Experiment04 (100%) : 80.424%
- 모두 같은 설정이지만, 항상 동일하진 않은 Acc lost 보임.

