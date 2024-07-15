[Base] ResNet18 W32A32: 69.76%
AbsMaxQuantizer W4A32: 50.38% (Ada에서 아무것도 안 했을 때 50.38 나오는거 확인함)

[In this paper, W4A32 is 69.71% from origin 69.68%]
AdaRound W4A32 | AbsMaxQuantizer | lr 0.001: 67.17%
AdaRound W4A32 | AbsMaxQuantizer | lr 0.01: 68.54% 
AdaRound W4A32 | MinMaxQuantizer | lr 0.001: 67.91%
AdaRound W4A32 | MinMaxQuantizer | lr 0.01: 68.95%
