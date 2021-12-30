# ETNet
Codes for Paper "ETNET: ENCODER-DECODER TEMPORAL NETWORK FOR FACIAL ACTION UNITS RECOGNITION"

# Getting Started
## Installation
The code runs in torch 1.10 and python3.8.5.
Some packages like tqdm, numpy, etc. please install when debuging the codes.

## Dataset
* [DISFA](https://www.runoob.com)
* [CK+](https://sites.pitt.edu/~emotion/ck-spread.htm)

Please preprocess the data refer to [JAANet](https://github.com/ZhiwenShao/PyTorch-JAANet)
  
## Operational recommendations
Due tour insufficient computing power, we have to split the process into two stage.
On the first stage, use JAANet to extract static features.
On the second stage, train or test on the ETNet model.

## Results
| method          | 1    | 2    | 4    | 6    | 9    | 12   | 25   | 26   | avg  |
| --------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| ours            | 63.4 | 65.4 | 67.2 | 40.3 | 47.4 | 75.2 | 89.8 | 71.3 | 65.0 |
| JAANet(21'IJCV) | 62.4 | 60.7 | 67.1 | 41.1 | 45.1 | 73.5 | 90.9 | 67.4 | 63.5 |
| RTATL(21'ACMM)  | 57.8 | 52.8 | 70.8 | 53.2 | 52.7 | 74.5 | 91.5 | 51.9 | 63.1 |
| SATNet(20'ICPR) | 41.2 | 33.1 | 63.0 | 56.4 | 43.0 | 73.1 | 82.9 | 60.6 | 56.7 |
| LPNet(19'CVPR)  | 29.9 | 24.7 | 72.7 | 46.8 | 49.6 | 72.9 | 93.8 | 65.0 | 56.9 |
| IdenNet(19'FG)  | 25.5 | 34.8 | 64.5 | 45.2 | 44.6 | 70.7 | 81.0 | 55.0 | 52.6 |
| ARL(19'TAC)     | 43.9 | 42.1 | 63.6 | 41.8 | 40.0 | 76.2 | 95.2 | 66.8 | 58.7 |
| DSIN(18'ECCV)   | 42.4 | 39.0 | 68.4 | 28.6 | 46.8 | 70.8 | 90.4 | 42.2 | 53.6 |

## TODO
- [ ] Sort the codes


