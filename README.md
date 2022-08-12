# PIDNet Model for City-Scapes Dataset

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->





<!-- /code_chunk_output -->

## Requirements
Install all requirements in `requirements.txt` file:
```nashorn js
pip3 install -r requirements.txt
```

## Train model
!! Before runing this command, you must change mode of this bash file by : 
```python
chmod +x run_train.sh
```

Training model by running this command
```nashorn js
./run_train.sh
```

## Evaluation 
!! Before runing this command, you must change mode of this bash file by : 
```python
chmod +x run_test.sh
```

Evaluate model by running this command 
```python
./run_test.sh
```

## Dataset Architecture
```nashorn js

data
    |
    |CityScapes
                |----gtFine
                |        |-- test
                |        |-- train
                |        |-- val
                |    
                |---- leftImg8bit
                         |-- test
                         |-- train
                         |-- val
```

## Change model PIDNet to PIDNet + DCN or PIDNet + MDCN

If you want to change model from PIDNet to PIDNet + DCN, you can change this line in `models/model_utils.py`
```python
def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False, use_dcn=False, modulated=True):
```
If you want to change to PIDNet + DCN, seting ```use_dcn=True``` and ```modulated=False``` 

If you want to change to PIDNet + MDCN, setting ```use_dcn=True``` and ```modulated=True```

## Checkpoint 
Checkpoint models are located in `output/pidnet_small_cityscapes`

For PIDNet checkpoint file is `checkpoint_480.pth.tar`

For PIDNet + DCN file is `DCNV1_600epoch/best.pt`

For PIDNet + MDCN file is `DCNV2_600epoch/best.pt`

!! In the Evaluation phase, change this line in `evaluate_model.py`
```python
def main(pt_model=False):
```
If you want to evaluate PIDNet, setting ``pt_model=False``

If you want to evaluate PIDNet + DCN, setting ``pt_model=True`` and change ``checkpoint_path='/home/tanpv/workspace/SelfDC/PIDNet/output/pidnet_small_cityscapes/DCNV1_600epoch/best.pt'``

If you want to evaluate PIDNet + MDCN, setting ``pt_model=True`` and change ``checkpoint_path='/home/tanpv/workspace/SelfDC/PIDNet/output/pidnet_small_cityscapes/DCNV2_600epoch/best.pt'``

