# VTire: A Bimodal Visuotactile Tire with High-Resolution Sensing Capability
<div align="center">
<img src=media/smart_tire.png width=90% />
</div>

## Environment
- Python version: ```python=3.8```
- Cuda version (or you can just use CPU): ```cuda=11.8``` (if you have another version of cuda, please install torch, torchvision and torchaudio that match your cuda version)
- Install required packages by
```
pip install -r requirements.txt
```
## Dataset
Install terrain classification dataset ```VisualTactileDataset``` and breakage detection dataset ```BrokenDetectionDatset``` from [Google Drive](https://drive.google.com/drive/folders/1wWM2jFsA3ivyrpDVqtxkUbVRcYdENJvy?usp=sharing). The structure of dataset is included in ```meta.txt```.

## Train
```
python train.py --cfg [config_file_name].yaml

```
## Test and Visualize the Results
```
python plot_results.py --path [log_dir]
```

## Available Resources
- Visualized results are presented in ```./media/```.
- Configurations for all experiments are included in ```./config/``` for replicating.

## Acknowledgements
- Our implementation of multimodal transformer is based on [Visuo-Tacttile-Transformer-for-Manipulation](https://github.com/yich7045/Visuo-Tactile-Transformers-for-Manipulation)