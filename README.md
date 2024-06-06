# Environment
- python=3.8
- cuda=11.8(if you have another version of cuda, please install torch, torchvision and torchaudio that match your cuda version)
'''
pip install -r requirements.txt
'''
# Train
```
# visual-tactile 
python train.py --cfg config_vt.yaml

# visual only
python train.py --cfg config_v.yaml

# tactile only
python train.py --cfg config_t.yaml
```
