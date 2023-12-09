# 

##### 0. [Optional] Create a new environment in SoL

```bash
module load mamba/latest # only for Sol
mamba create -n suprem python=3.9

source activate suprem
```

##### 1. Clone the GitHub repository

```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

##### 2. Download the pre-trained Swin UNETR checkpoint

```bash
cd pretrained_weights/
wget https://www.dropbox.com/scl/fi/gd1d7k9mac5azpwurds66/supervised_suprem_swinunetr_2100.pth?rlkey=xoqr7ey52rnese2k4hwmrlqrt
mv supervised_suprem_swinunetr_2100.pth\?rlkey\=xoqr7ey52rnese2k4hwmrlqrt supervised_suprem_swinunetr_2100.pth
cd ../
```

##### 3. Download the TotalSegmentator dataset

from [Zenodo](https://doi.org/10.5281/zenodo.6802613) (1,228 subjects) and save it to `/path/to/your/data/TotalSegmentator`

##### 4. Fine-tune the pre-trained Swin UNETR on TotalSegmentator

```bash
# Single GPU
RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/Totalsegmentator_dataset/Totalsegmentator_dataset/ # change to /path/to/your/data/TotalSegmentator
arch=swinunetr # support swinunetr, unet, and segresnet
pretrainpath=pretrained_weights/supervised_suprem_swinunetr_2100.pth
target_task=vertebrae
num_target_class=25
num_target_annotation=64

python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py --dist False --model_backbone $arch --log_name efficiency.$arch.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $pretrainpath --percent $num_target_annotation
```

##### 5. Evaluate the performance per class

```bash
RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/Totalsegmentator_dataset/Totalsegmentator_dataset/ # change to /path/to/your/data/TotalSegmentator
target_task=vertebrae
num_target_class=25
num_target_annotation=64

python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT test.py --dist False --model_backbone $arch --log_name efficiency.$arch.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $pretrainpath --train_type efficiency 
```