export http_proxy="http://127.0.0.1:7898" && export https_proxy="http://127.0.0.1:7898"

# 下载预训练权重到本地目录
cd PIGS
mkdir weights && cd weights
# X-PDNet pretrained weights from https://github.com/caodinhduc/X-PDNet-official
python -m gdown https://drive.google.com/uc?id=1ChJiTemWxG-3oTIbvOFLTo7PJsVxnZ-d
python -m gdown https://drive.google.com/uc?id=1rkPiWZ_313GGFMW1KLpyzM7a-nynchPV
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth