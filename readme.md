export http_proxy="http://127.0.0.1:7898" && export https_proxy="http://127.0.0.1:7898"

cd /code3/wjh/2025-PIGS/PIGS_final

conda create -n pigs325 python=3.9
conda activate pigs325

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118


pip install numpy==1.26
pip install opencv-python==4.10.0.84
pip install open3d
pip install plyfile
pip install lpips
pip install trimesh

wget https://api.anaconda.org/download/pytorch3d/pytorch3d/0.7.4/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2
conda install pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2

