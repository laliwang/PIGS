git clone https://github.com/laliwang/PIGS.git
cd PIGS
Code_path="$PWD"

conda create -n pigs325 python=3.9
conda activate pigs325

# for basic requirements
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_basic.txt
wget https://api.anaconda.org/download/pytorch3d/pytorch3d/0.7.4/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2
conda install pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2

# then for full requirements
pip install -r requirements_then.txt

# for thirdparty submodules
cd ${Code_path}/PIGO_module/submodules/diff-plane-rasterization
pip install . --no-build-isolation
cd ../simple-knn
pip install . --no-build-isolation
cd ${Code_path}/GHPS_module/renderpy
pip install .

# Optional
cd ${Code_path}/PIGO_module/planarsplat/submodules/diff-rect-rasterization
pip install .
cd ../quaternion-utils
pip install .

# for Segmentator from ScanNetv2
cd ${Code_path}/Data_preprocess/ScanNet++/Segmentator
cmake . && make