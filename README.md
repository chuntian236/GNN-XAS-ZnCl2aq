## Environment Setup and Installation

Two installation methods are provided:

1. Fully reproducible Conda environment `environment.yaml`.

2. Manual installation. 

---

### 1. Conda environment installation 

For full reproducibility, a reference Conda environment is provided in `environment.yml`. 

```bash
conda env create -f environment.yml
conda activate ml-xas

#### GPU Support
This environment assumes CUDA 12.2 available as a system module
Example: module load cuda/12.2



### 2. Manual Installation 

Below are commands used to build the development environment. 


```bash

conda create -n ml-xas python=3.10 -y
conda activate ml-xas

conda install numpy=1.23.5 scipy matplotlib scikit-learn jupyterlab ipython -y

module load cuda/12.2

pip install torch==2.2.0 torchvision==0.17.0 torchaudio lightning --index-url https://download.pytorch.org/whl/cu121

pip install captum

pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install matgl==1.3.0

pip install umap-learn torch-summary kaleido

pip install pymatgen ase "numpy<2"

python -m ipykernel install --user --name="jupyter_ml-xas"


