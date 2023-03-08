# FLSGAN-VC

### Dataset links and speech samples are provided below:

## Datasets

* <a href="http://festvox.org/cmu_arctic/"> CMU Arctic </a>
* <a href="https://datashare.ed.ac.uk/handle/10283/3061"> VCC2018 Non-Parallel Dataset </a>

## Generated Samples

### CMU Arctic
* <a href="https://drive.google.com/drive/folders/1Xdw2mdlt24JlBw_rzkQzkNd3eCnhH3-x?usp=sharing"> Female to Female </a>
* <a href="https://drive.google.com/drive/folders/1l1GCDw-FeG2mRzau1YMP2-DV_TewvT07?usp=sharing"> Female to Male </a>
* <a href="https://drive.google.com/drive/folders/1C_0OyNO38UjAkkShUWfPpJ8uYjD7Vcix?usp=sharing"> Male to Female </a>
* <a href="https://drive.google.com/drive/folders/186cQlhP6KVU9q_SlFaVOQs2Jht-vyb6h?usp=sharing"> Male to Male </a>

### VCC2018
* <a href="https://drive.google.com/drive/folders/1wpVLpLpYTULakPapYeEKKML-eXGzE4z9?usp=sharing"> Female to Female </a>
* <a href="https://drive.google.com/drive/folders/1XrX8M2plV48sZXf9pHfhzfnBx_KmbPOK?usp=sharing"> Female to Male </a>
* <a href="https://drive.google.com/drive/folders/1MwMPXWiIKIl8L_JTqmc3q0O8avgIaONy?usp=sharing"> Male to Female </a>
* <a href="https://drive.google.com/drive/folders/16UoKK2kqA09_ViVysz3vaEHVWlNmF1AT?usp=sharing"> Male to Male </a>

### More Generated samples of CMU Arctic and VCC2018 Dataset

<a href="https://drive.google.com/drive/folders/1pFLXpHJum-ajAkZEEhADRpTAsttgVS8K?usp=sharing">Samples of CMU Arctic and VCC 2018</a>

# Code

### Prerequisites
- Linux, macOS or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Downloading Datasets
Download and save in "Data" folder
* <a href="http://festvox.org/cmu_arctic/"> CMU Arctic </a>
* <a href="https://datashare.ed.ac.uk/handle/10283/3061"> VCC2018 Non-Parallel Dataset </a>


### Installation

- Clone this repo:
```bash
git clone https://github.com/BlueBlaze6335/FLSGAN-VC.git
cd FLSGAN-VC
```
- Install all the dependencies by
```bash
pip install -r requirements.txt
```
```
ipython==8.8.0
librosa==0.9.1
matplotlib==3.4.2
numpy==1.19.5
SoundFile==0.10.3.post1
tensorflow==2.6.0
tensorflow_addons==0.17.0
torch==1.12.0
torchaudio==0.12.0
tqdm==4.42.1
```

### Run
- Training
```bash
python train.py
```

- Testing
```bash
python inference.py
```

## Acknowledgments
Our code is heavily inspired by [MelGAN VC](https://github.com/MuradBozik/audio-style-transfer).
