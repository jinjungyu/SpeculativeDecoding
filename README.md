# Speculative Decoding Implementation

## Files
* main.py : Speculative Decoding 벤치마킹 스크립트. target model autoregressive + SD 실행하여 가속, Accepted Rate 측정 가능. 현재는 vicuna benchmark의 질문들을 입력 텍스트로 함
* sampling/autoregressive_sampling.py : Autoregressive Sampling 구현
* sampling/speculative_sampling.py : 2211.Fast Inference from Transformers via Speculative Decoding 논문 알고리즘 구현
* utils/*.py : 기타 기능들 구현

## Install
도커 환경을 예시로 들지만, Anaconda로 세팅해도 무방함.

0. Using Docker
```
docker run -it --gpus all --ipc=host  pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# install git
apt update && apt install git -y
```

1. Clone the OWQ repository
```
git clone https://github.com/xvyaward/GPTQ_PV
cd GPTQ_PV
git remote update
git checkout -t origin/wct_reordering
```
2. Install all the dependencies

```
pip install -r requirements.txt
```
3. Install OWQ CUDA kernel
```
cd owq/kernel
python setup_cuda.py install
```
<!-- * `torch`: tested on v2.0.0+cu117
* `transformers`: tested on v4.29.2
* `datasets`: tested on v2.12.0 -->

<!-- Experiments were conducted on a single NVIDIA A100 GPU with 80GB memory. We also confirmed that reconstruction using OWQ works on RTX 3090 GPU (24GB memory) for <= 30B models.

We have tested 3-bit CUDA kernel on the NVIDIA A100 GPU and A6000 GPU. -->

## Usage

### OWQ reconstruction and save packed model
```bash
# OPT (125m, 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b)
python main.py facebook/opt-1.3b c4 --wbits 3 --target_bit 3.01 --seed 42 --packing --save opt-1.3b_3_01.pth

# LLaMA (7b, 13b, 30b, 65b)
python llama.py huggyllama/llama-7b c4 --wbits 3 --target_bit 3.01 --seed 42 --packing --save llama-7b_3_01.pth
```
### measure PPL using packed model (MatMul). Result is equal to reconstruction
```
python main.py facebook/opt-1.3b c4 --load opt-1.3b_3_01_pack.pth
```
### benchmark end-to-end generation (MatVec).
```
python main.py facebook/opt-1.3b c4 --benchmark 128 --load opt-1.3b_3_01_pack.pth
```