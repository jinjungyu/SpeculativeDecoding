# Speculative Decoding Implementation
[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) 논문의 알고리즘의 구현체입니다.
## Files
* main.py : Speculative Decoding 벤치마킹 스크립트. target model autoregressive + SD 실행하여 가속, Accepted Rate 측정 가능. 현재는 vicuna benchmark의 질문들을 입력 텍스트로 함
* sampling/autoregressive_sampling.py : Autoregressive Sampling 구현
* sampling/speculative_sampling.py : Speculative Decoding 구현
* utils/*.py : 기타 기능들 구현

## Install
도커 환경을 예시로 들지만, Anaconda로 세팅해도 무방함.

```
docker run -it --gpus all --ipc=host  pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
apt update && apt install git -y
git clone https://github.com/jinjungyu/SpeculativeDecoding.git
cd SpeculativeDecoding
pip install -r requirements.txt
```
## Usage

### Single-GPU Benchmark
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --target_model_name facebook/opt-6.7b --approx_model_name facebook/opt-125m --max_tokens 128 --gamma 5
```
### Multi-GPU Benchmark
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --target_model_name meta-llama/Llama-2-13b-hf --approx_model_name openlm-research/open_llama_3b_v2 --max_tokens 512 --gamma 10
```

* --gamma : draft model이 생성하는 token 개수
* --max_tokens : 생성할 토큰 개수
* --verbose : input, output text 출력 여부

## Note
* Auto-regressive language model에 대해서 구현이 확인되었는데, mamba를 붙일 경우 key-value caching 과 같은 부분이 없기 때문에 sampling/kvcache_model.py, sampling_speculative_sampling.py 부분을 변경해야할 수도 있습니다.

* 테스트 데이터가 vicuna benchmark라는 80개의 짧은 질문으로 이루어진 데이터셋인데, 실제 논문들에서 사용하는 데이터 XSum 등으로 변경하려면 추가 구현이 필요합니다.

* 위에 기재된 도커로 시작했기 때문에 필요한 패키지가 더 있을 수 있음에 유의