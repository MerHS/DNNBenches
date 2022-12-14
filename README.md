# DNNBenches

Fine-grained benchmark suites for DNN parallelization techniques

## Benchmark list

### Base configuration

- Model: GPT2
- Dataset: Wikitext-2-raw-v1
- #GPU: 2
- Per-device batch size: 4
- Total epoch: 3
- Preprocessing Workers: 8
- Additional params:
  - fp16: True
  - seed: 42

### Default setting

- `scripts/train.sh`

| Criterion    | Value                |
| ------------ | -------------------- |
| Memory Usage | 13575MiB / 11988 MiB |
| Throughput   | 3.00it/s             |

### DeepSpeed

### PyTorch-DDP

- `scripts/accel.sh`

| Criterion    | Value               |
| ------------ | ------------------- |
| Memory Usage | 15620iB / 15620 MiB |
| Throughput   | 3.34it/s            |

### PyTorch-FSDP
