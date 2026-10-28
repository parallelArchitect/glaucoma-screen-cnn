# AMD MI300X — CompactGlaucomaNet Characterization

**Project:** glaucoma-screen  
**Model:** CompactGlaucomaNet  
**Purpose:** Preserve controlled MI300X measurements obtained using the project's actual model architecture and trained checkpoint.

## 1. Environment

Measured runtime:

- GPU: AMD Instinct MI300X VF
- VRAM: 191.69 GiB
- Architecture: gfx942
- Compute units: 304
- PyTorch: 2.12.0+rocm7.14.0
- HIP: 7.14.60850
- amdgpu driver: 6.19.14.31400000
- Host kernel: 6.8.0-138-generic

The measured container runtime differed from the marketplace description, which advertised PyTorch 2.10.0 / ROCm 7.2.4. Results in this document use the measured runtime above.

## 2. Model Verification

The production architecture was copied from:

`src/glaucoma_screen/models/compact_net.py`

Measured model properties:

- Trainable parameters: 2,793,953
- Input: RGB `[N, 3, 128, 128]`
- Output: `[N, 1]`
- GPU forward execution: PASS

The trained checkpoint:

`models/checkpoints/glaucoma_model.pt`

was identified as a direct PyTorch state dictionary.

Strict loading into `CompactGlaucomaNet`:

`load_state_dict(..., strict=True)` → PASS

This verified that the benchmarked checkpoint and architecture were compatible.

## 3. GEMM Baseline

PyTorch matrix multiplication, N=16384.

### FP16

- Best: 575.70 TFLOPS
- Average: 572.17 TFLOPS
- Best latency: 15.279 ms
- Average latency: 15.373 ms

### BF16

- Best: 613.98 TFLOPS
- Average: 608.70 TFLOPS
- Best latency: 14.326 ms
- Average latency: 14.451 ms

These are measured PyTorch GEMM results on this MI300X VF, not vendor peak specifications.

## 4. Synthetic Training Method

Synthetic tensors with the production input geometry `[N, 3, 128, 128]` were resident on the GPU.

The measured step consisted of:

1. optimizer zero_grad
2. CompactGlaucomaNet forward
3. BCEWithLogitsLoss
4. backward
5. AdamW optimizer step
6. GPU synchronization

This isolates model-side GPU computation.

It does NOT include:

- JustRAIGS JPEG decoding
- dataset reads
- torchvision transforms
- DataLoader overhead
- host-to-device input transfer
- validation
- checkpoint writing

Therefore these measurements MUST NOT be reported as end-to-end JustRAIGS epoch throughput.

## 5. BF16 Synthetic Training

| Batch | Images/s | Step ms | Peak GiB |
|---:|---:|---:|---:|
| 32 | 2934.2 | 10.906 | 0.748 |
| 64 | 3264.1 | 19.607 | 1.325 |
| 128 | 3151.6 | 40.615 | 2.495 |
| 256 | 3124.1 | 81.944 | 4.819 |
| 512 | 3059.0 | 167.375 | 9.467 |
| 1024 | 3042.5 | 336.564 | 18.764 |

Peak measured throughput: **3264.1 images/s at batch 64**.

This was an experimental BF16 execution path. The current training implementation does not use BF16.

No model-quality or convergence claim is supported by this measurement.

## 6. Current FP32 Training Path

Inspection of `train.py` showed the current training implementation uses ordinary FP32 execution with AdamW. No autocast, GradScaler, FP16, or BF16 training path was present.

| Batch | Images/s | Step ms | Peak GiB |
|---:|---:|---:|---:|
| 32 | 2540.8 | 12.594 | 1.331 |
| 64 | 2791.8 | 22.924 | 2.477 |
| 128 | 2798.5 | 45.739 | 4.771 |
| 256 | 2780.2 | 92.081 | 9.357 |
| 512 | 2734.8 | 187.217 | 18.529 |

Peak measured FP32 synthetic training throughput: **2798.5 images/s at batch 128**.

BF16 produced higher synthetic training throughput, but BF16 training behavior has NOT been validated for convergence or classifier performance.

## 7. Trained Checkpoint FP32 Inference

The actual trained `glaucoma_model.pt` checkpoint was loaded strictly and evaluated in inference mode.

| Batch | Images/s | Latency ms | Peak GiB |
|---:|---:|---:|---:|
| 1 | 669.5 | 1.494 | 0.095 |
| 8 | 5583.7 | 1.433 | 0.164 |
| 32 | 9741.8 | 3.285 | 0.403 |
| 64 | 10670.1 | 5.998 | 0.722 |
| 128 | 10778.0 | 11.876 | 1.358 |
| 256 | 10716.4 | 23.889 | 2.632 |
| 512 | 10650.0 | 48.075 | 5.178 |
| 1024 | 10649.4 | 96.156 | 10.272 |

Measured throughput plateaued around batch 64-128.

Peak observed value: **10,778.0 images/s at batch 128**.

## 8. torch.compile — Inference

The trained checkpoint was compared using eager execution and `torch.compile`.

Batch: 128

- Eager average: 10,847.8 images/s
- Compiled average: 12,582.4 images/s
- Measured speedup: **1.160x**
- First compile/run: 6.067 s

Numerical comparison against eager output:

- `torch.allclose`: PASS
- maximum absolute difference: 2.8610229e-06

Observation:

`torch.compile` produced approximately 16% higher inference throughput in this experiment while satisfying the selected numerical agreement criterion.

## 9. torch.compile — FP32 Training

Batch: 128

- Eager: 2802.2 images/s
- Compiled: 3222.1 images/s
- Measured speedup: **1.150x**
- First compiled training step: 12.665 s

Observation:

Compilation produced approximately 15% higher synthetic FP32 training throughput in this experiment.

This is a performance measurement. Full training reproducibility and final model-quality equivalence were not evaluated.

## 10. FP32 Matmul Precision A/B

PyTorch emitted a warning that faster FP32 matrix multiplication was available and suggested:

`torch.set_float32_matmul_precision("high")`

A controlled comparison was performed.

- `highest`: 10,822.2 images/s
- `high`: 10,958.0 images/s
- Speedup: 1.013x
- Maximum absolute output difference: 0.004635334
- Selected `torch.allclose` test: FAIL

The measured throughput improvement was only approximately 1.3%, while the output comparison failed the selected tolerance.

No change to the project's FP32 matmul precision is justified by this experiment alone.

## 11. HBM Copy Measurement

A separate large-tensor device copy experiment measured:

| Allocation | Effective read+write bandwidth |
|---:|---:|
| 1 GiB | 3945.6 GB/s |
| 2 GiB | 3830.3 GB/s |
| 4 GiB | 3855.8 GB/s |
| 8 GiB | 3842.8 GB/s |
| 16 GiB | 3892.0 GB/s |

This is a synthetic device-memory copy measurement and is not a model-training throughput measurement.

## 12. Supported Observations

The evidence supports the following observations:

1. CompactGlaucomaNet executes successfully on the tested MI300X ROCm environment.
2. The trained checkpoint strictly matches the 2,793,953-parameter architecture.
3. BF16 synthetic training was faster than the current FP32 path in the tested configuration.
4. `torch.compile` accelerated both trained-checkpoint inference and synthetic FP32 training by approximately 15-16%.
5. The compiled inference output satisfied the selected numerical agreement test against eager execution.
6. Increasing model batch size far beyond the observed throughput plateau consumed additional HBM without corresponding throughput improvement.

## 13. Not Established

These experiments do NOT establish:

- end-to-end JustRAIGS training throughput
- actual epoch duration
- BF16 convergence equivalence
- BF16 AUC/sensitivity/specificity equivalence
- compiled-training reproducibility across a complete training run
- clinical equivalence
- general MI300X performance outside the tested software/runtime configuration

Any production change requires validation using the preserved real dataset and the project's normal validation procedure.

## 14. Preserved Evidence

Raw measurement record:

`mi300x_glaucoma_results.txt`

SHA-256:

`71d87152a7e788e1b96edd11836620eeaae9c81a26c42bc3e5dbdf5e27f0771e`

The remote and local copies produced identical SHA-256 digests before the cloud instance was released.
