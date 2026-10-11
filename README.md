# Conv2D and ReLU kernels in C++

Three ways to compute a 3x3 convolution, plus a study of what fusing convolution, bias and ReLU
into one pass is worth.

- **Direct**: a naive reference, and a NEON kernel vectorised across the output row
- **im2col + GEMM**: lower the convolution to one matrix multiply
- **Winograd F(2x2, 3x3)**: 16 multiplications per 2x2 output tile instead of 36

## Results

Time for one 3x3 convolution (stride 1, padding 1), single thread, in milliseconds:

| Layer (in -> out @ size) | Direct naive | Direct NEON | im2col + GEMM | Winograd | Winograd vs im2col |
|---|---:|---:|---:|---:|---:|
| 3 -> 64 @ 224 | 81.6 | 2.87 | 3.19 | 6.49 | 0.49x |
| 64 -> 64 @ 56 | 101.1 | 7.11 | 2.85 | 1.60 | 1.78x |
| 128 -> 128 @ 28 | 105.4 | 10.93 | 2.57 | 1.37 | 1.87x |
| 256 -> 256 @ 14 | 97.2 | 29.42 | 2.55 | 1.58 | 1.62x |
| 512 -> 512 @ 7 | 100.6 | 59.75 | 3.36 | 2.98 | 1.13x |

No algorithm wins everywhere: the direct kernel is best with 3 input channels, Winograd on the
deeper layers. All outputs are checked against the naive reference.

Fusing conv + bias + ReLU into one pass, against three separate passes (same direct NEON kernel):

| Layer | Output tensor | Speed-up |
|---|---:|---:|
| 1 -> 16 @ 1024 | 64 MB | 1.72x |
| 3 -> 32 @ 512 | 32 MB | 1.12x |
| 3 -> 64 @ 224 | 12 MB | 1.10x |
| 64 -> 64 @ 56 and deeper | under 1 MB | 1.00x to 1.01x |

Fusion only pays where the convolution is cheap relative to the size of its output. On layers
with 64 or more input channels it makes no measurable difference.

Measured on an Apple M3 Pro (5 performance + 6 efficiency cores), Apple clang 21, `-O3 -mcpu=native`, no `-ffast-math`. Raw output is in `results/`.

## How Winograd is implemented

```
Y = A^T [ (G g G^T) .* (B^T d B) ] A      g: 3x3 kernel, d: 4x4 input tile, Y: 2x2 output tile
```

1. Kernel transform `U = G g G^T`, once, offline.
2. Input transform `V = B^T d B` on overlapping 4x4 tiles, four tiles at a time with NEON.
3. Sixteen GEMMs, `M = U * V`, one per transformed position: the only multiplications by weights.
4. Output transform `Y = A^T M A`.

The transforms are C++ templates over the value type, instantiated with `float` and with
`float32x4_t`, so the scalar and vector versions come from the same source.

## Requirements

- macOS on Apple Silicon (the SIMD code is ARM NEON) with Apple clang
- Homebrew `libomp`

```sh
brew install libomp
```

## Build and run

```sh
make          # build bench_conv
make run      # run the benchmark -> results/
```

## Files

| File | Contents |
|---|---|
| `conv.h` | `direct`, `direct3x3`, `im2col`, `im2col_gemm`, `winograd3x3`, bias/ReLU passes, `maxpool2x2` |
| `bench_conv.cpp` | Algorithm comparison and the fusion study |
| `deps/gemm.h` | The matrix multiply from [fast-gemm-cpp](https://github.com/SritarunLakkaraju/fast-gemm-cpp) |
| `results/` | Benchmark output from the test machine |

## Limitations

- The optimised kernels cover 3x3, stride 1 (im2col handles any kernel, stride and padding).
- One image at a time, single-threaded benchmarks, ARM NEON only.
- Fusion is implemented inside the direct kernel; the GEMM-based paths apply bias and ReLU in one
  combined pass afterwards.

## Part of a series

Five projects that build on each other:

1. [fast-gemm-cpp](https://github.com/SritarunLakkaraju/fast-gemm-cpp) - matrix multiply, the kernel everything else uses
2. [conv2d-kernels-cpp](https://github.com/SritarunLakkaraju/conv2d-kernels-cpp) - convolution on top of that GEMM
3. [tiny-inference-engine-cpp](https://github.com/SritarunLakkaraju/tiny-inference-engine-cpp) - a small inference engine on top of both
4. [int8-quantized-kernels-cpp](https://github.com/SritarunLakkaraju/int8-quantized-kernels-cpp) - the same ideas in 8-bit integers
5. [transformer-kernels-cpp](https://github.com/SritarunLakkaraju/transformer-kernels-cpp) - softmax, LayerNorm and attention

Each repository is self-contained: the headers it needs from the earlier projects are copied into `deps/`.
