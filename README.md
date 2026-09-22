# alpakaVendor

**Portable vendor library abstractions for heterogeneous computing.**

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![alpaka](https://img.shields.io/badge/alpaka-3.x-orange.svg)](https://github.com/alpaka-group/alpaka)

alpakaVendor is a header-only C++20 library that provides portable, type-safe abstractions for vendor-optimized libraries on top of the [alpaka](https://github.com/alpaka-group/alpaka) accelerator abstraction layer. 
Write once, run on any hardware — CPU, NVIDIA GPU, AMD GPU, or Intel GPU.

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| **FFT** | Available | 1D/2D/3D C2C, R2C/C2R transforms with batched and in-place support |
| **BLAS** | Available | BLAS Level-1, GEMV, GEMM, strided batched GEMM, SYRK, and TRSM |
| **Parallel Primitives** | Planned | Elementwise transforms, sorting, scans, reductions, and related building blocks |

## Features

- **Portable**: Single codebase targeting CPU (FFTW/OpenBLAS), NVIDIA GPU (cuFFT/cuBLAS), AMD GPU (rocFFT/rocBLAS), and Intel GPU (oneMKL DFT/BLAS) through alpaka's backend abstraction.
- **Type-safe**: Strong typing for real and complex value types with compile-time dimension selection (1D, 2D, 3D).
- **BLAS abstractions**: Portable BLAS operations for host and accelerator backends using alpaka queues and views.
- **RAII-based**: Plans are RAII objects that own backend handles and clean up automatically.
- **Batched transforms**: Efficient batched FFT operations with configurable strides and distances.
- **In-place real transforms**: Safe in-place R2C/C2R transforms with automatic padding management and real/complex view conversion via `SharedBufferFFT`.
- **Zero-overhead**: Header-only with no runtime overhead beyond the backend library calls.

## Quick Start

```cpp
#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>

#include <cmath>
#include <numbers>

// Create device and queue
auto dev = alpaka::onHost::makeHostDevice();
auto queue = dev.makeQueue();

// Allocate FFT-managed buffers
constexpr uint32_t N = 1024;
constexpr float frequency = 5.0f;  // signal frequency in cycles per sample
auto in  = alpaka::fft::onHost::alloc<float>(dev, N);
auto out = alpaka::fft::onHost::alloc<alpaka::math::Complex<float>>(dev, N);

// Generate a sine wave
for(uint32_t i = 0; i < N; ++i)
    in.data()[i] = std::sin(2.0f * std::numbers::pi_v<float> * frequency * float(i) / float(N));

// Create plan and execute R2C transform
auto plan = alpaka::fft::onHost::makePlan<float>(N)
                .r2c()
                .outOfPlace()
                .build(dev);

alpaka::fft::onHost::executeForward(queue, plan, in, out);
alpaka::onHost::wait(queue);

// The spectrum peak should be at bin 'frequency'
```

## Requirements

- C++20 compiler (GCC 11+, Clang 14+, nvcc 12+, icpx 2025.0+)
- CMake 3.25+
- [alpaka 3.x](https://github.com/alpaka-group/alpaka)
- [FFTW 3.x](http://www.fftw.org/) and OpenBLAS (for CPU backends)
- CUDA Toolkit with cuFFT/cuBLAS (for NVIDIA GPU backends, optional)
- ROCm with rocFFT/rocBLAS (for AMD GPU backends, optional)
- oneMKL DFT/BLAS and oneAPI (for oneAPI CPU or Intel GPU backends, optional)

## Installation

### Using CMake

```bash
cmake -B build -DalpakaV_DEP_FFTW=ON -DalpakaV_DEP_CUFFT=OFF -DalpakaV_DEP_ROCFFT=OFF
cmake --build build
ctest --test-dir build
```

### As a subdirectory

```cmake
add_subdirectory(alpakaVendor)
target_link_libraries(your_target PRIVATE alpakaVendor::alpakaVendor)
```

### Via FetchContent

```cmake
FetchContent_Declare(
    alpakaVendor
    GIT_REPOSITORY https://github.com/alpaka-group/alpakaVendor.git
    GIT_TAG main
)
FetchContent_MakeAvailable(alpakaVendor)
target_link_libraries(your_target PRIVATE alpakaVendor::alpakaVendor)
```

## CMake Options

| Option | Default          | Description |
|--------|------------------|-------------|
| `alpakaV_DEP_FFTW` | `ON`             | Enable FFTW host backend |
| `alpakaV_DEP_OPENBLAS` | `ON`         | Enable OpenBLAS host backend |
| `alpakaV_DEP_CUFFT` | `OFF`            | Enable cuFFT CUDA backend |
| `alpakaV_DEP_CUBLAS` | `OFF`          | Enable cuBLAS CUDA backend |
| `alpakaV_DEP_ROCFFT` | `OFF`            | Enable rocFFT HIP backend |
| `alpakaV_DEP_ROCBLAS` | `OFF`         | Enable rocBLAS HIP backend |
| `alpakaV_DEP_ONEMKL` | `OFF`            | Enable oneMKL DFT and BLAS oneAPI backends |
| `alpakaV_TESTS` | `ON` (top-level) | Build tests |
| `alpakaV_DOCS` | `OFF`            | Build documentation |

## Backend Selection

The backend is inferred from the device you pass to `build()`:

```cpp
auto builder = alpaka::fft::onHost::makePlan<float>(1024u).r2c();

// CPU (FFTW)
auto hostDevice = alpaka::onHost::makeHostDevice();
auto hostPlan = builder.build(hostDevice);

// NVIDIA GPU (cuFFT)
auto cudaDevSelector = alpaka::onHost::makeDeviceSelector(alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu);
auto cudaDevice = cudaDevSelector.makeDevice(0);
auto cudaPlan = builder.build(cudaDevice);

// AMD GPU (rocFFT)
auto hipDevSelector = alpaka::onHost::makeDeviceSelector(alpaka::api::hip, alpaka::deviceKind::amdGpu);
auto hipDevice = hipDevSelector.makeDevice(0);
auto hipPlan = builder.build(hipDevice);

// Intel oneAPI CPU (oneMKL DFT)
auto oneApiDevSelector = alpaka::onHost::makeDeviceSelector(alpaka::api::oneApi, alpaka::deviceKind::cpu);
auto oneApiDevice = oneApiDevSelector.makeDevice(0);
auto oneApiPlan = builder.build(oneApiDevice);
```

## Memory Layout

alpakaVendor follows alpaka's memory layout convention: **the last index is the fast-moving index** (row-major order).

```cpp
// 2D M x N transform: index 0 (M) is slow-moving, index 1 (N) is fast-moving
auto extents = alpaka::Vec<std::size_t, 2u>{M, N};
data[row * N + col] = value;  // row-major
```

## Documentation

Full documentation is available at [alpaka-group.github.io/alpakaVendor](https://alpaka-group.github.io/alpakaVendor/) (hosted on GitHub Pages).

Build locally:

```bash
cmake -B build -DalpakaV_DOCS=ON
cmake --build build --target docs
```

> **Note**: The documentation was generated with the assistance of an AI coding tool. While it has been reviewed for accuracy, please report any issues or unclear sections.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Mozilla Public License 2.0 — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [alpaka](https://github.com/alpaka-group/alpaka) — The accelerator abstraction layer this project builds upon
- [FFTW](http://www.fftw.org/) — The Fastest Fourier Transform in the West
- [cuFFT](https://developer.nvidia.com/cufft) — NVIDIA's FFT library
