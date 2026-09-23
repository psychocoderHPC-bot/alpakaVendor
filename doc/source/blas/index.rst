BLAS
====

alpakaVendor provides a BLAS layer under ``alpaka::blas`` with execution entry points in
``alpaka::blas::onHost``.

The goal is simple: keep the call sites close to plain BLAS, but let you hand alpaka queues and alpaka views to the
wrapper instead of translating everything yourself.

Topics
------

.. toctree::
   :maxdepth: 1

   blas_tutorial.rst

What is available today?
------------------------

- **Level 1:** ``copy``, ``swap``, ``scal``, ``axpy``, ``dot``, ``nrm2``, ``asum``, ``iamax``
- **Level 2:** ``gemv``
- **Level 3:** ``gemm``, ``stridedBatchedGemm``, ``trsm``

How to read the BLAS views
--------------------------

The wrappers work directly with alpaka mdspan-like buffers and views:

- 1D views are treated as vectors
- 2D views are treated as matrices
- 3D views are treated as ``[batch, row, column]`` for strided batched GEMM

As in alpaka, the last index is the contiguous one. For a matrix ``A(rows, cols)``, ``A[{r, c}]`` means row ``r`` and
column ``c``.

Views passed to the 2D and 3D BLAS routines must be row-major dense: the column stride must be exactly 1 and the leading
dimension (the row stride) must be at least ``cols``. Violations raise ``std::invalid_argument``.

Quick example
-------------

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-setup]
   :end-before: //! [blas-tutorial-setup]

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-gemm]
   :end-before: //! [blas-tutorial-gemm]

Annotations instead of data copies
----------------------------------

The public helpers let you describe how an existing view should be interpreted:

- ``transposed(A)``
- ``conjTransposed(A)``
- ``upper(A)`` / ``lower(A)``
- ``unitDiag(A)`` / ``nonUnitDiag(A)``

These annotations can be stacked. For example, ``unitDiag(lower(A))`` marks a lower-triangular matrix whose diagonal is
implicitly one, and ``conjTransposed(A)`` asks BLAS to use the Hermitian transpose without creating a temporary copy.
Complex ``gemv`` with ``conjTransposed(A)`` is currently not available on the CUDA/cuBLAS and HIP/rocBLAS row-major paths.

Backend notes
-------------

- Real scalars ``float`` and ``double`` are supported.
- Complex values use ``alpaka::math::Complex``.
- Host execution is backed by OpenBLAS / CBLAS when enabled.
- CUDA, HIP, and oneAPI backends are mapped to the corresponding vendor BLAS libraries when those backends are built.
- ``alpaka::blas::Options`` carries backend hints such as math mode or algorithm selection. Backends that do not expose
  those knobs simply ignore the hint.

Backend option behavior
-----------------------

``Options`` values are best-effort hints, not a promise of bit-identical behavior across vendor libraries. Current backend
handling is:

.. list-table:: BLAS option handling by backend
   :header-rows: 1

   * - Backend
     - ``Precision``
     - ``Algorithm``
   * - OpenBLAS / CBLAS host
     - Accepted, currently ignored.
     - Accepted, currently ignored.
   * - CUDA / cuBLAS
     - ``exact`` selects pedantic math mode for single-precision real and complex routines. GEMM and strided batched GEMM
       also pass pedantic compute types for ``float``, ``double``, and complex variants.
     - ``deterministic`` disables cuBLAS atomics and ``fastest`` enables them for GEMM, strided batched GEMM, GEMV, and
       TRSM.
   * - HIP / rocBLAS
     - Accepted, currently ignored by the implemented rocBLAS calls.
     - ``deterministic`` disables rocBLAS atomics and ``fastest`` enables them for GEMM, strided batched GEMM, GEMV, and
       TRSM when the rocBLAS handle exposes atomics mode.
   * - oneAPI / oneMKL
     - ``exact`` requests oneMKL standard compute mode for GEMM, batched GEMM, and TRSM.
     - ``deterministic`` requests standard compute mode. ``fastest`` requests oneMKL alternate compute mode for
       single-precision real and complex GEMM, batched GEMM, and TRSM paths when oneMKL supports it.
