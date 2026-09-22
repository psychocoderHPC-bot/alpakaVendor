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
- **Level 3:** ``gemm``, ``stridedBatchedGemm``, ``syrk``, ``trsm``

How to read the BLAS views
--------------------------

The wrappers work directly with alpaka mdspan-like buffers and views:

- 1D views are treated as vectors
- 2D views are treated as matrices
- 3D views are treated as ``[batch, row, column]`` for strided batched GEMM

As in alpaka, the last index is the contiguous one. For a matrix ``A(rows, cols)``, ``A[{r, c}]`` means row ``r`` and
column ``c``.

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

SYRK: symmetric rank-k update
-----------------------------

``syrk`` computes the selected triangle of

``C = alpha * op(A) * op(A)^T + beta * C``

with ``op(A)`` the transpose (or, for real operands equivalently, the conjugate transpose) of the stored matrix
``A`` of shape ``n x k`` and ``C`` ``n x n``; only the triangle selected by ``upper(C)`` or ``lower(C)`` is updated.
The formula is the real symmetric rank-k form: the second factor is the plain transpose ``op(A)^T`` (never a
Hermitian/conjugate-transposed right-hand side, which is the domain of the complex ``herk`` routine). The opposite
triangle and any padding are left unchanged.

- Real scalar types ``float`` and ``double`` only.
- ``A`` may be annotated ``transposed(A)`` or ``conjTransposed(A)``; for real operands ``conjTransposed(A)`` is
  equivalent to ``transposed(A)`` (conjugation is the identity on real types) and is normalized to the transposed
  operation.
- ``alpha`` and ``beta`` are always converted exactly once, at the public entry, into the canonical scalar type of the
  operands; the backend dispatch receives the already-converted values and never re-casts them.
- Backends: OpenBLAS/CBLAS host, CUDA/cuBLAS, HIP/rocBLAS, and oneAPI/oneMKL.
- Row-major handling: the views follow alpaka's memory layout (last index is contiguous), and the wrappers perform the
  necessary layout translation for the vendor libraries.

Options for SYRK (what each backend honors):

- OpenBLAS/CBLAS host: ``Precision`` and ``Algorithm`` are accepted and currently ignored.
- CUDA/cuBLAS: ``Precision::exact`` selects the pedantic math mode for single-precision SYRK. ``Algorithm``:
  ``deterministic`` disables cuBLAS atomics for the SYRK call and ``fastest`` enables them.
- HIP/rocBLAS: ``Precision`` is accepted and currently ignored. ``Algorithm``: ``deterministic`` disables rocBLAS
  atomics for the SYRK call and ``fastest`` enables them when the rocBLAS handle exposes atomics mode.
- oneAPI/oneMKL: ``Precision::exact`` requests the oneMKL standard compute mode and ``Algorithm::deterministic`` the
  standard mode as well; ``Algorithm::fastest`` requests the oneMKL alternate compute mode for single-precision SYRK
  when oneMKL supports it (best-effort, falling back to the routine default otherwise).

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
