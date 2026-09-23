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

- **Level 1:** ``copy``, ``swap``, ``scal``, ``axpy``, ``dot``, ``dotc``, ``nrm2``, ``asum``, ``iamax``
- **Level 2:** ``gemv``
- **Level 3:** ``gemm``, ``stridedBatchedGemm``, ``syrk``, ``herk``, ``trsm``

How to read the BLAS views
--------------------------

The wrappers work directly with alpaka mdspan-like buffers and views:

- 1D views are treated as vectors
- 2D views are treated as matrices
- 3D views are treated as ``[batch, row, column]`` for strided batched GEMM

As in alpaka, the last index is the contiguous one. For a matrix ``A(rows, cols)``, ``A[{r, c}]`` means row ``r`` and
column ``c``. Row and batch byte pitches must be exact multiples of the element size; non-multiple pitches throw
``std::invalid_argument``.

Routine reference
-----------------

.. list-table:: BLAS routines
   :header-rows: 1
   :widths: 20 45 35

   * - Routine
     - Operation
     - Scalar types
   * - ``dot``
     - ``result[0] = sum_i x[i] * y[i]``
     - ``float``, ``double``, ``alpaka::math::Complex<float>``, ``alpaka::math::Complex<double>``
   * - ``dotc``
     - ``result[0] = sum_i conj(x[i]) * y[i]`` (first operand conjugated)
     - ``float``, ``double``, ``alpaka::math::Complex<float>``, ``alpaka::math::Complex<double>``

``dotc`` maps to the vendor conjugate-dot-product routines (``*dotc`` elsewhere) and, like ``dot``, is
available on the OpenBLAS/CBLAS host, cuBLAS, rocBLAS, and oneMKL host paths. It is not provided for OpenMP or the
generic native alpaka CPU queues.

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
HERK: Hermitian rank-k update
-----------------------------

``herk`` computes the selected triangle of

``C = alpha * op(A) * op(A)^H + beta * C``

with ``op(A)`` the as-stored matrix or its conjugate transpose, of shape ``n x k``, and ``C`` ``n x n``; only the
triangle selected by ``upper(C)`` or ``lower(C)`` is updated. The formula is the complex Hermitian rank-k form: the
second factor is the conjugate transpose ``op(A)^H``.

- Complex scalar types ``alpaka::math::Complex<float>`` and ``alpaka::math::Complex<double>`` only.
- ``A`` may be annotated ``conjTransposed(A)`` or left plain; the plain ``transposed(A)`` annotation is rejected
  because it is not a standard HERK operation.
- ``alpha`` and ``beta`` must be real values; complex coefficients are rejected.
- On an actual update the written diagonal is real (its imaginary part is discarded); the opposite triangle and any
  padding are left unchanged.
- ``k == 0`` or ``alpha == 0`` produce ``beta * C`` on the selected triangle without reading ``A``; the degenerate
  path is a queued triangle-scale kernel, so it stays ordered with respect to other work on the same queue.
- Backends: OpenBLAS/CBLAS host, CUDA/cuBLAS, HIP/rocBLAS, and oneAPI/oneMKL.


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

.. -- begin issue-37 Options subsection --

Options
-------

Every BLAS entry point takes an optional ``alpaka::blas::Options`` argument that carries backend hints. The default
value is:

.. code-block:: cpp

   namespace alpaka::blas
   {
       enum class Precision
       {
           exact,
           backendDefault
       };

       enum class Algorithm
       {
           backendDefault,
           deterministic,
           fastest
       };

       struct Options
       {
           Precision precision = Precision::exact; ///< Preferred math mode when the backend supports one.
           Algorithm algorithm = Algorithm::backendDefault; ///< Preferred backend algorithm, if selectable.
       };
   }

The two fields are therefore:

- ``precision`` -- ``Precision::exact`` by default. ``exact`` asks for the precise scalar type requested by the user,
  while ``Precision::backendDefault`` lets the backend choose its own default math mode.
- ``algorithm`` -- ``Algorithm::backendDefault`` by default. ``Algorithm::deterministic`` and
  ``Algorithm::fastest`` let a backend trade reproducibility against speed when it exposes such a knob.

Options are best-effort hints: they are never required for a call to be valid, and a backend that has no matching knob
simply ignores the field.

Host backends
~~~~~~~~~~~~~

The OpenBLAS / CBLAS host backends **accept but ignore all options**. The host dispatch functions take the ``Options``
argument to keep the public signature uniform and mark it ``[[maybe_unused]]``; neither ``precision`` nor ``algorithm``
is read. Passing the defaults, ``backendDefault``, or any other combination therefore has no effect on host execution.

.. -- end issue-37 Options subsection --
