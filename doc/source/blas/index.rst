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

``iamax`` returns a 1-based index of the entry with the largest absolute value; an empty vector (n == 0)
returns 0.

How to read the BLAS views
--------------------------

The wrappers work directly with alpaka mdspan-like buffers and views:

- 1D views are treated as vectors
- 2D views are treated as matrices
- 3D views are treated as ``[batch, row, column]`` for strided batched GEMM, with the batch stride given by the view's
  z-pitch

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

Queues and lifetimes
--------------------

All ``alpaka::blas`` routines are enqueued on the alpaka queue. On a default (non-blocking) queue they complete
**asynchronously**: the call only enqueues work and may return before the computation has completed. A blocking queue
returns only after the enqueued work has completed.

- Wait for the queue with ``alpaka::onHost::wait(queue)`` before reading any result, including the single-element
  reduction or result buffers that receive the outputs of ``dot``, ``nrm2``, ``asum``, and ``iamax``.
- Every operand view (inputs and outputs) and the result buffer must stay alive until the enqueued work has completed.
  Views written by the backend must also be mutable.
- The wrappers capture the operand views and data pointers when the routine is called; resizing, reallocating, or
  destroying a buffer before the queue is waited leaves the enqueued work with dangling pointers.

For accelerator backends the result buffer must additionally be accessible to the backend that writes it. CUDA and HIP
compute the scalar result on the device, so the buffer must be device-accessible. For oneAPI ``iamax``, oneMKL writes
the index on the device and a subsequent host task dereferences it
(``include/alpaka/blas/internal/api/oneapi/blas.hpp``), so the result buffer must also be host-accessible (for example
shared or host USM). Host results are ordinary host-visible buffers.

Reduction result buffers
------------------------

``dot``, ``nrm2``, ``asum``, and ``iamax`` write their scalar result into a single-element output view. The wrapper
validates that contract before dispatching:

- the view must have exactly one element and a non-null data pointer;
- its element type must match the routine's result type: ``dot`` uses the vector scalar type, ``nrm2`` and ``asum``
  use the corresponding real type, and ``iamax`` uses a 32-bit signed integer (``int``); the wider oneMKL result width
  is tracked in issue #19;

A violation raises ``std::invalid_argument`` on the host before any backend call.

The result is produced asynchronously on the queue. On CUDA and HIP the vendor libraries are placed in *device* pointer
mode internally for the reduction and the previous pointer mode is restored afterwards, so the result pointer must be
device-accessible. On SYCL the oneMKL ``iamax`` implementation increments the result inside a ``host_task`` and thus
accesses it from the host, so the ``iamax`` result view must be host-accessible (shared/unified) until issue #18/#19 are
resolved. Other SYCL reductions must be device-accessible (shared/unified or USM device).
Use unified memory or copy the result back to the host after ``queue.wait()``.

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

Triangular solves with ``trsm``
-------------------------------

``trsm(queue, side, alpha, A, B)`` solves a triangular system with multiple right-hand sides and overwrites ``B`` with
the solution ``X``:

- ``Side::left`` solves ``op(A) * X = alpha * B``
- ``Side::right`` solves ``X * op(A) = alpha * B``

Here ``op(A)`` is ``A``, ``transposed(A)``, or ``conjTransposed(A)`` depending on the annotation stacked onto ``A``.
``transposed`` uses the plain transpose and ``conjTransposed`` uses the conjugate (Hermitian) transpose, so the
transposed combinations are selected purely through annotations and no temporary copy of ``A`` is created.

``A`` must be annotated with ``upper(A)`` or ``lower(A)`` to select the stored triangular half. Passing a matrix
without such an annotation (``Triangle::full``) is rejected at runtime with ``std::invalid_argument``. The
``unitDiag(A)`` and ``nonUnitDiag(A)`` annotations select the diagonal semantics: ``unitDiag`` treats every diagonal
entry as implicitly one and ignores the stored values, while ``nonUnitDiag`` (the default) reads the stored diagonal.

``A`` must also describe a square matrix, i.e. the order of ``op(A)`` must have equal row and column extents. ``B``
then has to match the triangular operand: ``A.rows == B.rows`` for ``Side::left`` and ``A.rows == B.cols`` for
``Side::right``. Violations of these requirements are reported as ``std::invalid_argument``.

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
     - ``exact`` selects pedantic math mode for single-precision real and complex routines; this math mode is applied to
       every dispatched routine, including the Level-1 calls. GEMM and strided batched GEMM also pass pedantic compute
       types for ``float``, ``double``, and complex variants.
     - ``deterministic`` disables cuBLAS atomics and ``fastest`` enables them for GEMM, strided batched GEMM, GEMV, and
       TRSM.
   * - HIP / rocBLAS
     - Accepted, currently ignored by the implemented rocBLAS calls.
     - ``deterministic`` disables rocBLAS atomics and ``fastest`` enables them for GEMM, strided batched GEMM, GEMV, and
       TRSM when the rocBLAS handle exposes atomics mode.
   * - oneAPI / oneMKL
     - ``exact`` requests oneMKL standard compute mode for GEMM, batched GEMM, and TRSM. GEMV ignores ``Options``, so
       ``exact`` has no effect there.
     - ``deterministic`` requests standard compute mode. ``fastest`` requests oneMKL alternate compute mode for
       single-precision real and complex GEMM, batched GEMM, and TRSM paths when oneMKL supports it. GEMV ignores
       ``Options``, so no algorithm hint is applied there.

Routine applicability is best-effort and routine-specific: the per-routine notes above and in the SYRK/herk sections
are authoritative. In particular, the oneAPI GEMV path ignores all ``Options``, while the CUDA math mode selected by
``Precision`` is applied to every dispatched single-precision routine, Level-1 included.

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

.. _blas-annotations-reference:

Annotations and options
-----------------------

This section is the compact reference for the enum types, their exact C++ names, and the defaults behind the
annotation helpers. All types live in ``alpaka::blas``.

Enumerations
++++++++++++

.. code-block:: cpp

   namespace alpaka::blas
   {
       enum class Transpose { none, transposed, conjugateTransposed };
       enum class Triangle { full, upper, lower };
       enum class Diagonal { nonUnit, unit };
       enum class Side { left, right };
       enum class Precision { exact, backendDefault };
       enum class Algorithm { backendDefault, deterministic, fastest };
   }

.. list-table:: Enum values and defaults
   :header-rows: 1

   * - Enum
     - Values (declaration order)
     - Default
   * - ``Transpose``
     - ``none``, ``transposed``, ``conjugateTransposed``
     - ``Transpose::none``
   * - ``Triangle``
     - ``full``, ``upper``, ``lower``
     - ``Triangle::full``
   * - ``Diagonal``
     - ``nonUnit``, ``unit``
     - ``Diagonal::nonUnit``
   * - ``Side``
     - ``left``, ``right``
     - No default; passed explicitly to ``trsm``
   * - ``Precision``
     - ``exact``, ``backendDefault``
     - ``Precision::exact``
   * - ``Algorithm``
     - ``backendDefault``, ``deterministic``, ``fastest``
     - ``Algorithm::backendDefault``

Options
+++++++

``Options`` is an aggregate and is the default argument of the execution entry points, so default-constructing it
gives:

.. code-block:: cpp

   struct Options
   {
       Precision precision = Precision::exact;
       Algorithm algorithm = Algorithm::backendDefault;
   };

Annotation defaults
+++++++++++++++++++

The helpers wrap a view in ``AnnotatedView<T_View>``, whose members carry the interpretation and start at:

.. code-block:: cpp

   struct AnnotatedView
   {
       T_View view;
       Transpose transpose = Transpose::none;
       Triangle triangle = Triangle::full;   // full is the default
       Diagonal diagonal = Diagonal::nonUnit;
   };

A plain (unannotated) view behaves exactly like those defaults: no transpose, the full matrix, and the stored
diagonal.

Helper to value mapping
+++++++++++++++++++++++

.. list-table:: Annotation helpers and the enum value they set
   :header-rows: 1

   * - Helper
     - Sets
   * - ``transposed(A)``
     - ``Transpose::transposed``
   * - ``conjTransposed(A)``
     - ``Transpose::conjugateTransposed``
   * - ``upper(A)``
     - ``Triangle::upper``
   * - ``lower(A)``
     - ``Triangle::lower``
   * - ``unitDiag(A)``
     - ``Diagonal::unit``
   * - ``nonUnitDiag(A)``
     - ``Diagonal::nonUnit``

There is no helper for ``Transpose::none`` or ``Triangle::full``; leave the view unannotated, or simply do not apply
that helper, to get the default. ``Side`` is not an annotation: it is a required argument of ``trsm``.

Stacking and conflicting annotations
+++++++++++++++++++++++++++++++++++++

Annotations are cumulative. Applying a second helper starts from the already-annotated view, so independent kinds
combine, for example ``unitDiag(lower(A))`` or ``conjTransposed(upper(A))``.

- Re-applying ``upper``/``lower`` or ``unitDiag``/``nonUnitDiag`` keeps the last value; the diagonal helpers therefore
  overwrite each other. The transpose helpers also keep the last value, so ``transposed(conjTransposed(A))`` ends as
  ``Transpose::transposed`` and ``conjTransposed(transposed(A))`` ends as ``Transpose::conjugateTransposed``.
- Mixing ``upper()`` and ``lower()`` on the same view is rejected at the second call, which throws
  ``std::invalid_argument``. Applying ``upper()`` after ``lower()`` throws
  ``"Conflicting triangular annotation: lower then upper."``; applying ``lower()`` after ``upper()`` throws
  ``"Conflicting triangular annotation: upper then lower."``
- ``trsm`` additionally requires a triangular annotation. If neither ``upper(A)`` nor ``lower(A)`` was applied, so
  the triangle is still ``Triangle::full``, it throws
  ``std::invalid_argument("Triangular BLAS operations require upper(A) or lower(A).")``.
