BLAS Tutorial
=============

This page is the practical tour of the BLAS wrapper. Every snippet below comes from a compiled and executed test in
``doc/code/tutorial_blas.cpp``.

.. note::

   All routines are enqueued on the alpaka queue. On a default (non-blocking) queue they run asynchronously, whereas a
   blocking queue returns results synchronously. Call ``alpaka::onHost::wait(queue)`` before reading any output, and
   keep every operand view and result buffer alive (and mutable where it is written) until the queue has been waited.

Step 1: Select a backend and create a queue
-------------------------------------------

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-setup]
   :end-before: //! [blas-tutorial-setup]

Step 2: Level-1 vector routines
-------------------------------

This is the small toolbox you reach for all the time: copy data, swap buffers, scale in place, do an AXPY update, and
compute simple reductions.

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-level1]
   :end-before: //! [blas-tutorial-level1]

What this covers in one go:

- ``copy(queue, x, z)``
- ``swap(queue, x, y)``
- ``scal(queue, alpha, x)``
- ``axpy(queue, alpha, x, y)``
- ``dot(queue, x, y, result)``
- ``dotc(queue, x, y, result)``
- ``nrm2(queue, x, result)``
- ``asum(queue, x, result)``
- ``iamax(queue, x, result)``

``iamax`` returns a 1-based index; empty vectors (n == 0) return 0.

Alongside ``dot`` the tutorial computes ``dotc``, which is identical for the real-valued data used there and therefore
also yields ``28.0`` in ``dotcResult``. The difference shows up only for complex operands: ``dotc`` conjugates the
first operand, i.e. ``result[0] = sum_i conj(x[i]) * y[i]``, while ``dot`` multiplies the operands as stored.

Result buffers
``````````````

The reductions ``dot``, ``nrm2``, ``asum``, and ``iamax`` expect a single-element result view with a non-null data
pointer, and its element type must match the routine's result type:

- ``dot`` writes the vector scalar type (for complex input this is the complex scalar);
- ``nrm2`` and ``asum`` write the real type (``float`` for a complex float input);
- ``iamax`` writes a 32-bit signed integer (``int``); the wider oneMKL result width is tracked in issue #19.

If any of these do not hold the wrapper raises ``std::invalid_argument`` before touching a backend. The result is
written asynchronously, so call ``queue.wait()`` before reading it. On CUDA and HIP the reduction runs with the vendor
handle in device pointer mode, therefore the result buffer must be device-accessible (unified memory or a device
buffer); on SYCL the oneMKL ``iamax`` accesses the result from a ``host_task``, so the ``iamax`` result view must be
host-accessible (shared/unified) until issue #18/#19 are resolved, while the other reductions must be device-accessible
(shared/unified or USM device). Copy it back to the host after the wait if needed.

Step 3: GEMV with and without transpose
---------------------------------------

``gemv`` computes ``y = alpha * op(A) * x + beta * y``. The useful part is that ``op(A)`` can be the matrix as stored,
its transpose, or, where the selected backend supports it, its conjugate transpose.

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-gemv]
   :end-before: //! [blas-tutorial-gemv]

The first call uses ``A`` as-is. The second uses ``transposed(A)`` so the same buffer can be read as a ``3 x 2``
matrix without moving data.

``A`` must be row-major dense: the column stride must be exactly 1 and the leading dimension (the row stride) must be at
least the number of columns, otherwise ``std::invalid_argument`` is thrown.

Step 4: GEMM and transpose annotations
--------------------------------------

``gemm`` is the workhorse for matrix-matrix products:

``C = alpha * op(A) * op(B) + beta * C``

The first half of the example is the plain real-valued case. The second half shows ``conjTransposed(H)`` on a complex
matrix.

As for all 2D/3D BLAS routines here, the matrices must be row-major dense with column stride 1 and leading dimension
(row stride) at least ``cols``, otherwise ``std::invalid_argument`` is thrown.

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-gemm]
   :end-before: //! [blas-tutorial-gemm]

A small but important detail: the optional ``Options`` object is where backend-specific knobs live. If a backend does
not support a knob, the wrapper keeps the call valid and just ignores the hint.

Step 5: Triangular solve with side, triangle, and diagonal annotations
-----------------------------------------------------------------------

``trsm`` solves triangular systems in place. The matrix annotation tells the backend which half is valid and whether the
stored diagonal should be read at all.

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-trsm]
   :end-before: //! [blas-tutorial-trsm]

In this example:

- ``Side::left`` means ``op(A) * X = alpha * B``; ``Side::right`` would mean ``X * op(A) = alpha * B``
- ``lower(triangular)`` says only the lower half matters and sets ``Triangle::lower``; ``upper(A)`` or ``lower(A)`` is
  mandatory and a matrix without either annotation is rejected with ``std::invalid_argument`` at runtime
- ``unitDiag(...)`` says the diagonal is implicitly one and sets ``Diagonal::unit``; the stored diagonal values are
  ignored, while ``nonUnitDiag(...)`` reads the stored diagonal instead
- ``op(A)`` is selected by ``A``, ``transposed(A)``, or ``conjTransposed(A)``, so the transposed and conjugate-transposed
  combinations are available without copying ``A``
- ``A`` must be square, and ``B`` must match ``A.rows`` for ``Side::left`` or ``A.cols`` for ``Side::right`` after
  applying the transpose annotation

The exact enum values and the default of every annotation are listed in :ref:`blas-annotations-reference` on the BLAS
index page.

Step 6: Symmetric rank-k update (SYRK)
--------------------------------------

``syrk`` updates a symmetric matrix from a rank-k product:

``C = alpha * op(A) * op(A)^T + beta * C``

``op(A)`` on ``A`` may be ``transposed(A)`` or ``conjTransposed(A)``. Because conjugation is the identity on real
values, ``conjTransposed(A)`` is equivalent to ``transposed(A)`` and the wrapper normalizes it to the transposed
operation for real operands. The second factor is the plain transpose ``op(A)^T``, the real symmetric rank-k form.
Real scalar types ``float`` and ``double`` are supported.

The view on ``C`` must declare which triangle is updated:

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-syrk]
   :end-before: //! [blas-tutorial-syrk]

Only the selected triangle of ``C`` is written; the opposite triangle and any padding are left unchanged. The example
uses ``upper(C)``, so the diagonal counts as part of the selected triangle and the ``(1, 0)`` entry stays untouched.

Step 7: Hermitian rank-k update (HERK)
--------------------------------------

``herk`` is the complex Hermitian counterpart: it computes the selected triangle of
``C = alpha * op(A) * conjTranspose(op(A)) + beta * C``. The scalar coefficients must be real, ``A`` may be passed
as-is or as ``conjTransposed(A)`` (plain ``transposed(A)`` is not a standard HERK operation), and ``C`` must be
annotated ``upper(C)`` or ``lower(C)``. The result is Hermitian with a real diagonal, so on an actual update the
written diagonal's imaginary part is discarded.

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-herk]
   :end-before: //! [blas-tutorial-herk]

Step 8: Strided batched GEMM
----------------------------

If your data already lives in a ``[batch, row, column]`` view, ``stridedBatchedGemm`` applies the same matrix product
to every batch.

The wrapper validates the operands at runtime and throws ``std::invalid_argument`` when:

- the batch counts of ``A``, ``B``, and ``C`` do not match,
- ``op(A).cols`` does not equal ``op(B).rows``, or
- the extent of ``C`` does not match ``op(A).rows`` x ``op(B).cols``.

The batch stride is taken from the view's z-pitch, so naturally contiguous batches need no extra padding. The batched
views must be row-major dense (column stride 1).

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :start-after: //! [blas-tutorial-batched-gemm]
   :end-before: //! [blas-tutorial-batched-gemm]

That is often enough for small batched dense kernels without dropping down to vendor-specific APIs.

Each batch view must be row-major dense with column stride 1 and leading dimension (row stride) at least ``cols``, otherwise
``std::invalid_argument`` is thrown.

Complete example
----------------

.. raw:: html

   <details>
   <summary><strong>Full source: tutorial_blas.cpp</strong></summary>

.. literalinclude:: ../../../doc/code/tutorial_blas.cpp
   :language: C++
   :caption: tutorial_blas.cpp

.. raw:: html

   </details>
