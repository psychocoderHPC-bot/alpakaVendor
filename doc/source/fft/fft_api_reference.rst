FFT API Reference
==================

This page documents the public FFT API of alpakaVendor.

Common types
------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       enum class Transform { c2c, r2c, c2r };
       enum class Direction { forward, backward };
       enum class Normalization { none };
       enum class Placement { inPlace, outOfPlace };
       enum class WorkspacePolicy { backendManaged, userProvided };

       template<typename T_Index, uint32_t T_dim>
       using Extents = alpaka::Vec<T_Index, T_dim>;

       template<typename T_Index, uint32_t T_dim>
       using Strides = alpaka::Vec<T_Index, T_dim>;
   }

Transform types
+++++++++++++++

- ``Transform::c2c``: Complex-to-complex transform
- ``Transform::r2c``: Real-to-complex transform
- ``Transform::c2r``: Complex-to-real transform

Placement
+++++++++

- ``Placement::outOfPlace``: Input and output in separate buffers (default)
- ``Placement::inPlace``: Input and output share the same buffer

Workspace policy
++++++++++++++++

- ``WorkspacePolicy::backendManaged``: Backend allocates workspace automatically (default)
- ``WorkspacePolicy::userProvided``: User provides workspace via ``setWorkspace()``

Layout
------

.. code-block:: cpp

   namespace alpaka::fft
   {
       template<alpaka::concepts::Vector T_Extents>
       struct Layout
       {
           T_Extents extents{};
           alpaka::Vec<std::size_t, T_Extents::dim()> inStrides{};   ///< Byte-strides (alpaka pitch order)
           alpaka::Vec<std::size_t, T_Extents::dim()> outStrides{};  ///< Byte-strides (alpaka pitch order)
           alpaka::GetValueType_t<T_Extents> batch = 1u;
           std::size_t inDistance = 0u;   ///< Byte distance between input batches
           std::size_t outDistance = 0u;  ///< Byte distance between output batches
       };
   }

Plan options
------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       struct PlanOptions
       {
           Normalization normalization = Normalization::none;
           Placement placement = Placement::outOfPlace;
           WorkspacePolicy workspacePolicy = WorkspacePolicy::backendManaged;
       };
   }

Type traits
-----------

.. code-block:: cpp

   namespace alpaka::fft
   {
       template<typename T>
       concept RealScalar = std::same_as<T, float> || std::same_as<T, double>;

       template<typename T>
       concept ComplexScalar = /* std::complex<RealScalar> */;

       template<typename T>
       using Complex_t = typename Complex<T>::type;  // float -> std::complex<float>

       template<typename T>
       using Real_t = typename Real<T>::type;  // std::complex<float> -> float
   }

Padding helpers
---------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       // Complex extent for R2C: N/2 + 1
       [[nodiscard]] constexpr std::size_t r2cComplexExtent(std::size_t realExtent);

       // Padded real extent for in-place R2C: 2 * (N/2 + 1)
       [[nodiscard]] constexpr std::size_t r2cPaddedRealExtent(std::size_t realExtent);

       // Logical complex extents for R2C transform
       template<std::size_t T_dim>
       [[nodiscard]] constexpr T_Extents r2cComplexExtent(T_Extents realExtents);

       // Physical real storage extents for in-place R2C
       template<std::size_t T_dim>
       [[nodiscard]] constexpr T_Extents r2cPaddedRealExtent(T_Extents realExtents);

       // Product of all extents
       template<std::size_t T_dim>
       [[nodiscard]] constexpr std::size_t product(T_Extents const& extents);

       // Contiguous strides for given extents
       template<std::size_t T_dim>
       [[nodiscard]] constexpr T_Extents contiguousStrides(T_Extents const& extents);
   }

In-place real storage
---------------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       template<typename T_Real, std::size_t T_dim>
       struct InPlaceRealStorage
       {
           T_Extents logicalRealExtents{};
           T_Extents physicalRealExtents{};
           T_Extents logicalComplexExtents{};
           std::size_t logicalRealElements = 0u;
           std::size_t physicalRealElements = 0u;
           std::size_t logicalComplexElements = 0u;
       };

       template<typename T_Real, std::size_t T_dim>
       [[nodiscard]] constexpr auto makeInPlaceRealStorage(T_Extents logicalRealExtents);
   }

Plan
----

.. code-block:: cpp

   namespace alpaka::fft::onHost
   {
       template<typename T_Api, typename T_Value, alpaka::concepts::Vector T_Extents>
       class Plan
       {
       public:
           [[nodiscard]] auto transform() const noexcept;
           [[nodiscard]] auto const& layout() const noexcept;
           [[nodiscard]] auto const& options() const noexcept;
           [[nodiscard]] auto workspaceBytes() const noexcept;

           // Bind user-managed workspace (IMdSpan-compatible: view, buffer, mdspan)
           void setWorkspace(alpaka::concepts::IMdSpan auto& span);

           template<typename T_In, typename T_Out>
           void execute(auto& queue, T_In const& in, T_Out& out, Direction direction);

           void keepAlive(auto& queue) const;
       };
   }

``setWorkspace()`` accepts any alpaka ``IMdSpan``-compatible object (``View``, ``SharedBuffer``,
``SharedBufferFFT``, etc.).  The required byte size is obtained via ``workspaceBytes()``.

PlanBuilder
-----------

.. code-block:: cpp

   namespace alpaka::fft::onHost
   {
       template<typename T_Value, alpaka::concepts::Vector T_Extents>
       class PlanBuilder
       {
       public:
           PlanBuilder& c2c();
           PlanBuilder& r2c();
           PlanBuilder& c2r();
           PlanBuilder& extents(alpaka::concepts::VectorOrScalar auto const& value);
           PlanBuilder& batch(alpaka::GetValueType_t<T_Extents> value);

           /** Set byte-strides per dimension (alpaka pitch convention).
           *
           * Each value is the number of bytes to advance to the next element along
           * the corresponding dimension. The last dimension is fast-moving.
           * Backends convert to element strides automatically. */
           PlanBuilder& strides(
               alpaka::concepts::VectorOrScalar auto const& in,
               alpaka::concepts::VectorOrScalar auto const& out);

           /** Set byte-distance between consecutive batches.
           *
           * The number of bytes between the start of batch i and batch i+1.
           * A value of 0 defaults to product(extents) * sizeof(value_type).
           * Backends convert to element distance automatically. */
           PlanBuilder& distances(std::size_t inDistanceBytes, std::size_t outDistanceBytes);

           PlanBuilder& inPlace();
           PlanBuilder& outOfPlace();
           PlanBuilder& backendManagedWorkspace();
           PlanBuilder& userProvidedWorkspace();

           [[nodiscard]] auto build(auto& device) const;
       };
   }

Execution helpers
-----------------

.. code-block:: cpp

   namespace alpaka::fft::onHost
   {
       // Execute forward/out-of-place transform
       template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
       void executeForward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out);

       // Execute backward/out-of-place transform
       template<typename T_Plan, typename T_Queue, typename T_In, typename T_Out>
       void executeBackward(T_Queue& queue, T_Plan& plan, T_In const& in, T_Out& out);

       // Execute in-place R2C transform
       template<typename T_Api, RealScalar T_Real, ...>
       auto executeR2CInPlace(T_Queue& queue, T_Plan& plan, SharedBufferFFT<T_Api, T_Real, ...>& buffer);

       // Execute in-place C2R transform
       template<typename T_Api, ComplexScalar T_Complex, ...>
       auto executeC2RInPlace(
           T_Queue& queue, T_Plan& plan,
           SharedBufferFFT<T_Api, T_Complex, ...>& buffer);
   }

Allocation helpers
------------------

.. code-block:: cpp

   namespace alpaka::fft::onHost
   {
       // Allocate FFT-managed buffer with automatic real/complex reinterpretation support
       template<typename T_Type>
       [[nodiscard]] auto allocForFFT(auto const& device, auto const& extents);

       // Allocate unified memory buffer for FFT
       template<typename T_Type>
       [[nodiscard]] auto allocUnifiedForFFT(auto const& device, auto const& extents);

       // Allocate mapped memory buffer for FFT
       template<typename T_Type>
       [[nodiscard]] auto allocMappedForFFT(auto const& device, auto const& extents);

       // Allocate deferred buffer for FFT
       template<typename T_Type>
       [[nodiscard]] auto allocDeferredForFFT(auto const& queue, auto const& extents);
   }

SharedBufferFFT
---------------

.. code-block:: cpp

   namespace alpaka::fft
   {
       template<alpaka::concepts::Api T_Api, typename T_Type, ...>
       struct SharedBufferFFT : alpaka::View<T_Api, T_Type, T_Extents, T_MemAlignment>
       {
           // Get byte capacity of the buffer
           [[nodiscard]] auto byteCapacity() const noexcept;

           // Get reference count
           [[nodiscard]] auto getUseCount() const noexcept;

           // Check if buffer is valid
           [[nodiscard]] explicit operator bool() const noexcept;

           // Reinterpret buffer as different type using explicit extents
           template<typename T_Other>
           [[nodiscard]] auto reinterpretBuffer(auto const& extents) const;

           // FFT-aware convenience views
           [[nodiscard]] auto asReal() const;
           [[nodiscard]] auto asComplex() const;
       };
   }
