#pragma once

#include "metal/DeviceCapabilities.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::metal {

enum class AllocationFailure : uint8_t {
  None,
  Capacity, // Unclassified capacity refusal from an alternate backing.
  EngineBudget,
  HostPressure,
  DriverRejected,
};

struct AllocationResult final {
  AllocationFailure failure;
  AllocationResult(bool granted)
      : failure(granted ? AllocationFailure::None
                        : AllocationFailure::Capacity) {}
  AllocationResult(AllocationFailure reason) : failure(reason) {}
  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == AllocationFailure::None;
  }
};

[[nodiscard]] constexpr const char *allocationFailureName(
    AllocationFailure failure) noexcept {
  switch (failure) {
  case AllocationFailure::None: return "none";
  case AllocationFailure::Capacity: return "allocation capacity unavailable";
  case AllocationFailure::EngineBudget: return "engine memory budget exceeded";
  case AllocationFailure::HostPressure: return "host memory reserve protected";
  case AllocationFailure::DriverRejected:
    return "Metal driver rejected allocation";
  }
  return "unknown allocation failure";
}

// Physical allocators use this callback to obtain engine-governed headroom
// without depending on the engine policy type. The operation runs while the
// caller's reservation is held and returns false without side effects when
// admission is denied.
using AllocationAdmission =
    std::function<AllocationResult(uint64_t, const std::function<void()> &)>;

enum class BufferStorage {
  Shared,
  Private,
};

class MetalBackend;
class CommandTicket;

// A cheap, copyable reference to a backend-owned Metal allocation. Views keep
// the base allocation alive and do not increase the tracked allocation count.
class MetalBuffer final {
public:
  MetalBuffer();
  ~MetalBuffer();
  MetalBuffer(const MetalBuffer &);
  MetalBuffer &operator=(const MetalBuffer &);
  MetalBuffer(MetalBuffer &&) noexcept;
  MetalBuffer &operator=(MetalBuffer &&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] uint64_t sizeBytes() const noexcept;
  [[nodiscard]] BufferStorage storage() const noexcept;
  // Returns nullptr for private buffers. The pointer covers this view only.
  [[nodiscard]] void *contents() const noexcept;
  // GPU address of the view's first byte, for kernels that reach a buffer
  // through an address another buffer holds.
  [[nodiscard]] uint64_t gpuAddress() const noexcept;
  // Allocation identity and exact view range, including Private storage.
  // This compares metadata only; it never maps or reads device contents.
  [[nodiscard]] bool sameView(const MetalBuffer &other) const noexcept;

private:
  struct Impl;
  explicit MetalBuffer(std::shared_ptr<Impl> impl);

  std::shared_ptr<Impl> impl_;

  friend class MetalBackend;
};

struct DispatchSize {
  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

struct BufferBinding {
  uint32_t index = 0;
  MetalBuffer buffer;
};

// The pointed-to data only needs to remain valid until submit() returns.
struct BytesBinding {
  uint32_t index = 0;
  const void *data = nullptr;
  uint64_t sizeBytes = 0;
};

struct ComputeDispatch {
  std::string pipelineName;
  std::vector<BufferBinding> buffers;
  std::vector<BytesBinding> bytes;
  DispatchSize threadgroups;
  DispatchSize threadsPerThreadgroup;
};

struct CommandTiming {
  double gpuSeconds = 0.0;
  double wallSeconds = 0.0;
};

// GPU time of one dispatch replayed as its own command while profiling.
struct DispatchTiming {
  std::string pipelineName;
  double gpuSeconds = 0.0;
};

// Move-only ownership of one submitted Metal command. Completion is signalled
// without blocking the submitting thread; wait() is normally called only
// after the host event loop receives the completion notification.
// Destroying or replacing an unfinished ticket waits for GPU completion and
// retains its allocations throughout that wait.
class CommandTicket final {
public:
  CommandTicket();
  ~CommandTicket();
  CommandTicket(const CommandTicket &) = delete;
  CommandTicket &operator=(const CommandTicket &) = delete;
  CommandTicket(CommandTicket &&) noexcept;
  CommandTicket &operator=(CommandTicket &&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] uint64_t sequence() const noexcept;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] CommandTiming wait();

private:
  struct State;
  explicit CommandTicket(std::shared_ptr<State> state);

  std::shared_ptr<State> state_;

  friend class MetalBackend;
};

using CommandCompletion = std::function<void(uint64_t sequence)>;

// Bytes one allocation added between two memoryStats() readings.
[[nodiscard]] inline uint64_t allocationDelta(uint64_t before, uint64_t after) {
  if (after < before)
    throw std::logic_error("Metal allocation accounting moved backwards");
  return after - before;
}

struct MetalMemoryStats {
  // Sum of MTLResource.allocatedSize for live base buffers created through
  // this backend. Views share their base allocation and add no bytes.
  uint64_t allocatedBytes = 0;
  uint64_t peakAllocatedBytes = 0;

  // Most recently sampled Metal device-wide process counter. Allocation and
  // command lifecycle boundaries refresh it; status reads never synchronize
  // with an in-flight GPU command.
  uint64_t deviceCurrentAllocatedBytes = 0;
  // Highest sampled device.currentAllocatedSize. Sampled after allocations
  // and pipeline creation, before submission, and on host-side retirement.
  uint64_t devicePeakAllocatedBytes = 0;
};

class MetalBackendError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// A normal capacity failure. Callers may evict cache or return a retryable
// admission error; the Metal backend remains healthy.
class MetalAllocationError final : public MetalBackendError {
public:
  explicit MetalAllocationError(
      std::string message,
      AllocationFailure failure = AllocationFailure::DriverRejected)
      : MetalBackendError(std::move(message)), failure_(failure) {}
  [[nodiscard]] AllocationFailure failure() const noexcept { return failure_; }
private:
  AllocationFailure failure_;
};

// The capabilities a backend reads, without loading kernels or allocating:
// enough to refuse an unsupported Mac before a model is downloaded.
[[nodiscard]] DeviceCapabilities probeDeviceCapabilities();

// Permits exactly one submitted-but-not-applied command on its command queue.
class MetalBackend final {
public:
  // Every buffer stays wired until residencyKeepAliveSeconds pass without a
  // command (see allocateBuffer).
  explicit MetalBackend(std::string metallibPath,
                        double commandTimeoutSeconds = 120.0,
                        double residencyKeepAliveSeconds = 600.0);
  ~MetalBackend();
  // Invoked before allocations and submissions; may throw to stop bootstrap.
  void setOperationGuard(std::function<void()> guard);
  void checkOperation() const;
  // Stop new submissions before teardown. Commands already committed to the
  // GPU retain their normal lifetime.
  void stop() noexcept;

  MetalBackend(const MetalBackend &) = delete;
  MetalBackend &operator=(const MetalBackend &) = delete;
  MetalBackend(MetalBackend &&) noexcept;
  MetalBackend &operator=(MetalBackend &&) noexcept;

  [[nodiscard]] const DeviceCapabilities &capabilities() const noexcept;

  // Every buffer the backend allocates or wraps belongs to one residency set,
  // attached to the command queue, until its last view is gone. Metal by
  // itself wires a buffer only while a command uses it and a few seconds
  // after, so memory pressure could compress idle state or drop idle weights
  // and the next request would wait to get them back. A member is wired from
  // its allocation on until the keep-alive passes without a command, and
  // again from the next command: memory goes back to macOS when the engine
  // releases it, not when macOS chooses.
  [[nodiscard]] MetalBuffer
  allocateBuffer(uint64_t bytes, BufferStorage storage = BufferStorage::Shared,
                 std::string_view label = {});
  // A private, hazard-untracked buffer that kernels reach only through GPU
  // addresses held in other buffers, as they reach KV pages: the residency
  // set makes it resident for every command, so nothing names it per command
  // or dispatch. Fails with MetalAllocationError unless Metal allocates
  // exactly `bytes`, the amount admission charged.
  [[nodiscard]] MetalBuffer allocateAddressed(uint64_t bytes,
                                              std::string_view label = {});

  // Wraps page-aligned shared memory without copying it. The lifetime token
  // is retained by Metal's deallocator, including any internal buffer owners
  // that outlive our C++ views and completed tickets.
  [[nodiscard]] MetalBuffer wrapSharedMemory(void *address, uint64_t bytes,
                                             std::shared_ptr<void> lifetime,
                                             std::string_view label = {});
  [[nodiscard]] MetalBuffer view(const MetalBuffer &base, uint64_t offsetBytes,
                                 uint64_t lengthBytes) const;

  // The bytes whose residency the keep-alive has ended, until the next
  // command holds them again; Metal unwires them shortly after the end.
  [[nodiscard]] uint64_t lapsedResidentBytes() const noexcept;

  // Encodes exactly one compute dispatch, commits it, waits for completion,
  // and reports both GPU and end-to-end wall time.
  [[nodiscard]] CommandTiming submit(const ComputeDispatch &dispatch);

  // Encodes an ordered dispatch list into one command buffer and waits for it.
  [[nodiscard]] CommandTiming
  submitCommand(std::span<const ComputeDispatch> dispatches);

  // Encodes and commits without waiting. The completion callback only
  // notifies host control flow; command results and errors are consumed from
  // the returned ticket. A second command is rejected until wait() consumes
  // the first ticket, preserving the one-in-flight runtime invariant.
  [[nodiscard]] CommandTicket
  submitAsync(const ComputeDispatch &dispatch,
              CommandCompletion completion = {});
  [[nodiscard]] CommandTicket
  submitCommandAsync(std::span<const ComputeDispatch> dispatches,
                     CommandCompletion completion = {});

  // Development profiling replays a multi-dispatch command synchronously,
  // one dispatch per command buffer. Even submitCommandAsync() then blocks,
  // invokes completion inline and returns an already-completed ticket.
  // Production serving leaves this disabled. Benchmarks read and clear the
  // per-dispatch timings with takeDispatchProfile().
  void setDispatchProfiling(bool enabled) noexcept;
  [[nodiscard]] std::vector<DispatchTiming> takeDispatchProfile();

  [[nodiscard]] MetalMemoryStats memoryStats() const noexcept;
  // Explicit safe-point refresh for memory admission/reclamation code. A
  // control-plane status query must use memoryStats() so it can never wait
  // behind an active Metal command.
  [[nodiscard]] MetalMemoryStats refreshMemoryStats() const noexcept;
  [[nodiscard]] uint64_t submissionCount() const noexcept;
  // True from a submission until its ticket has been consumed: while memory
  // the command reaches through addresses must stay allocated.
  [[nodiscard]] bool commandInFlight() const noexcept;
  [[nodiscard]] size_t pipelineCount() const noexcept;
  [[nodiscard]] bool healthy() const noexcept;
  // Serving-loop check of the command in flight. Terminal results may invoke
  // completion here if the driver callback is delayed.
  // Timeout marks the backend unhealthy without releasing in-flight resources.
  void checkHealth();
  [[nodiscard]] bool needsHealthCheck() const noexcept;
  [[nodiscard]] std::string unhealthyReason() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::metal
