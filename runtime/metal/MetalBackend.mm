#import "MetalBackend.hpp"
#include "CommandWatchdog.hpp"
#include "Residency.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <IOKit/IOKitLib.h>
#include <dispatch/dispatch.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <unistd.h>

namespace splash::metal {
namespace {

// The accelerator entry that backs a Metal device publishes gpu-core-count.
// The device's registry ID names that entry or a child of it; the first
// IOAccelerator service is the fallback, since Apple silicon Macs have one
// GPU. Zero means the property was not found anywhere.
uint32_t gpuCoreCountForDevice(uint64_t registryId) noexcept {
    uint32_t count = 0;
    const auto read = [&](io_registry_entry_t entry) {
        if (!entry) return false;
        CFTypeRef value = IORegistryEntryCreateCFProperty(
            entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value) {
            int64_t number = 0;
            if (CFGetTypeID(value) == CFNumberGetTypeID() &&
                CFNumberGetValue(static_cast<CFNumberRef>(value),
                                 kCFNumberSInt64Type, &number) &&
                number > 0 && number <= 4096) {
                count = static_cast<uint32_t>(number);
            }
            CFRelease(value);
        }
        return count != 0;
    };
    io_registry_entry_t entry = IOServiceGetMatchingService(
        kIOMainPortDefault, IORegistryEntryIDMatching(registryId));
    for (int depth = 0; entry && depth < 4 && !read(entry); ++depth) {
        io_registry_entry_t parent = MACH_PORT_NULL;
        if (IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent) !=
            KERN_SUCCESS) {
            parent = MACH_PORT_NULL;
        }
        IOObjectRelease(entry);
        entry = parent;
    }
    if (entry) IOObjectRelease(entry);
    if (!count) {
        io_registry_entry_t accelerator = IOServiceGetMatchingService(
            kIOMainPortDefault, IOServiceMatching("IOAccelerator"));
        if (accelerator) {
            read(accelerator);
            IOObjectRelease(accelerator);
        }
    }
    return count;
}

std::string stringFromNSString(NSString *value) {
    if (!value) return {};
    const char *utf8 = value.UTF8String;
    return utf8 ? utf8 : "";
}

std::string errorDescription(NSError *error) {
    if (!error) return "unknown Metal error";
    std::string result = stringFromNSString(error.localizedDescription);
    return result.empty() ? "unknown Metal error" : result;
}

void readMacosVersion(DeviceCapabilities &capabilities) {
    const NSOperatingSystemVersion os =
        NSProcessInfo.processInfo.operatingSystemVersion;
    const auto component = [](NSInteger value) {
        return value > 0 ? static_cast<uint32_t>(value) : 0U;
    };
    capabilities.macosMajor = component(os.majorVersion);
    capabilities.macosMinor = component(os.minorVersion);
    capabilities.macosPatch = component(os.patchVersion);
}

// The backend and probeDeviceCapabilities() share one reading of the device,
// so the probe judges a Mac by the values the engine validates.
void readDeviceCapabilities(id<MTLDevice> device,
                            DeviceCapabilities &capabilities) {
    capabilities.deviceName = stringFromNSString(device.name);
    capabilities.gpuCoreCount = gpuCoreCountForDevice(device.registryID);
    // Apple GPU families nest, so the device's is the last one supported
    // counting up from Apple7.
    uint32_t family = 0;
    for (uint32_t next = 7;
         [device supportsFamily:static_cast<MTLGPUFamily>(1000 + next)]; ++next)
        family = next;
    capabilities.appleGpuFamily = family;
    capabilities.physicalMemoryBytes = NSProcessInfo.processInfo.physicalMemory;
    capabilities.recommendedMaxWorkingSetBytes =
        device.recommendedMaxWorkingSetSize;
    capabilities.maxBufferLengthBytes = device.maxBufferLength;
    capabilities.maxThreadgroupMemoryBytes = device.maxThreadgroupMemoryLength;
    MTLSize maximumThreads = device.maxThreadsPerThreadgroup;
    capabilities.maxThreadgroupWidth = maximumThreads.width;
    capabilities.hasUnifiedMemory = device.hasUnifiedMemory;
}

NSUInteger checkedNSUInteger(uint64_t value, std::string_view field) {
    if (value > std::numeric_limits<NSUInteger>::max()) {
        throw MetalBackendError(std::string(field) + " exceeds NSUInteger");
    }
    return static_cast<NSUInteger>(value);
}

MTLSize metalSize(const DispatchSize &size, std::string_view field) {
    if (!size.x || !size.y || !size.z) {
        throw MetalBackendError(std::string(field) + " must be non-zero");
    }
    return MTLSizeMake(checkedNSUInteger(size.x, field),
                       checkedNSUInteger(size.y, field),
                       checkedNSUInteger(size.z, field));
}

bool multiplyOverflows(uint64_t left, uint64_t right) {
    return right && left > std::numeric_limits<uint64_t>::max() / right;
}

// Entries of a kernel's buffer argument table on every Apple GPU family.
constexpr uint32_t kBufferArgumentEntries = 31;

double steadySeconds() noexcept {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char *commandStatusName(MTLCommandBufferStatus status) noexcept {
    switch (status) {
    case MTLCommandBufferStatusNotEnqueued: return "not_enqueued";
    case MTLCommandBufferStatusEnqueued: return "enqueued";
    case MTLCommandBufferStatusCommitted: return "committed";
    case MTLCommandBufferStatusScheduled: return "scheduled";
    case MTLCommandBufferStatusCompleted: return "completed";
    case MTLCommandBufferStatusError: return "error";
    }
    return "unknown";
}

template <typename T>
void raisePeak(std::atomic<T> &peak, T value) noexcept {
    T current = peak.load(std::memory_order_relaxed);
    while (value > current &&
           !peak.compare_exchange_weak(current, value,
                                       std::memory_order_relaxed)) {}
}

// Hashes pipeline names as views, so a cache lookup builds no string.
struct PipelineNameHash {
    using is_transparent = void;
    size_t operator()(std::string_view name) const noexcept {
        return std::hash<std::string_view>{}(name);
    }
};

NSString *checkedNSString(std::string_view value, std::string_view field) {
    NSString *result = [[NSString alloc]
        initWithBytes:value.data()
        length:value.size()
        encoding:NSUTF8StringEncoding];
    if (!result) {
        throw MetalBackendError(std::string(field) + " is not UTF-8");
    }
    return result;
}

}  // namespace

struct AllocationAccounting {
    std::atomic<uint64_t> allocatedBytes{0};
    std::atomic<uint64_t> peakAllocatedBytes{0};
};

struct MetalAllocation {
    // Own the host mapping for our views as well as the Metal deallocator.
    // Validation wrappers may not retain the supplied deallocator block.
    std::shared_ptr<void> externalOwner;
    __strong id<MTLBuffer> buffer = nil;
    std::shared_ptr<AllocationAccounting> accounting;
    uint64_t bytes = 0;
    BufferStorage storage = BufferStorage::Shared;
    // The residency set the buffer belongs to, held weakly as allocations
    // may outlive the backend. The set retains the buffer, and with it its
    // memory, so the last view takes it out.
    std::weak_ptr<Residency> residency;

    ~MetalAllocation() {
        if (auto kept = residency.lock()) kept->remove(buffer);
        if (accounting && bytes) {
            accounting->allocatedBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
        }
    }
};

struct MetalBuffer::Impl {
    std::shared_ptr<MetalAllocation> allocation;
    uint64_t offsetBytes = 0;
    uint64_t lengthBytes = 0;
};

struct BackendAsyncState {
    __strong id<MTLDevice> device = nil;
    mutable std::atomic<uint64_t> deviceCurrentAllocatedBytes{0};
    mutable std::atomic<uint64_t> devicePeakAllocatedBytes{0};
    std::atomic<bool> healthy{true};
    mutable std::mutex healthMutex;
    std::string healthReason;
    mutable std::mutex gateMutex;
    uint64_t nextSequence = 0;
    uint64_t activeSequence = 0;
    size_t activeDispatchCount = 0;
    __weak id<MTLCommandBuffer> activeCommand = nil;
    std::function<void(id<MTLCommandBuffer>)> activeCompletion;
    CommandWatchdog commandWatchdog;
    bool stopping = false;

    uint64_t sampleDeviceMemory() const noexcept {
        if (!device) return 0;
        uint64_t current = static_cast<uint64_t>(device.currentAllocatedSize);
        deviceCurrentAllocatedBytes.store(current, std::memory_order_relaxed);
        raisePeak(devicePeakAllocatedBytes, current);
        return current;
    }

    void ensureHealthy() const {
        if (healthy.load(std::memory_order_acquire)) return;
        std::lock_guard lock(healthMutex);
        throw MetalBackendError("Metal backend is unhealthy: " + healthReason);
    }

    void markUnhealthy(std::string reason) {
        {
            std::lock_guard lock(healthMutex);
            if (healthReason.empty()) healthReason = std::move(reason);
        }
        healthy.store(false, std::memory_order_release);
    }

    uint64_t beginSubmission(size_t dispatchCount) {
        ensureHealthy();
        std::lock_guard lock(gateMutex);
        if (stopping)
            throw MetalBackendError("Metal backend is stopping");
        if (activeSequence) {
            throw MetalBackendError(
                "Metal backend already has an in-flight command");
        }
        if (nextSequence == std::numeric_limits<uint64_t>::max()) {
            throw MetalBackendError("Metal command sequence exhausted");
        }
        activeSequence = ++nextSequence;
        activeDispatchCount = dispatchCount;
        return activeSequence;
    }

    bool commitSubmission(uint64_t sequence, id<MTLCommandBuffer> command,
                          std::function<void(id<MTLCommandBuffer>)> completion) {
        std::lock_guard lock(gateMutex);
        if (stopping) return false;
        activeCommand = command;
        activeCompletion = std::move(completion);
        commandWatchdog.start(sequence, steadySeconds());
        [command commit];
        return true;
    }

    void releaseSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
        if (activeSequence == sequence) {
            activeSequence = 0;
            activeCommand = nil;
            activeCompletion = {};
        }
    }

    void completeSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
    }

    void checkCommandHealth() {
        id<MTLCommandBuffer> command = nil;
        std::function<void(id<MTLCommandBuffer>)> complete;
        {
            std::lock_guard lock(gateMutex);
            if (commandWatchdog.expired(steadySeconds())) {
                command = activeCommand;
                const auto status = command ? command.status
                                            : MTLCommandBufferStatusNotEnqueued;
                // Recover terminal results even if the driver has not delivered
                // its callback. Finish outside the gate: it takes the ticket lock.
                if (command && (status == MTLCommandBufferStatusCompleted ||
                                status == MTLCommandBufferStatusError)) {
                    complete = activeCompletion;
                } else {
                    std::ostringstream message;
                    message << "Metal command completion timed out after "
                            << commandWatchdog.timeoutSeconds()
                            << " seconds (sequence=" << activeSequence
                            << ", status=" << (command ? commandStatusName(status)
                                                       : "unavailable")
                            << ", dispatches=" << activeDispatchCount << ')';
                    markUnhealthy(message.str());
                }
            }
        }
        if (complete) complete(command);
        ensureHealthy();
    }

    [[nodiscard]] bool hasActiveSubmission() const noexcept {
        std::lock_guard lock(gateMutex);
        return activeSequence != 0;
    }
};

struct CommandTicket::State {
    std::shared_ptr<BackendAsyncState> backend;
    std::vector<std::shared_ptr<MetalAllocation>> retainedAllocations;
    CommandCompletion completion;
    mutable std::mutex mutex;
    std::condition_variable condition;
    uint64_t sequence = 0;
    CommandTiming timing;
    std::chrono::steady_clock::time_point wallStart;
    std::string error;
    bool completed = false;
    bool released = false;

    void finishCommand(id<MTLCommandBuffer> command) {
        auto wallEnd = std::chrono::steady_clock::now();
        CommandTiming timing;
        timing.gpuSeconds =
            command.GPUEndTime - command.GPUStartTime;
        if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds < 0.0) {
            timing.gpuSeconds = 0.0;
        }
        timing.wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();

        std::string error;
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::ostringstream message;
            message << "Metal command " << sequence << " failed";
            if (command.error) {
                message << ": " << errorDescription(command.error);
            }
            error = message.str();
        }

        finish(timing, std::move(error));
    }

    void finish(CommandTiming result, std::string failure = {}) {
        CommandCompletion notify;
        {
            std::lock_guard lock(mutex);
            // Host recovery, late callbacks, and discarded commands all share
            // this completion path; only the first result may publish or notify.
            if (completed) return;
            backend->completeSubmission(sequence);
            if (!failure.empty()) backend->markUnhealthy(failure);
            timing = result;
            error = std::move(failure);
            completed = true;
            notify = completion;
        }
        if (notify) {
            try {
                notify(sequence);
            } catch (...) {
                backend->markUnhealthy(
                    "Metal completion callback threw an exception");
            }
        }
        condition.notify_all();
    }

    void release() noexcept {
        bool shouldRelease = false;
        {
            std::lock_guard lock(mutex);
            if (!released) {
                released = true;
                retainedAllocations.clear();
                shouldRelease = true;
            }
        }
        if (shouldRelease && backend) {
            // Refresh admission telemetry on the consuming thread after GPU
            // completion, before allowing the next submission.
            if (backend->healthy.load(std::memory_order_acquire))
                backend->sampleDeviceMemory();
            backend->releaseSubmission(sequence);
        }
    }

    void abandon() noexcept {
        {
            std::unique_lock lock(mutex);
            condition.wait(lock, [this] { return completed; });
        }
        release();
    }
};

struct MetalBackend::Impl {
    std::function<void()> operationGuard;

    bool dispatchProfiling = false;
    std::vector<DispatchTiming> dispatchProfile;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    // Allocations hold it weakly: they may outlive the backend.
    std::shared_ptr<Residency> residency;
    __strong id<MTLLibrary> library = nil;
    // Looked up for every dispatch on the encode path, which the GPU waits
    // for; a hit allocates nothing.
    std::unordered_map<std::string, id<MTLComputePipelineState>,
                       PipelineNameHash, std::equal_to<>>
        pipelines;

    DeviceCapabilities capabilities;
    std::shared_ptr<AllocationAccounting> accounting =
        std::make_shared<AllocationAccounting>();
    std::shared_ptr<BackendAsyncState> asyncState =
        std::make_shared<BackendAsyncState>();
    mutable std::mutex commandMutex;

    uint64_t sampleDeviceMemory() const noexcept {
        return asyncState->sampleDeviceMemory();
    }

    void ensureHealthy() const {
        asyncState->ensureHealthy();
    }

    void markUnhealthy(std::string reason) {
        asyncState->markUnhealthy(std::move(reason));
    }

    MetalBuffer wrap(std::shared_ptr<MetalAllocation> allocation) {
        auto result = std::make_shared<MetalBuffer::Impl>();
        result->lengthBytes = allocation->buffer.length;
        result->allocation = std::move(allocation);
        return MetalBuffer(std::move(result));
    }

    MetalBuffer registerBuffer(id<MTLBuffer> buffer, BufferStorage storage,
                               std::shared_ptr<void> externalOwner = {}) {
        auto allocation = std::make_shared<MetalAllocation>();
        allocation->externalOwner = std::move(externalOwner);
        allocation->buffer = buffer;
        allocation->accounting = accounting;
        allocation->bytes = buffer.allocatedSize;
        allocation->storage = storage;
        residency->add(buffer);
        allocation->residency = residency;
        raisePeak(accounting->peakAllocatedBytes,
                  accounting->allocatedBytes.fetch_add(
                      allocation->bytes, std::memory_order_relaxed) +
                      allocation->bytes);
        sampleDeviceMemory();
        return wrap(std::move(allocation));
    }

    MetalAllocation &allocationOf(const MetalBuffer &buffer) const {
        if (!buffer.impl_ || !buffer.impl_->allocation ||
            buffer.impl_->allocation->accounting != accounting) {
            throw MetalBackendError(
                "Metal buffer is empty or belongs to another backend");
        }
        return *buffer.impl_->allocation;
    }

    id<MTLComputePipelineState> pipeline(std::string_view name) {
        if (name.empty()) {
            throw MetalBackendError("Metal pipeline name must not be empty");
        }
        if (const auto cached = pipelines.find(name); cached != pipelines.end())
            return cached->second;
        id<MTLComputePipelineState> result = newPipeline(name);
        pipelines.emplace(name, result);
        sampleDeviceMemory();
        return result;
    }

    id<MTLComputePipelineState> newPipeline(std::string_view name) {
        NSString *key = checkedNSString(name, "pipeline name");
        id<MTLFunction> function = [library newFunctionWithName:key];
        if (!function) {
            throw MetalBackendError(
                "missing Metal function: " + std::string(name));
        }
        NSError *error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) {
            throw MetalBackendError(
                "unable to create Metal pipeline " + std::string(name) +
                ": " + errorDescription(error));
        }
        return result;
    }
};

MetalBuffer::MetalBuffer() = default;
MetalBuffer::~MetalBuffer() = default;
MetalBuffer::MetalBuffer(const MetalBuffer &) = default;
MetalBuffer &MetalBuffer::operator=(const MetalBuffer &) = default;
MetalBuffer::MetalBuffer(MetalBuffer &&) noexcept = default;
MetalBuffer &MetalBuffer::operator=(MetalBuffer &&) noexcept = default;

MetalBuffer::MetalBuffer(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalBuffer::operator bool() const noexcept {
    return impl_ && impl_->allocation && impl_->allocation->buffer;
}

uint64_t MetalBuffer::sizeBytes() const noexcept {
    return impl_ ? impl_->lengthBytes : 0;
}

bool MetalBuffer::sameView(const MetalBuffer &other) const noexcept {
    if (impl_ == other.impl_) return true;
    return impl_ && other.impl_ &&
           impl_->allocation == other.impl_->allocation &&
           impl_->offsetBytes == other.impl_->offsetBytes &&
           impl_->lengthBytes == other.impl_->lengthBytes;
}

BufferStorage MetalBuffer::storage() const noexcept {
    return impl_ && impl_->allocation ? impl_->allocation->storage
                                      : BufferStorage::Shared;
}

void *MetalBuffer::contents() const noexcept {
    if (!impl_ || !impl_->allocation ||
        impl_->allocation->storage != BufferStorage::Shared) {
        return nullptr;
    }
    void *base = impl_->allocation->buffer.contents;
    if (!base) return nullptr;
    return static_cast<uint8_t *>(base) + impl_->offsetBytes;
}

uint64_t MetalBuffer::gpuAddress() const noexcept {
    if (!impl_ || !impl_->allocation || !impl_->allocation->buffer) return 0;
    return impl_->allocation->buffer.gpuAddress + impl_->offsetBytes;
}

CommandTicket::CommandTicket() = default;

CommandTicket::CommandTicket(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

CommandTicket::~CommandTicket() {
    if (state_) state_->abandon();
}

CommandTicket::CommandTicket(CommandTicket &&) noexcept = default;

CommandTicket &CommandTicket::operator=(CommandTicket &&other) noexcept {
    if (this == &other) return *this;
    if (state_) state_->abandon();
    state_ = std::move(other.state_);
    return *this;
}

CommandTicket::operator bool() const noexcept {
    return static_cast<bool>(state_);
}

uint64_t CommandTicket::sequence() const noexcept {
    return state_ ? state_->sequence : 0;
}

bool CommandTicket::ready() const noexcept {
    if (!state_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->completed;
}

CommandTiming CommandTicket::wait() {
    if (!state_) throw MetalBackendError("Metal command ticket is empty");
    CommandTiming timing;
    std::string error;
    {
        std::unique_lock lock(state_->mutex);
        state_->condition.wait(lock, [this] { return state_->completed; });
        timing = state_->timing;
        error = state_->error;
    }
    state_->release();
    if (!error.empty()) throw MetalBackendError(error);
    return timing;
}

MetalBackend::MetalBackend(std::string metallibPath, double commandTimeoutSeconds,
                           double residencyKeepAliveSeconds)
    : impl_(std::make_unique<Impl>()) {
    impl_->asyncState->commandWatchdog = CommandWatchdog(commandTimeoutSeconds);
    if (!std::isfinite(residencyKeepAliveSeconds) ||
        residencyKeepAliveSeconds <= 0.0) {
        throw MetalBackendError(
            "residency keep-alive must be finite and positive");
    }
    @autoreleasepool {
        if (metallibPath.empty()) {
            throw MetalBackendError("metallib path must not be empty");
        }
        // Check the OS floor before loading Metal resources so an unsupported
        // system reports the version requirement first.
        readMacosVersion(impl_->capabilities);
        if (!impl_->capabilities.meetsMinimumMacos()) {
            throw MetalBackendError(
                "Splash requires macOS " +
                std::to_string(DeviceCapabilities::kMinimumMacosMajor) + '.' +
                std::to_string(DeviceCapabilities::kMinimumMacosMinor) +
                " or newer; this Mac runs macOS " +
                impl_->capabilities.macosVersion());
        }
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw MetalBackendError("Metal device unavailable");
        }
        impl_->asyncState->device = impl_->device;
        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) {
            throw MetalBackendError("unable to create Metal command queue");
        }

        NSString *path = checkedNSString(metallibPath, "metallib path");
        NSError *error = nil;
        NSData *fileData = [NSData dataWithContentsOfFile:path
                                                 options:0
                                                   error:&error];
        if (!fileData) {
            throw MetalBackendError(
                "unable to read metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        // The library keeps the bytes read here, whatever later replaces the
        // path; the dispatch data retains them rather than copying them.
        dispatch_data_t data = dispatch_data_create(
            fileData.bytes, fileData.length, nullptr, ^{ (void)fileData; });
        error = nil;
        impl_->library =
            [impl_->device newLibraryWithData:data error:&error];
        if (!impl_->library) {
            throw MetalBackendError(
                "unable to load metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        // Ending residency dispatches a kernel built here, so no pipeline or
        // driver program is compiled when a keep-alive lapses.
        impl_->residency = std::make_shared<Residency>(
            impl_->device, impl_->queue,
            impl_->newPipeline(Residency::kKickPipeline),
            residencyKeepAliveSeconds);
        impl_->sampleDeviceMemory();

        readDeviceCapabilities(impl_->device, impl_->capabilities);
    }
    impl_->sampleDeviceMemory();
}

MetalBackend::~MetalBackend() { stop(); }

void MetalBackend::stop() noexcept {
    if (!impl_) return;
    std::lock_guard lock(impl_->asyncState->gateMutex);
    impl_->asyncState->stopping = true;
}
MetalBackend::MetalBackend(MetalBackend &&) noexcept = default;
MetalBackend &MetalBackend::operator=(MetalBackend &&other) noexcept {
    if (this != &other) {
        stop();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

const DeviceCapabilities &MetalBackend::capabilities() const noexcept {
    return impl_->capabilities;
}

DeviceCapabilities probeDeviceCapabilities() {
    @autoreleasepool {
        DeviceCapabilities capabilities;
        readMacosVersion(capabilities);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw MetalBackendError("Metal device unavailable");
        readDeviceCapabilities(device, capabilities);
        return capabilities;
    }
}

void MetalBackend::checkOperation() const {
    impl_->ensureHealthy();
    if (impl_->operationGuard) impl_->operationGuard();
}

void MetalBackend::setOperationGuard(std::function<void()> guard) {
    impl_->operationGuard = std::move(guard);
}

MetalBuffer MetalBackend::allocateBuffer(uint64_t bytes,
                                         BufferStorage storage,
                                         std::string_view label) {
    checkOperation();
    if (!bytes) throw MetalBackendError("Metal buffer size must be positive");
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("Metal buffer exceeds maxBufferLength");
    }

    MTLResourceOptions options = storage == BufferStorage::Shared
        ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
    id<MTLBuffer> buffer = [impl_->device
        newBufferWithLength:checkedNSUInteger(bytes, "buffer size")
        options:options];
    if (!buffer) throw MetalAllocationError("Metal buffer allocation failed");
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");
    return impl_->registerBuffer(buffer, storage);
}

MetalBuffer MetalBackend::allocateAddressed(uint64_t bytes,
                                            std::string_view label) {
    checkOperation();
    if (!bytes) throw MetalBackendError("Metal buffer size must be positive");
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("Metal buffer exceeds maxBufferLength");
    }
    id<MTLBuffer> buffer = [impl_->device
        newBufferWithLength:checkedNSUInteger(bytes, "buffer size")
        options:MTLResourceStorageModePrivate |
                MTLResourceHazardTrackingModeUntracked];
    if (!buffer) throw MetalAllocationError("Metal buffer allocation failed");
    if (buffer.allocatedSize != bytes) {
        throw MetalAllocationError(
            "Metal allocated " + std::to_string(buffer.allocatedSize) +
            " bytes for an addressed buffer of " + std::to_string(bytes));
    }
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");
    return impl_->registerBuffer(buffer, BufferStorage::Private);
}

MetalBuffer MetalBackend::wrapSharedMemory(
    void *address, uint64_t bytes, std::shared_ptr<void> lifetime,
    std::string_view label) {
    checkOperation();
    if (!address || !bytes) {
        throw MetalBackendError("shared memory address and size are required");
    }
    if (!lifetime) {
        throw MetalBackendError("shared memory lifetime token is required");
    }
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("shared memory exceeds maxBufferLength");
    }
    long systemPageSize = sysconf(_SC_PAGESIZE);
    if (systemPageSize <= 0) {
        throw MetalBackendError("unable to determine system page size");
    }
    uint64_t pageSize = static_cast<uint64_t>(systemPageSize);
    if (reinterpret_cast<uintptr_t>(address) % pageSize || bytes % pageSize) {
        throw MetalBackendError(
            "shared memory address and size must be page-aligned");
    }

    id<MTLBuffer> buffer = [impl_->device
        newBufferWithBytesNoCopy:address
        length:checkedNSUInteger(bytes, "shared memory size")
        options:MTLResourceStorageModeShared
        deallocator:^(void *, NSUInteger) {
            // Metal may retain the buffer beyond our last C++ view/ticket,
            // including while a completed command's handler is returning.
            // Keep its backing owner until Metal actually releases it.
            (void)lifetime;
        }];
    if (!buffer) {
        throw MetalBackendError("zero-copy Metal buffer creation failed");
    }
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");
    return impl_->registerBuffer(buffer, BufferStorage::Shared,
                                 std::move(lifetime));
}

MetalBuffer MetalBackend::view(const MetalBuffer &base,
                               uint64_t offsetBytes,
                               uint64_t lengthBytes) const {
    impl_->ensureHealthy();
    if (!base.impl_ || !base.impl_->allocation) {
        throw MetalBackendError("cannot view an empty Metal buffer");
    }
    if (base.impl_->allocation->accounting.get() != impl_->accounting.get()) {
        throw MetalBackendError("Metal buffer belongs to another backend");
    }
    if (!lengthBytes || offsetBytes > base.impl_->lengthBytes ||
        lengthBytes > base.impl_->lengthBytes - offsetBytes) {
        std::ostringstream message;
        message << "Metal buffer view is out of range: offset=" << offsetBytes
                << " length=" << lengthBytes
                << " base_length=" << base.impl_->lengthBytes;
        throw MetalBackendError(message.str());
    }
    auto result = std::make_shared<MetalBuffer::Impl>();
    result->allocation = base.impl_->allocation;
    result->offsetBytes = base.impl_->offsetBytes + offsetBytes;
    result->lengthBytes = lengthBytes;
    return MetalBuffer(std::move(result));
}

uint64_t MetalBackend::lapsedResidentBytes() const noexcept {
    return impl_->residency->lapsedBytes();
}

CommandTiming MetalBackend::submit(const ComputeDispatch &dispatch) {
    return submitAsync(dispatch).wait();
}

CommandTiming MetalBackend::submitCommand(
    std::span<const ComputeDispatch> dispatches) {
    return submitCommandAsync(dispatches).wait();
}

CommandTicket MetalBackend::submitAsync(
    const ComputeDispatch &dispatch, CommandCompletion completion) {
    return submitCommandAsync(
        std::span<const ComputeDispatch>(&dispatch, 1),
        std::move(completion));
}

void MetalBackend::setDispatchProfiling(bool enabled) noexcept {
    impl_->dispatchProfiling = enabled;
}

std::vector<DispatchTiming> MetalBackend::takeDispatchProfile() {
    return std::exchange(impl_->dispatchProfile, {});
}

CommandTicket MetalBackend::submitCommandAsync(
    std::span<const ComputeDispatch> dispatches,
    CommandCompletion completion) {
    checkOperation();
    if (dispatches.empty()) {
        throw MetalBackendError("Metal command must contain a dispatch");
    }
    if (impl_->dispatchProfiling && dispatches.size() > 1) {
        // Replay serially, one command per dispatch, then hand back an
        // already-completed ticket carrying the summed timing so callers
        // observe the usual asynchronous contract.
        CommandTiming total;
        for (const ComputeDispatch &dispatch : dispatches) {
            CommandTiming timing = submitAsync(dispatch).wait();
            impl_->dispatchProfile.push_back(
                {dispatch.pipelineName, timing.gpuSeconds});
            total.gpuSeconds += timing.gpuSeconds;
            total.wallSeconds += timing.wallSeconds;
        }
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = impl_->asyncState;
        ticketState->sequence = impl_->asyncState->beginSubmission(dispatches.size());
        ticketState->timing = total;
        ticketState->completed = true;
        if (completion) completion(ticketState->sequence);
        return CommandTicket(std::move(ticketState));
    }
    struct PreparedDispatch {
        const ComputeDispatch *source = nullptr;
        MTLSize groups{};
        MTLSize threads{};
        uint64_t threadCount = 0;
        __strong id<MTLComputePipelineState> pipeline = nil;
    };
    std::vector<PreparedDispatch> prepared;
    prepared.reserve(dispatches.size());
    for (const ComputeDispatch &dispatch : dispatches) {
        PreparedDispatch item;
        item.source = &dispatch;
        item.groups = metalSize(dispatch.threadgroups, "threadgroups");
        item.threads = metalSize(
            dispatch.threadsPerThreadgroup, "threadsPerThreadgroup");
        if (multiplyOverflows(dispatch.threadsPerThreadgroup.x,
                              dispatch.threadsPerThreadgroup.y) ||
            multiplyOverflows(dispatch.threadsPerThreadgroup.x *
                                  dispatch.threadsPerThreadgroup.y,
                              dispatch.threadsPerThreadgroup.z)) {
            throw MetalBackendError("threadsPerThreadgroup size overflows");
        }
        item.threadCount = dispatch.threadsPerThreadgroup.x *
            dispatch.threadsPerThreadgroup.y *
            dispatch.threadsPerThreadgroup.z;

        // Each binding takes its own entry of the argument table.
        uint32_t indices = 0;
        const auto claim = [&](uint32_t index) {
            if (index >= kBufferArgumentEntries) {
                throw MetalBackendError(
                    "compute binding index exceeds the argument table");
            }
            if (indices & (uint32_t{1} << index)) {
                throw MetalBackendError("duplicate compute binding index");
            }
            indices |= uint32_t{1} << index;
        };
        for (const BufferBinding &binding : dispatch.buffers) {
            if (!binding.buffer.impl_ || !binding.buffer.impl_->allocation) {
                std::ostringstream message;
                message << "compute dispatch '" << dispatch.pipelineName
                        << "' contains an empty buffer at index "
                        << binding.index;
                throw MetalBackendError(message.str());
            }
            if (binding.buffer.impl_->allocation->accounting.get() !=
                impl_->accounting.get()) {
                throw MetalBackendError(
                    "compute dispatch buffer belongs to another backend");
            }
            claim(binding.index);
        }
        for (const BytesBinding &binding : dispatch.bytes) {
            if (!binding.data || !binding.sizeBytes) {
                throw MetalBackendError("compute byte binding is empty");
            }
            checkedNSUInteger(binding.sizeBytes, "byte binding size");
            claim(binding.index);
        }
        prepared.push_back(item);
    }

    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    for (PreparedDispatch &item : prepared) {
        item.pipeline = impl_->pipeline(item.source->pipelineName);
        if (item.threadCount >
            item.pipeline.maxTotalThreadsPerThreadgroup) {
            throw MetalBackendError(
                "threadsPerThreadgroup exceeds pipeline capability");
        }
    }

    auto ticketState = std::make_shared<CommandTicket::State>();
    ticketState->backend = impl_->asyncState;
    ticketState->completion = std::move(completion);
    std::unordered_set<const MetalAllocation *> retained;
    for (const ComputeDispatch &dispatch : dispatches) {
        for (const BufferBinding &binding : dispatch.buffers) {
            const auto &allocation = binding.buffer.impl_->allocation;
            if (retained.insert(allocation.get()).second) {
                ticketState->retainedAllocations.push_back(allocation);
            }
        }
    }
    ticketState->sequence = impl_->asyncState->beginSubmission(dispatches.size());

    auto failBeforeCommit = [&](std::string message) {
        impl_->markUnhealthy(message);
        impl_->asyncState->releaseSubmission(ticketState->sequence);
        throw MetalBackendError(std::move(message));
    };

    auto wallStart = std::chrono::steady_clock::now();
    // Metal may autorelease the command and its encoder, and the serving
    // loop's pool never drains, so their temporary ownership ends with this
    // submission (under the validation layer an autoreleased command holds
    // every member of the residency set). The command retains everything the
    // GPU still needs.
    @autoreleasepool {
        id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
        if (!command) {
            failBeforeCommit("unable to create Metal command buffer");
        }
        ticketState->wallStart = wallStart;
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            failBeforeCommit("unable to create Metal compute encoder");
        }
        try {
            for (const PreparedDispatch &item : prepared) {
                const ComputeDispatch &dispatch = *item.source;
                [encoder setComputePipelineState:item.pipeline];
                for (const BufferBinding &binding : dispatch.buffers) {
                    const MetalBuffer::Impl &buffer = *binding.buffer.impl_;
                    [encoder setBuffer:buffer.allocation->buffer
                                offset:checkedNSUInteger(buffer.offsetBytes,
                                                         "buffer offset")
                               atIndex:binding.index];
                }
                for (const BytesBinding &binding : dispatch.bytes) {
                    [encoder setBytes:binding.data
                               length:checkedNSUInteger(binding.sizeBytes,
                                                        "byte binding size")
                              atIndex:binding.index];
                }
                [encoder dispatchThreadgroups:item.groups
                         threadsPerThreadgroup:item.threads];
            }
            [encoder endEncoding];
        } catch (...) {
            impl_->asyncState->releaseSubmission(ticketState->sequence);
            throw;
        }

        // Driver callbacks only complete the ticket. Device-wide memory
        // telemetry is sampled on the host before submission and when
        // consuming the result.
        [command addCompletedHandler:^(id<MTLCommandBuffer> completedCommand) {
            ticketState->finishCommand(completedCommand);
        }];
        impl_->sampleDeviceMemory();
        impl_->residency->use();
        std::shared_ptr<BackendAsyncState> backend = impl_->asyncState;
        if (!backend->healthy.load(std::memory_order_acquire)) {
            ticketState->finish({}, "Metal backend became unhealthy before command submission");
        } else if (!backend->commitSubmission(ticketState->sequence, command,
                       [weakTicket = std::weak_ptr(ticketState)](id<MTLCommandBuffer> completed) {
                           if (auto ticket = weakTicket.lock()) ticket->finishCommand(completed);
                       })) {
            ticketState->finish({}, "Metal backend stopped before command submission");
        }
    }
    return CommandTicket(std::move(ticketState));
}

MetalMemoryStats MetalBackend::memoryStats() const noexcept {
    // Reading MTLDevice.currentAllocatedSize can synchronize with an active
    // command on some Apple GPUs. Every allocation and command lifecycle
    // boundary already samples it, so status must use the cached atomic value
    // rather than turning a control-plane query into a GPU barrier.
    return {
        impl_->accounting->allocatedBytes.load(std::memory_order_relaxed),
        impl_->accounting->peakAllocatedBytes.load(std::memory_order_relaxed),
        impl_->asyncState->deviceCurrentAllocatedBytes.load(
            std::memory_order_relaxed),
        impl_->asyncState->devicePeakAllocatedBytes.load(
            std::memory_order_relaxed),
    };
}

MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
    impl_->sampleDeviceMemory();
    return memoryStats();
}

uint64_t MetalBackend::submissionCount() const noexcept {
    std::lock_guard lock(impl_->asyncState->gateMutex);
    return impl_->asyncState->nextSequence;
}

bool MetalBackend::commandInFlight() const noexcept {
    return impl_->asyncState->hasActiveSubmission();
}

size_t MetalBackend::pipelineCount() const noexcept {
    std::lock_guard lock(impl_->commandMutex);
    return impl_->pipelines.size();
}

void MetalBackend::checkHealth() {
    impl_->asyncState->checkCommandHealth();
}

bool MetalBackend::healthy() const noexcept {
    return impl_->asyncState->healthy.load(std::memory_order_acquire);
}

std::string MetalBackend::unhealthyReason() const {
    std::lock_guard lock(impl_->asyncState->healthMutex);
    return impl_->asyncState->healthReason;
}

}  // namespace splash::metal
