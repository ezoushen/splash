#include "../../../runtime/metal/MetalBackend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using splash::metal::AllocationFailure;
using splash::metal::MetalAllocationError;
using splash::metal::BufferBinding;
using splash::metal::BufferStorage;
using splash::metal::BytesBinding;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBackendError;
using splash::metal::MetalBuffer;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

struct TemporaryMetallib final {
    TemporaryMetallib()
        : path((std::filesystem::temp_directory_path() /
                "splash-metal-backend.XXXXXX").string()) {
        const int descriptor = ::mkstemp(path.data());
        if (descriptor < 0)
            throw std::runtime_error("could not create temporary metallib");
        ::close(descriptor);
    }
    ~TemporaryMetallib() { ::unlink(path.c_str()); }
    std::string path;
};

template <typename Function>
void requireBackendError(Function &&function, const std::string &message) {
    try {
        function();
    } catch (const MetalBackendError &) {
        return;
    }
    fail(message);
}

class MethodReplacement final {
public:
    MethodReplacement(id object, SEL selector, IMP replacement) {
        method_ = class_getInstanceMethod(object_getClass(object), selector);
        require(method_ != nullptr, "probe fault method is missing");
        original = method_setImplementation(method_, replacement);
    }
    ~MethodReplacement() { method_setImplementation(method_, original); }
    IMP original = nullptr;
private:
    Method method_ = nullptr;
};

thread_local bool inCompletionHandler = false;
IMP originalCompletedHandler = nullptr;
IMP originalAllocatedSize = nullptr;
std::promise<void> completedOnGpu;
std::promise<void> completionReturned;
std::shared_future<void> releaseMemoryQuery;
std::atomic<unsigned> completionMemoryQueries{0};
std::atomic<unsigned> memoryQueries{0};

void observeCompletion(id command, SEL selector, MTLCommandBufferHandler handler) {
    reinterpret_cast<void (*)(id, SEL, MTLCommandBufferHandler)>(
        originalCompletedHandler)(command, selector, ^(id<MTLCommandBuffer> completed) {
        completedOnGpu.set_value();
        inCompletionHandler = true;
        handler(completed);
        inCompletionHandler = false;
        completionReturned.set_value();
    });
}

NSUInteger delayedCompletionMemoryQuery(id device, SEL selector) {
    ++memoryQueries;
    if (inCompletionHandler) {
        ++completionMemoryQueries;
        releaseMemoryQuery.wait();
    }
    return reinterpret_cast<NSUInteger (*)(id, SEL)>(originalAllocatedSize)(device, selector);
}

void completionDoesNotWaitForMemoryTelemetry(const std::string &metallibPath) {
    MetalBackend backend(metallibPath, 0.1);
    auto buffer = backend.allocateBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(buffer.contents()) = 0;
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch;
    dispatch.pipelineName = "test_add_u32";
    dispatch.buffers = {{0, buffer}};
    dispatch.bytes = {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}};
    dispatch.threadgroups = {1, 1, 1};
    dispatch.threadsPerThreadgroup = {1, 1, 1};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    std::promise<void> release;
    releaseMemoryQuery = release.get_future().share();
    auto gpuDone = completedOnGpu.get_future();
    auto callbackDone = completionReturned.get_future();
    MethodReplacement completion(command, @selector(addCompletedHandler:),
                                 reinterpret_cast<IMP>(observeCompletion));
    originalCompletedHandler = completion.original;
    MethodReplacement memory(device, @selector(currentAllocatedSize),
                             reinterpret_cast<IMP>(delayedCompletionMemoryQuery));
    originalAllocatedSize = memory.original;
    auto ticket = backend.submitAsync(dispatch);
    const bool completed = gpuDone.wait_for(std::chrono::seconds(5)) ==
                           std::future_status::ready;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool ready = ticket.ready();
    bool healthy = true;
    try { backend.checkHealth(); }
    catch (const MetalBackendError &) { healthy = false; }
    release.set_value();
    const unsigned queriesBeforeConsumption = memoryQueries;
    (void)ticket.wait();
    require(callbackDone.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
            "completion handler did not drain after telemetry was released");
    require(completed, "test GPU command did not complete");
    require(ready && healthy && completionMemoryQueries == 0,
            "completed GPU work depends on memory telemetry and can trip the watchdog");
    require(memoryQueries > queriesBeforeConsumption,
            "consuming a completed command did not refresh admission telemetry");
    const unsigned queriesAfterConsumption = memoryQueries;
    (void)ticket.wait();
    require(memoryQueries == queriesAfterConsumption,
            "an already-released ticket queried device memory again");
    require(*static_cast<uint32_t *>(buffer.contents()) == increment,
            "completion telemetry test produced the wrong result");
    std::cout << "PASS GPU completion independent of memory telemetry\n";
}

id<MTLSharedEvent> commandWatchdogGate = nil;
std::promise<void> delayedCompletionStarted;
std::promise<void> delayedCompletionReturned;
std::shared_future<void> releaseCompletionNotification;
IMP originalCommandStatus = nullptr;
std::atomic<void *> failedCommand{nullptr};
bool injectCommandFailure = false;
std::atomic<bool> delayNextCompletion{false};

IMP originalCommandCommit = nullptr;
void commitBehindWatchdogGate(id command, SEL selector) {
    [command encodeWaitForEvent:commandWatchdogGate value:1];
    reinterpret_cast<void (*)(id, SEL)>(originalCommandCommit)(command, selector);
}

std::atomic<unsigned> commits{0};
IMP originalCountedCommit = nullptr;
void countCommit(id command, SEL selector) {
    ++commits;
    reinterpret_cast<void (*)(id, SEL)>(originalCountedCommit)(command, selector);
}

MTLCommandBufferStatus terminalCommandStatus(id command, SEL selector) {
    if ((__bridge void *)command == failedCommand.load())
        return MTLCommandBufferStatusError;
    return reinterpret_cast<MTLCommandBufferStatus (*)(id, SEL)>(
        originalCommandStatus)(command, selector);
}

void delayCompletionNotification(id command, SEL selector, MTLCommandBufferHandler handler) {
    if (!delayNextCompletion.exchange(false)) {
        reinterpret_cast<void (*)(id, SEL, MTLCommandBufferHandler)>(
            originalCompletedHandler)(command, selector, handler);
        return;
    }
    reinterpret_cast<void (*)(id, SEL, MTLCommandBufferHandler)>(
        originalCompletedHandler)(command, selector, ^(id<MTLCommandBuffer> completed) {
        require(completed.status == MTLCommandBufferStatusCompleted,
                "delayed notification test did not complete on the GPU");
        if (injectCommandFailure)
            failedCommand.store((__bridge void *)completed);
        delayedCompletionStarted.set_value();
        releaseCompletionNotification.wait();
        handler(completed);
        delayedCompletionReturned.set_value();
    });
}

void terminalCommandRecovers(const std::string &metallibPath, bool failed,
                                   bool pendingNext = false) {
    MetalBackend backend(metallibPath, 0.1);
    auto buffer = backend.allocateBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(buffer.contents()) = 0;
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch;
    dispatch.pipelineName = "test_add_u32";
    dispatch.buffers = {{0, buffer}};
    dispatch.bytes = {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}};
    dispatch.threadgroups = {1, 1, 1};
    dispatch.threadsPerThreadgroup = {1, 1, 1};
    id<MTLCommandQueue> queue = [MTLCreateSystemDefaultDevice() newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    std::promise<void> release;
    releaseCompletionNotification = release.get_future().share();
    delayedCompletionStarted = std::promise<void>{};
    delayedCompletionReturned = std::promise<void>{};
    auto gpuDone = delayedCompletionStarted.get_future();
    auto callbackDone = delayedCompletionReturned.get_future();
    std::atomic<bool> healthy{true};
    std::atomic<unsigned> notifications{0};
    std::string error;
    injectCommandFailure = failed;
    delayNextCompletion = true;
    {
        MethodReplacement status(command, @selector(status),
                                 reinterpret_cast<IMP>(terminalCommandStatus));
        originalCommandStatus = status.original;
        MethodReplacement completion(command, @selector(addCompletedHandler:),
                                     reinterpret_cast<IMP>(delayCompletionNotification));
        originalCompletedHandler = completion.original;
        auto ticket = backend.submitAsync(dispatch, [&](uint64_t) { ++notifications; });
        require(gpuDone.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "GPU did not reach the delayed completion handler");
        require(!ticket.ready(), "test did not delay the completion notification");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const auto check = [&] {
            try { backend.checkHealth(); }
            catch (const MetalBackendError &) { healthy = false; }
        };
        auto peer = std::async(std::launch::async, check);
        check();
        peer.get();
        require(ticket.ready(), "terminal command still depends on its completion handler");
        try { (void)ticket.wait(); }
        catch (const MetalBackendError &failure) { error = failure.what(); }
        require(notifications == 1, "host completion did not notify exactly once");
        require(callbackDone.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready,
                "test released the callback before consuming its result");
        require(*static_cast<uint32_t *>(buffer.contents()) == increment,
                "host completion returned the wrong GPU result");
        dispatch.buffers.clear();
        buffer = {};
        require(backend.memoryStats().allocatedBytes == 0,
                "terminal ticket retained allocations until the callback returned");
        splash::metal::CommandTicket next;
        if (!failed) {
            buffer = backend.allocateBuffer(sizeof(uint32_t));
            *static_cast<uint32_t *>(buffer.contents()) = 0;
            dispatch.buffers = {{0, buffer}};
            if (pendingNext) {
                commandWatchdogGate = [MTLCreateSystemDefaultDevice() newSharedEvent];
                MethodReplacement commit(command, @selector(commit),
                                         reinterpret_cast<IMP>(commitBehindWatchdogGate));
                originalCommandCommit = commit.original;
                next = backend.submitAsync(dispatch);
            } else {
                next = backend.submitAsync(dispatch);
            }
        }
        // Metal may serialize later status notifications behind this handler.
        release.set_value();
        require(callbackDone.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "delayed completion handler did not drain");
        if (pendingNext) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::string failure;
            try { backend.checkHealth(); }
            catch (const MetalBackendError &caught) { failure = caught.what(); }
            require(!next.ready() && failure.find("sequence=2") != std::string::npos,
                    "late callback disarmed the next command's watchdog");
            commandWatchdogGate.signaledValue = 1;
            (void)next.wait();
            commandWatchdogGate = nil;
        } else if (next) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!next.ready() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            require(next.ready(), "backend did not resume after the delayed callback");
            (void)next.wait();
        }
    }
    failedCommand.store(nullptr);
    require(healthy == !failed && notifications == 1,
            "completed GPU work timed out while its notification was delayed");
    if (failed) {
        require(!backend.healthy() && error.find("Metal command 1 failed") != std::string::npos,
                "delayed GPU failure was lost or misclassified: " + error);
        requireBackendError([&] { (void)backend.submitAsync(dispatch); },
                            "failed GPU command admitted further work");
        std::cout << "PASS delayed GPU failure preserves its error\n";
        return;
    }
    require(error.empty(), "successful command failed: " + error);
    require(*static_cast<uint32_t *>(buffer.contents()) == increment,
            "backend did not continue after the delayed completion");
    std::cout << (pendingNext ? "PASS late completion preserves the next watchdog\n"
                             : "PASS terminal command completes without its callback\n");
}

void pendingCommandStillTimesOut(const std::string &metallibPath) {
    MetalBackend backend(metallibPath, 0.1);
    auto buffer = backend.allocateBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(buffer.contents()) = 0;
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch;
    dispatch.pipelineName = "test_add_u32";
    dispatch.buffers = {{0, buffer}};
    dispatch.bytes = {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}};
    dispatch.threadgroups = {1, 1, 1};
    dispatch.threadsPerThreadgroup = {1, 1, 1};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    commandWatchdogGate = [device newSharedEvent];
    splash::metal::CommandTicket ticket;
    {
        MethodReplacement commit(command, @selector(commit),
                                 reinterpret_cast<IMP>(commitBehindWatchdogGate));
        originalCommandCommit = commit.original;
        ticket = backend.submitAsync(dispatch);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool pending = !ticket.ready();
    std::string failure;
    try { backend.checkHealth(); }
    catch (const MetalBackendError &error) { failure = error.what(); }
    commandWatchdogGate.signaledValue = 1;
    (void)ticket.wait();
    commandWatchdogGate = nil;
    require(pending && !backend.healthy(), "pending GPU command escaped the watchdog");
    require(failure.find("sequence=1") != std::string::npos &&
                failure.find("dispatches=1") != std::string::npos &&
                (failure.find("status=committed") != std::string::npos ||
                 failure.find("status=scheduled") != std::string::npos),
            "command timeout lost its submission diagnostics: " + failure);
    requireBackendError([&] { (void)backend.submitAsync(dispatch); },
                        "timed-out backend accepted more work");
    require(*static_cast<uint32_t *>(buffer.contents()) == increment,
            "timed-out command lost resources before GPU completion");
    std::cout << "PASS pending GPU command watchdog and resource lifetime\n";
}

std::atomic<unsigned> blitEncoders{0};
std::atomic<unsigned> computeEncoders{0};
IMP originalBlitEncoder = nullptr;
IMP originalComputeEncoder = nullptr;
id countBlitEncoder(id command, SEL selector) {
    ++blitEncoders;
    return reinterpret_cast<id (*)(id, SEL)>(originalBlitEncoder)(command, selector);
}
id countComputeEncoder(id command, SEL selector) {
    ++computeEncoders;
    return reinterpret_cast<id (*)(id, SEL)>(originalComputeEncoder)(command, selector);
}

// A lapsed keep-alive ends residency with one dispatch of a kernel built with
// the library, never a blit whose driver program compiles at that moment, and
// destroying a backend that holds its set submits no GPU work at all.
void residencyEndsWithoutBlits(const std::string &metallibPath) {
    constexpr double kKeepAliveSeconds = 0.2;
    id<MTLCommandBuffer> command =
        [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
    commits = 0;
    blitEncoders = 0;
    computeEncoders = 0;
    MethodReplacement committing(command, @selector(commit),
                                 reinterpret_cast<IMP>(countCommit));
    originalCountedCommit = committing.original;
    MethodReplacement blits(command, @selector(blitCommandEncoder),
                            reinterpret_cast<IMP>(countBlitEncoder));
    originalBlitEncoder = blits.original;
    MethodReplacement computes(command, @selector(computeCommandEncoder),
                               reinterpret_cast<IMP>(countComputeEncoder));
    originalComputeEncoder = computes.original;
    unsigned lapseCommits = 0, lapseComputes = 0;
    {
        MetalBackend backend(metallibPath, 120.0, kKeepAliveSeconds);
        const uint64_t page = static_cast<uint64_t>(getpagesize());
        MetalBuffer lapsing = backend.allocateBuffer(page);
        backend.keepResident(lapsing);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while ((!backend.lapsedResidentBytes() || !commits) &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        lapseCommits = commits.exchange(0);
        lapseComputes = computeEncoders.exchange(0);
        // Keeping another buffer holds the set again for the teardown.
        MetalBuffer held = backend.allocateBuffer(page);
        backend.keepResident(held);
        require(backend.lapsedResidentBytes() == 0, "a kept buffer did not hold the set");
    }
    require(lapseCommits == 1 && lapseComputes == 1,
            "a lapsed keep-alive did not end residency with one compute dispatch");
    require(commits == 0 && computeEncoders == 0,
            "backend teardown submitted GPU work");
    require(blitEncoders == 0, "residency encoded a blit");
    std::cout << "PASS residency ends without blits\n";
}

// Kept buffers stay held until the keep-alive passes without a command, the
// next command holds them again at once, and a buffer's last view takes it
// out of the set.
void keptBuffersStayResident(const std::string &metallibPath) {
    constexpr double kKeepAliveSeconds = 1.0;
    MetalBackend backend(metallibPath, 120.0, kKeepAliveSeconds);
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    MetalBuffer dropped = backend.allocateBuffer(page);
    MetalBuffer used = backend.allocateBuffer(page);
    const uint64_t each = backend.memoryStats().allocatedBytes / 2;
    const auto start = std::chrono::steady_clock::now();
    backend.keepResident(backend.view(dropped, 0, 64));
    backend.keepResident(used);
    require(backend.lapsedResidentBytes() == 0, "kept buffers were not held at once");
    requireBackendError([&] { backend.keepResident(dropped); },
                        "the base of a kept view was kept again");
    while (!backend.lapsedResidentBytes() &&
           std::chrono::steady_clock::now() - start < std::chrono::seconds(5))
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const std::chrono::duration<double> lapsedAfter = std::chrono::steady_clock::now() - start;
    require(backend.lapsedResidentBytes() == 2 * each &&
                lapsedAfter.count() >= kKeepAliveSeconds,
            "kept buffers did not lapse once the keep-alive passed without a command");
    dropped = {};
    require(backend.lapsedResidentBytes() == each,
            "a buffer whose last view is gone is still kept");
    const uint32_t count = 1, increment = 7;
    *static_cast<uint32_t *>(used.contents()) = 0;
    ComputeDispatch dispatch{"test_add_u32", {{0, used}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    auto ticket = backend.submitAsync(dispatch);
    require(backend.lapsedResidentBytes() == 0,
            "a command did not hold the kept buffers again");
    (void)ticket.wait();
    require(*static_cast<uint32_t *>(used.contents()) == increment,
            "a command on a kept buffer produced the wrong result");
    std::cout << "PASS kept buffers stay resident keep_alive_seconds=" << kKeepAliveSeconds
              << " lapsed_after_seconds=" << lapsedAfter.count() << '\n';
}

// Keeping, lapsing and holding again race the heartbeat while another thread
// drops kept buffers, as command completion can, and the backend is then
// destroyed with its heartbeat live and a kept buffer outliving it. Nothing
// may block, and every command must see its buffer.
void residencyRacesTheHeartbeat(const std::string &metallibPath) {
    constexpr double kKeepAliveSeconds = 0.05;
    constexpr int kRounds = 24;
    auto backend = std::make_unique<MetalBackend>(metallibPath, 120.0,
                                                  kKeepAliveSeconds);
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    MetalBuffer used = backend->allocateBuffer(page);
    backend->keepResident(used);
    *static_cast<uint32_t *>(used.contents()) = 0;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<MetalBuffer> handed;
    bool finished = false;
    std::thread dropper([&] {
        std::unique_lock lock(mutex);
        while (!finished || !handed.empty()) {
            ready.wait(lock, [&] { return finished || !handed.empty(); });
            std::vector<MetalBuffer> drop = std::move(handed);
            handed.clear();
            lock.unlock();
            drop.clear();
            lock.lock();
        }
    });
    const uint32_t count = 1, increment = 1;
    ComputeDispatch dispatch{"test_add_u32", {{0, used}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    int lapses = 0;
    for (int round = 0; round < kRounds; ++round) {
        MetalBuffer kept = backend->allocateBuffer(page);
        backend->keepResident(kept);
        {
            std::lock_guard lock(mutex);
            handed.push_back(std::move(kept));
        }
        ready.notify_one();
        // Every third round lets the heartbeat end residency, so that its
        // command holds the set again.
        if (round % 3 == 2) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!backend->lapsedResidentBytes() &&
                   std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            lapses += backend->lapsedResidentBytes() != 0;
        }
        (void)backend->submitAsync(dispatch).wait();
    }
    {
        std::lock_guard lock(mutex);
        finished = true;
    }
    ready.notify_one();
    dropper.join();
    require(*static_cast<uint32_t *>(used.contents()) == kRounds,
            "a command racing the residency heartbeat produced the wrong result");
    require(lapses == kRounds / 3, "residency did not lapse between the racing commands");
    (void)backend->submitAsync(dispatch).wait();
    backend.reset();
    used = {};
    std::cout << "PASS residency races the heartbeat rounds=" << kRounds
              << " lapses=" << lapses << '\n';
}

IMP originalNewBuffer = nullptr;
MTLResourceOptions lastBufferOptions = 0;
id recordBufferOptions(id device, SEL selector, NSUInteger length,
                       MTLResourceOptions options) {
    lastBufferOptions = options;
    return reinterpret_cast<id (*)(id, SEL, NSUInteger, MTLResourceOptions)>(
        originalNewBuffer)(device, selector, length, options);
}

IMP originalBufferAllocatedSize = nullptr;
NSUInteger paddedAllocatedSize(id buffer, SEL selector) {
    return reinterpret_cast<NSUInteger (*)(id, SEL)>(
               originalBufferAllocatedSize)(buffer, selector) + 16384;
}

// Addressed buffers are private and untracked, allocated at exactly their
// size and reached only through GPU addresses in a table. The residency set
// makes them resident for every command, also once its keep-alive has
// lapsed; one dispatch reads what the previous one wrote through them; and
// buffers released and allocated again between commands work at once.
void addressedBuffersThroughTables(const std::string &metallibPath) {
    constexpr double kKeepAliveSeconds = 0.2;
    constexpr uint32_t kBuffers = 6, kWords = 16384, kRounds = 60;
    constexpr uint64_t kBytes = uint64_t{kWords} * sizeof(uint32_t);
    MetalBackend backend(metallibPath, 120.0, kKeepAliveSeconds);
    MetalBuffer table = backend.allocateBuffer(kBuffers * sizeof(uint64_t));
    MetalBuffer mismatches = backend.allocateBuffer(sizeof(uint32_t));
    const uint64_t before = backend.memoryStats().allocatedBytes;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    std::vector<MetalBuffer> buffers(kBuffers);
    {
        MethodReplacement options(device, @selector(newBufferWithLength:options:),
                                  reinterpret_cast<IMP>(recordBufferOptions));
        originalNewBuffer = options.original;
        buffers[0] = backend.allocateAddressed(kBytes, "addressed-test");
    }
    require(lastBufferOptions == (MTLResourceStorageModePrivate |
                                  MTLResourceHazardTrackingModeUntracked),
            "an addressed buffer is not private and hazard-untracked");
    {
        id<MTLBuffer> sample = [device newBufferWithLength:kBytes
            options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
        MethodReplacement padded(sample, @selector(allocatedSize),
                                 reinterpret_cast<IMP>(paddedAllocatedSize));
        originalBufferAllocatedSize = padded.original;
        try {
            (void)backend.allocateAddressed(kBytes);
            fail("an addressed buffer larger than its size was accepted");
        } catch (const MetalAllocationError &) {
        }
    }
    for (uint32_t index = 1; index < kBuffers; ++index)
        buffers[index] = backend.allocateAddressed(kBytes);
    require(buffers[0].storage() == BufferStorage::Private && !buffers[0].contents() &&
                buffers[0].sizeBytes() == kBytes &&
                backend.memoryStats().allocatedBytes == before + kBuffers * kBytes,
            "addressed buffers were not allocated or counted at their size");
    require(buffers[0].gpuAddress() &&
                backend.view(buffers[0], 4096, 4096).gpuAddress() ==
                    buffers[0].gpuAddress() + 4096,
            "a view's GPU address does not start at its offset");
    require(backend.lapsedResidentBytes() == kBuffers * kBytes,
            "addressed buffers are not members of the residency set");

    auto *entries = static_cast<uint64_t *>(table.contents());
    for (uint32_t index = 0; index < kBuffers; ++index)
        entries[index] = buffers[index].gpuAddress();
    const uint32_t words = kWords;
    uint32_t seed = 0;
    const std::array<ComputeDispatch, 2> command{
        ComputeDispatch{"addressed_write_u32", {{0, table}},
            {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
            {kWords / 256, kBuffers, 1}, {256, 1, 1}},
        ComputeDispatch{"addressed_check_u32", {{0, table}, {3, mismatches}},
            {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
            {kWords / 256, kBuffers, 1}, {256, 1, 1}}};
    uint32_t regrown = 0;
    bool lapsedRound = false;
    for (uint32_t round = 0; round < kRounds; ++round) {
        if (round % 5 == 4) {
            const uint32_t index = round % kBuffers;
            buffers[index] = {};
            require(backend.memoryStats().allocatedBytes ==
                        before + (kBuffers - 1) * kBytes,
                    "a released addressed buffer is still counted");
            buffers[index] = backend.allocateAddressed(kBytes);
            entries[index] = buffers[index].gpuAddress();
            ++regrown;
        }
        if (round == kRounds / 2) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!backend.lapsedResidentBytes() &&
                   std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            require(backend.lapsedResidentBytes() == kBuffers * kBytes,
                    "addressed buffers did not lapse with the residency set");
            lapsedRound = true;
        }
        seed = 0x9e3779b9u * (round + 1);
        *static_cast<uint32_t *>(mismatches.contents()) = 0;
        auto ticket = backend.submitCommandAsync(command);
        require(backend.commandInFlight(), "a submitted command is not in flight");
        (void)ticket.wait();
        require(!backend.commandInFlight(), "a consumed command is still in flight");
        require(*static_cast<uint32_t *>(mismatches.contents()) == 0,
                "round " + std::to_string(round) +
                    " read wrong data through the addresses of its buffers");
    }
    buffers.clear();
    require(backend.memoryStats().allocatedBytes == before &&
                backend.lapsedResidentBytes() == 0,
            "released addressed buffers stayed counted or in the residency set");
    std::cout << "PASS addressed buffers through tables rounds=" << kRounds
              << " regrown=" << regrown << " lapsed_round=" << lapsedRound << '\n';
}

void sharedMemoryCompletionLifetime(MetalBackend &backend) {
    struct Gate {
        std::mutex mutex;
        std::condition_variable condition;
        bool entered = false;
        bool release = false;
        std::atomic<bool> ownerReleased{false};
    };
    auto gate = std::make_shared<Gate>();
    const size_t bytes = static_cast<size_t>(getpagesize());
    void *address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
    require(address != MAP_FAILED, "unable to allocate lifetime witness");
    auto owner = std::shared_ptr<void>(address, [gate, bytes](void *memory) {
        munmap(memory, bytes);
        gate->ownerReleased.store(true);
    });
    auto buffer = backend.wrapSharedMemory(address, bytes, owner);
    owner.reset();
    const uint32_t count = 1, increment = 1;
    ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    auto ticket = backend.submitAsync(dispatch, [gate](uint64_t) {
        std::unique_lock lock(gate->mutex);
        gate->entered = true;
        gate->condition.notify_all();
        gate->condition.wait_for(lock, std::chrono::seconds(5),
                                [&] { return gate->release; });
    });
    dispatch.buffers.clear();
    buffer = {};
    {
        std::unique_lock lock(gate->mutex);
        require(gate->condition.wait_for(lock, std::chrono::seconds(5),
                                         [&] { return gate->entered; }),
                "lifetime witness completion did not arrive");
    }
    require(!gate->ownerReleased.load(),
            "external memory released while the command ticket still owns it");
    // Applying the ticket drops its C++ allocations. Metal may release the
    // underlying buffer before, during or after the completion callback;
    // only the ticket-owned lifetime above and eventual release are required.
    (void)ticket.wait();
    {
        std::lock_guard lock(gate->mutex);
        gate->release = true;
    }
    gate->condition.notify_all();
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    while (!gate->ownerReleased.load() &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(gate->ownerReleased.load(),
            "external memory leaked after Metal released its buffer");
}

void run(const std::string &metallibPath) {
    NSData *libraryData = [NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:metallibPath.c_str()]];
    TemporaryMetallib temporary;
    NSString *temporaryPath = [NSString stringWithUTF8String:temporary.path.c_str()];
    require([libraryData writeToFile:temporaryPath options:0 error:nullptr],
            "could not copy test metallib");
    MetalBackend backend(temporary.path);
    const auto guardedBytes = backend.memoryStats().allocatedBytes;
    const auto denyOperation = [] {
        throw MetalAllocationError("test host pressure", AllocationFailure::HostPressure);
    };
    backend.setOperationGuard(denyOperation);
    try {
        (void)backend.allocateBuffer(16384, BufferStorage::Shared);
        fail("operation guard admitted an allocation");
    } catch (const MetalAllocationError &error) {
        require(error.failure() == AllocationFailure::HostPressure,
                "operation guard lost its failure classification");
    }
    require(backend.healthy() && backend.memoryStats().allocatedBytes == guardedBytes,
            "operation guard leaked memory or poisoned the backend");
    backend.setOperationGuard({});

    NSData *replacement = [@"replaced after library loading"
        dataUsingEncoding:NSUTF8StringEncoding];
    require([replacement writeToFile:temporaryPath
                            options:NSDataWritingAtomic error:nullptr],
            "could not replace temporary metallib");
    // All existing pipeline/dispatch checks below run from the original
    // loaded library even though its former path now contains invalid bytes.
    const auto &capabilities = backend.capabilities();
    require(!capabilities.deviceName.empty(), "device name is empty");
    require(capabilities.physicalMemoryBytes > 0,
            "physical memory capability is missing");
    require(capabilities.recommendedMaxWorkingSetBytes > 0,
            "recommended working set capability is missing");
    require(capabilities.maxBufferLengthBytes > 0,
            "maximum buffer length capability is missing");
    require(capabilities.appleGpuFamily >= 9,
            "Apple GPU family capability is missing");
    require(capabilities.maxThreadgroupMemoryBytes >= 32 * 1024,
            "threadgroup memory capability is insufficient");
    require(capabilities.maxThreadgroupWidth >= 256,
            "threadgroup thread capability is insufficient");
    const auto probed = splash::metal::probeDeviceCapabilities();
    require(probed.deviceName == capabilities.deviceName &&
                probed.appleGpuFamily == capabilities.appleGpuFamily &&
                probed.macosVersion() == capabilities.macosVersion() &&
                !probed.validationMessage(),
            "the device check read the device differently from the backend");
    require(backend.healthy(), "new backend is unhealthy");
    require(backend.submissionCount() == 0, "new backend has submissions");
    require(backend.pipelineCount() == 0, "pipeline cache is not empty");

    constexpr uint32_t kElementCount = 64;
    constexpr uint32_t kViewElementCount = kElementCount / 2;
    constexpr uint32_t kIncrement = 7;
    constexpr uint64_t kAllocationBytes =
        sizeof(uint32_t) * kElementCount;

    MetalBuffer base = backend.allocateBuffer(
        kAllocationBytes, BufferStorage::Shared, "metal-backend-test");
    require(base && base.contents(), "shared allocation is not CPU-visible");
    require(base.sizeBytes() == kAllocationBytes,
            "allocation length is unexpected");

    auto stats = backend.memoryStats();
    const uint64_t actualAllocationBytes = stats.allocatedBytes;
    require(actualAllocationBytes >= base.sizeBytes(),
            "actual live allocation bytes were not tracked");
    require(stats.peakAllocatedBytes == actualAllocationBytes,
            "peak allocation bytes were not tracked");
    require(stats.deviceCurrentAllocatedBytes >= actualAllocationBytes,
            "device allocation counter is smaller than backend allocations");
    require(stats.devicePeakAllocatedBytes >=
                stats.deviceCurrentAllocatedBytes,
            "device peak counter is smaller than current allocations");

    auto *values = static_cast<uint32_t *>(base.contents());
    for (uint32_t i = 0; i < kElementCount; ++i) values[i] = i;

    MetalBuffer view = backend.view(
        base, sizeof(uint32_t) * kViewElementCount,
        sizeof(uint32_t) * kViewElementCount);
    require(view.contents() == values + kViewElementCount,
            "view contents pointer has the wrong offset");
    require(view.sameView(backend.view(base, sizeof(uint32_t) * kViewElementCount,
                                      sizeof(uint32_t) * kViewElementCount)) &&
                view.sameView(backend.view(view, 0, view.sizeBytes())) &&
                !view.sameView(base) && !view.sameView(MetalBuffer{}) &&
                MetalBuffer{}.sameView(MetalBuffer{}),
            "buffer view identity does not compare allocation and exact range");
    require(backend.memoryStats().allocatedBytes == actualAllocationBytes,
            "view was counted as a new allocation");

    double lastWallSeconds = 0.0;
    {
        ComputeDispatch dispatch;
        dispatch.pipelineName = "test_add_u32";
        dispatch.buffers.push_back(BufferBinding{0, view});
        dispatch.bytes.push_back(BytesBinding{
            1, &kViewElementCount, sizeof(kViewElementCount)});
        dispatch.bytes.push_back(
            BytesBinding{2, &kIncrement, sizeof(kIncrement)});
        dispatch.threadgroups = {1, 1, 1};
        dispatch.threadsPerThreadgroup = {kViewElementCount, 1, 1};

        backend.setOperationGuard(denyOperation);
        try {
            (void)backend.submit(dispatch);
            fail("operation guard admitted a GPU submission");
        } catch (const MetalAllocationError &error) {
            require(error.failure() == AllocationFailure::HostPressure &&
                        backend.healthy() && backend.submissionCount() == 0,
                    "guarded submission lost its cause or altered the backend");
        }
        backend.setOperationGuard({});

        for (int runIndex = 0; runIndex < 2; ++runIndex) {
            splash::metal::CommandTiming timing;
            if (!runIndex) {
                timing = backend.submit(dispatch);
            } else {
                std::promise<uint64_t> completedSequence;
                auto notified = completedSequence.get_future();
                auto ticket = backend.submitAsync(
                    dispatch, [&](uint64_t sequence) {
                        completedSequence.set_value(sequence);
                    });
                require(ticket && ticket.sequence() > 0,
                        "async submission returned an empty ticket");
                requireBackendError(
                    [&] { (void)backend.submit(dispatch); },
                    "a second in-flight command was accepted");
                timing = ticket.wait();
                require(ticket.ready(),
                        "completed async ticket is not ready");
                // Completion is published before the callback runs, so
                // wait() may return first; only the callback's own signal
                // shows the notification was delivered.
                require(notified.wait_for(std::chrono::seconds(5)) ==
                                std::future_status::ready &&
                            notified.get() == ticket.sequence(),
                        "async completion notification was not delivered");
            }
            require(std::isfinite(timing.gpuSeconds) &&
                        timing.gpuSeconds >= 0.0,
                    "GPU timing is invalid");
            require(std::isfinite(timing.wallSeconds) &&
                        timing.wallSeconds > 0.0,
                    "wall timing is invalid");
            lastWallSeconds = timing.wallSeconds;
        }
    }

    require(backend.submissionCount() == 2,
            "successful submissions were not counted");
    require(backend.pipelineCount() == 1,
            "pipeline cache did not reuse the pipeline");
    for (uint32_t i = 0; i < kViewElementCount; ++i) {
        require(values[i] == i, "dispatch wrote before the buffer view");
    }
    for (uint32_t i = kViewElementCount; i < kElementCount; ++i) {
        require(values[i] == i + 2 * kIncrement,
                "dispatch produced an incorrect result");
    }

    {
        ComputeDispatch first;
        first.pipelineName = "test_add_u32";
        first.buffers.push_back(BufferBinding{0, view});
        first.bytes.push_back(BytesBinding{
            1, &kViewElementCount, sizeof(kViewElementCount)});
        first.bytes.push_back(
            BytesBinding{2, &kIncrement, sizeof(kIncrement)});
        first.threadgroups = {1, 1, 1};
        first.threadsPerThreadgroup = {kViewElementCount, 1, 1};
        std::vector<ComputeDispatch> command{first, first};
        (void)backend.submitCommand(command);
    }
    require(backend.submissionCount() == 3,
            "explicit operation list did not use one command buffer");
    for (uint32_t i = kViewElementCount; i < kElementCount; ++i) {
        require(values[i] == i + 4 * kIncrement,
                "multi-dispatch command produced an incorrect result");
    }

    requireBackendError(
        [&] { (void)backend.view(base, base.sizeBytes(), 1); },
        "out-of-range view was accepted");

    ComputeDispatch missingPipeline;
    missingPipeline.pipelineName = "does_not_exist";
    requireBackendError(
        [&] { (void)backend.submit(missingPipeline); },
        "missing pipeline was accepted");
    {
        // A binding takes one of the argument table's 31 entries of its own.
        ComputeDispatch rebound;
        rebound.pipelineName = "test_add_u32";
        rebound.buffers = {{0, view}};
        rebound.bytes = {{0, &kIncrement, sizeof(kIncrement)}};
        requireBackendError(
            [&] { (void)backend.submit(rebound); },
            "a binding index bound twice was accepted");
        rebound.bytes = {{31, &kIncrement, sizeof(kIncrement)}};
        requireBackendError(
            [&] { (void)backend.submit(rebound); },
            "a binding index past the argument table was accepted");
    }
    require(backend.healthy(),
            "a descriptor error incorrectly poisoned the backend");
    require(backend.unhealthyReason().empty(),
            "healthy backend has an unhealthy reason");
    require(backend.submissionCount() == 3,
            "failed pre-commit dispatch was counted as submitted");
    require(backend.pipelineCount() == 1,
            "failed pipeline lookup polluted the cache");

    base = MetalBuffer{};
    require(backend.memoryStats().allocatedBytes == actualAllocationBytes,
            "a live view did not retain its base allocation");
    view = MetalBuffer{};
    stats = backend.memoryStats();
    require(stats.allocatedBytes == 0,
            "released allocation remains in live byte accounting");
    require(stats.peakAllocatedBytes == actualAllocationBytes,
            "peak allocation accounting changed after release");
    require(stats.devicePeakAllocatedBytes >=
                stats.deviceCurrentAllocatedBytes,
            "device peak allocation accounting regressed");

    MetalBuffer privateBuffer = backend.allocateBuffer(
        16, BufferStorage::Private, "private-test");
    require(privateBuffer.contents() == nullptr,
            "private allocation unexpectedly exposed CPU contents");
    require(privateBuffer.sameView(backend.view(privateBuffer, 0, 16)) &&
                !privateBuffer.sameView(backend.view(privateBuffer, 0, 8)),
            "private buffer view identity depended on CPU visibility");
    privateBuffer = MetalBuffer{};
    require(backend.memoryStats().allocatedBytes == 0,
            "private allocation release was not tracked");

    sharedMemoryCompletionLifetime(backend);
    require(capabilities.gpuCoreCount >= 1 && capabilities.gpuCoreCount <= 4096,
            "GPU core count was not read from the IORegistry");
    require(capabilities.meetsMinimumMacos(),
            "the running macOS version was not recorded");

    std::cout << "PASS MetalBackend device=\"" << capabilities.deviceName
              << "\" apple_gpu_family=" << capabilities.appleGpuFamily
              << " gpu_core_count=" << capabilities.gpuCoreCount
              << " macos=" << capabilities.macosVersion()
              << " recommended_working_set="
              << capabilities.recommendedMaxWorkingSetBytes
              << " last_wall_seconds=" << lastWallSeconds << '\n';
}

}  // namespace

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            std::cerr << "usage: metal_backend_test <test.metallib>\n";
            return 2;
        }
        try {
            completionDoesNotWaitForMemoryTelemetry(argv[1]);
            terminalCommandRecovers(argv[1], false);
            terminalCommandRecovers(argv[1], false, true);
            terminalCommandRecovers(argv[1], true);
            pendingCommandStillTimesOut(argv[1]);
            keptBuffersStayResident(argv[1]);
            residencyRacesTheHeartbeat(argv[1]);
            residencyEndsWithoutBlits(argv[1]);
            addressedBuffersThroughTables(argv[1]);
            run(argv[1]);
        } catch (const std::exception &error) {
            std::cerr << "FAIL: unexpected exception: " << error.what()
                      << '\n';
            return 1;
        }
    }
    return 0;
}
