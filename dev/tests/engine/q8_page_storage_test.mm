#include "ops/PageStorage.hpp"
#include "engine/MemoryGovernor.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void requireThrows(Function &&function, const char *message) {
    try {
        function();
    } catch (const Exception &) {
        return;
    }
    throw std::runtime_error(message);
}

// A pool's extents hold whole alignment units, so that every tensor region of
// an extent starts 64 KiB-aligned: the 512-byte-per-page scale regions need
// 128 pages for four KV heads and 256 for two, BF16 one or two. The size is
// chosen per pool between half and one and a half times the 128 MiB target,
// leaving the fewest of its pages over, the one nearest the target on a tie.
constexpr kv::Layout kvLayout{16, 4, 256};
constexpr kv::Layout compactLayout{10, 2, 256};
constexpr kv::Layout bf16Layout{16, 4, 256, kv::Format::BFloat16};
constexpr kv::Layout compactBf16Layout{10, 2, 256, kv::Format::BFloat16};
static_assert(kv::kExtentRegionAlignmentBytes == 64 * 1024);
static_assert(kvLayout.bytesPerModelPage() == 1'064'960);
static_assert(kvLayout.extentAlignmentPages() == 128);
static_assert(kvLayout.minimumExtentPages() == 128 && kvLayout.maximumExtentPages() == 128);
static_assert(kvLayout.extentPagesFor(127) == 0 && kvLayout.extentPagesFor(128) == 128 &&
              kvLayout.extentPagesFor(1000) == 128);
static_assert(compactLayout.bytesPerModelPage() == 332'800);
static_assert(compactLayout.extentAlignmentPages() == 256);
static_assert(compactLayout.minimumExtentPages() == 256 &&
              compactLayout.maximumExtentPages() == 512);
static_assert(compactLayout.extentPagesFor(255) == 0 && compactLayout.extentPagesFor(511) == 256);
// Both sizes leave nothing over: 512 pages (162.5 MiB) is nearer the target
// than 256 (81.25 MiB). With 256 pages over, only 256 leaves nothing.
static_assert(compactLayout.extentPagesFor(10'240) == 512);
static_assert(compactLayout.extentPagesFor(10'496) == 256);
static_assert(bf16Layout.extentAlignmentPages() == 1 && bf16Layout.minimumExtentPages() == 32 &&
              bf16Layout.maximumExtentPages() == 96);
// 448 pages divide by 32, 56 and 64 (128 MiB, the target); 97 leaves one
// page over 32, 48 and 96 extents, of which 48 (96 MiB) is nearest.
static_assert(bf16Layout.extentPagesFor(31) == 0 && bf16Layout.extentPagesFor(448) == 64 &&
              bf16Layout.extentPagesFor(97) == 48);
static_assert(compactBf16Layout.extentAlignmentPages() == 2 &&
              compactBf16Layout.minimumExtentPages() == 104 &&
              compactBf16Layout.maximumExtentPages() == 306);

// The host reaches a page through spans of its tensors, layer by layer as
// keys, key scales, values and value scales, BF16 without scales. Kernels
// find a page's keys at the page's slab in the layer's region, so that is
// where each layer's spans start, and the spans of an extent's pages cover
// the extent once, without gaps or overlaps.
void requireSpansTileExtent(const kv::PageStorage &storage, uint32_t firstPage) {
    const kv::Layout layout = storage.layout();
    const bool scaled = layout.format == kv::Format::Int8;
    const uint32_t tensors = scaled ? 4 : 2;
    std::byte *const extent = storage.spans(firstPage).front().data();
    std::vector<std::span<std::byte>> all;
    for (uint32_t index = 0; index < storage.extentPages(); ++index) {
        const auto spans = storage.spans(firstPage + index);
        require(spans.size() == layout.attentionLayers * tensors,
                "a page does not have one span per tensor of every layer");
        for (uint32_t span = 0; span < spans.size(); ++span) {
            const uint64_t bytes = scaled && span % 2 ? layout.scaleBytesPerLayerPage()
                                                      : layout.dataBytesPerLayerPage();
            require(spans[span].size() == bytes, "a page's span holds another tensor's bytes");
        }
        for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer) {
            require(spans[layer * tensors].data() ==
                        extent + storage.layer(layer).kv.offset +
                            uint64_t{index} * layout.dataBytesPerLayerPage(),
                    "a layer's spans do not start at the page's keys in its region");
        }
        all.insert(all.end(), spans.begin(), spans.end());
    }
    std::sort(all.begin(), all.end(),
              [](auto left, auto right) { return left.data() < right.data(); });
    std::byte *next = extent;
    for (const auto span : all) {
        require(span.data() == next, "the spans of an extent's pages leave a gap or overlap");
        next += span.size();
    }
    require(next == extent + storage.extentBytes(), "the spans of an extent's pages do not cover it");
}

void run(const std::string &metallib) {
    metal::MetalBackend backend(metallib);
    auto baseline = backend.memoryStats();
    uint64_t observed = std::max(
        baseline.allocatedBytes, baseline.deviceCurrentAllocatedBytes);
    constexpr uint64_t giB = 1ULL << 30;
    constexpr uint64_t hostReserve = 64 * 1024;
    std::optional<uint64_t> fakeHostAvailable = hostReserve + 3 * giB;
    MemoryGovernor bounded(
        backend, observed + 64 * 1024, hostReserve,
        [&fakeHostAvailable] {
            return fakeHostAvailable;
        });
    auto reservation = bounded.tryReserve(64 * 1024);
    require(reservation.has_value() &&
                bounded.snapshot().reservedBytes == 64 * 1024,
            "memory governor did not reserve exact growth bytes");
    metal::AllocationFailure failure;
    require(!bounded.tryReserve(1, &failure).has_value() &&
                failure == metal::AllocationFailure::EngineBudget &&
                bounded.snapshot().deniedReservations == 1,
            "memory governor oversold its hard ceiling");
    reservation->commit();
    auto admit = bounded.allocationAdmission();
    const auto driverDenied = admit(1024, [] {
      throw metal::MetalAllocationError("injected driver allocation denial");
    });
    require(!driverDenied &&
                driverDenied.failure == metal::AllocationFailure::DriverRejected &&
                bounded.snapshot().reservedBytes == 0,
            "driver allocation denial did not release its reservation");
    bool defectPropagated = false;
    try {
      (void)admit(1024, [] {
        throw metal::MetalBackendError("injected backend defect");
      });
    } catch (const metal::MetalBackendError &) {
      defectPropagated = true;
    }
    require(defectPropagated && bounded.snapshot().reservedBytes == 0,
            "admission swallowed a backend defect or leaked its reservation");
    require(admit(1024, [] {}) && bounded.snapshot().reservedBytes == 0,
            "driver allocation denial poisoned later admission");
    // A request may cross the warning margin while the idle headroom still
    // clears it. Its exact refusal reason must remain retryable, and the
    // refusal holds host pressure so that paced reclaim starts.
    fakeHostAvailable = hostReserve + giB + 512;
    bool allocated = false;
    const auto hostDenied = admit(1024, [&] { allocated = true; });
    require(!hostDenied && !allocated &&
                hostDenied.failure == metal::AllocationFailure::HostPressure &&
                bounded.snapshot().pressure == MemoryPressure::Warning &&
                bounded.snapshot().reservedBytes == 0,
            "request-sized host refusal lost its cause or ran allocation");
    fakeHostAvailable = hostReserve + 3 * giB;
    bounded.setPressure(MemoryPressure::Critical);
    require(!bounded.tryReserve(1).has_value(),
            "critical pressure did not stop new growth");
    bounded.setPressure(MemoryPressure::Normal);
    require(bounded.tryReserve(1).has_value(),
            "normal pressure did not reopen physical admission");
    fakeHostAvailable = hostReserve;
    require(!bounded.tryReserve(1).has_value(),
            "host reserve did not stop unified-memory growth");
    // Reaching the reserve is a warning that sheds cache in paced passes;
    // only the OS critical verdict drops every evictable entry.
    require(bounded.snapshot().pressure == MemoryPressure::Warning,
            "reaching the host reserve was treated as critical");
    fakeHostAvailable = hostReserve / 2;
    require(bounded.snapshot().pressure == MemoryPressure::Warning &&
                !bounded.tryReserve(1).has_value(),
            "low availability escalated to destructive system pressure");
    fakeHostAvailable = hostReserve + giB / 2;
    require(bounded.snapshot().pressure == MemoryPressure::Warning &&
                !bounded.snapshot().growthAllowed &&
                !bounded.tryReserve(1).has_value(),
            "low host headroom did not request proactive cache reclaim");
    fakeHostAvailable = hostReserve + 3 * giB / 2;
    require(bounded.snapshot().pressure == MemoryPressure::Warning,
            "warning pressure recovered without crossing the hysteresis");
    fakeHostAvailable = hostReserve + 3 * giB;
    require(bounded.snapshot().pressure == MemoryPressure::Normal &&
                bounded.snapshot().growthAllowed,
            "host recovery did not reopen physical admission");
    {
        auto reservation = bounded.tryReserve(64 * 1024);
        require(reservation.has_value(), "engine capacity reservation failed");
        const auto full = bounded.snapshot();
        require(!full.growthAllowed && full.hostGrowthAllowed,
                "engine budget exhaustion was confused with host pressure");
    }
    fakeHostAvailable.reset();
    require(!bounded.tryReserve(1).has_value() &&
                !bounded.snapshot().hostMeasurementValid,
            "missing host memory telemetry did not fail closed");
    MemoryPressurePolicy missingPolicy;
    auto missing = bounded.snapshot();
    auto missingDirective = missingPolicy.update(missing, 0.0, false);
    require(missing.pressure == MemoryPressure::Warning &&
                !missingDirective.evictAllUnpinnedPrefixes &&
                missingDirective.targetBytes == 0,
            "missing telemetry discarded valid cache");
    bounded.setPressure(MemoryPressure::Critical);
    require(missingPolicy.update(bounded.snapshot(), 1.0, false).evictAllUnpinnedPrefixes,
            "missing telemetry hid critical system pressure");
    bounded.setPressure(MemoryPressure::Normal);
    fakeHostAvailable = hostReserve + 3 * giB;
    require(bounded.snapshot().hostHeadroomBytes == 3 * giB,
            "host memory headroom accounting is wrong");

    // Low reclaimable memory must not reopen physical growth or suppress
    // the existing pressure reclaimer. Reclaimable file cache can.
    HostMemoryPages pressurePages{.free = hostReserve, .fileBacked = giB / 2};
    fakeHostAvailable = estimateHostAvailableMemory(pressurePages, 1);
    MemoryPressurePolicy hostPolicy;
    auto hostDirective = hostPolicy.update(bounded.snapshot(), 0.0, false);
    require(!bounded.tryReserve(1).has_value() &&
                bounded.snapshot().pressure == MemoryPressure::Warning &&
                hostDirective.reclaimEmptyKvExtents &&
                !hostDirective.evictAllUnpinnedPrefixes &&
                hostDirective.targetBytes == giB,
            "low reclaimable memory bypassed bounded pressure recovery");
    pressurePages.fileBacked = 3 * giB;
    fakeHostAvailable = estimateHostAvailableMemory(pressurePages, 1);
    require(bounded.snapshot().growthAllowed &&
                bounded.tryReserve(1).has_value() &&
                !hostPolicy.update(bounded.snapshot(), 1000.0, false).reclaimEmptyKvExtents,
            "reclaimable host recovery did not reopen normal admission");
    bounded.setPressure(MemoryPressure::Warning);
    require(bounded.snapshot().pressure == MemoryPressure::Warning &&
                bounded.snapshot().growthAllowed &&
                bounded.tryReserve(1).has_value(),
            "system warning blocked growth despite sufficient host headroom");
    fakeHostAvailable = hostReserve + giB / 2;
    require(!bounded.tryReserve(1).has_value(),
            "system warning bypassed insufficient host headroom");
    fakeHostAvailable = hostReserve + 3 * giB;
    bounded.setPressure(MemoryPressure::Normal);

    MemoryPressurePolicy policy;
    MemoryGovernorSnapshot policySnapshot;
    policySnapshot.pressure = MemoryPressure::Warning;
    policySnapshot.hostMeasurementValid = true;
    policySnapshot.systemPressure = MemoryPressure::Warning;
    policySnapshot.hostHeadroomBytes = 3 * giB / 2;
    auto firstDirective = policy.update(policySnapshot, 0.0, false);
    require(firstDirective.reclaimEmptyKvExtents &&
                !firstDirective.evictAllUnpinnedPrefixes &&
                firstDirective.targetBytes == giB / 2,
            "warning pressure ignored measured headroom");
    require(policy.update(policySnapshot, 500.0, false).targetBytes == 0 &&
                policy.update(policySnapshot, 999.0, false).targetBytes == 0,
            "warning pressure reclaimed again before telemetry settled");
    // Another application consumed more memory in the SAME warning episode.
    // Earlier reclaimed bytes must not offset this new deficit.
    policySnapshot.hostHeadroomBytes = giB / 4;
    require(policy.update(policySnapshot, 1000.0, false).targetBytes == giB,
            "persistent warning did not request a new bounded shrink pass");
    policySnapshot.systemPressure = MemoryPressure::Normal;
    policySnapshot.hostHeadroomBytes = 7 * giB / 4;
    require(policy.update(policySnapshot, 2000.0, false).targetBytes == giB / 4,
            "pressure recovery ignored the current smaller deficit");
    policySnapshot.pressure = MemoryPressure::Normal;
    require(!policy.update(policySnapshot, 2100.0, false).reclaimEmptyKvExtents,
            "normal pressure requested cache reclaim");
    policySnapshot.pressure = MemoryPressure::Warning;
    require(policy.update(policySnapshot, 2101.0, false).targetBytes == giB / 4,
            "a new pressure episode inherited an old cooldown");
    policySnapshot.systemPressure = MemoryPressure::Warning;
    policySnapshot.hostHeadroomBytes = 3 * giB;
    const auto advisory = policy.update(policySnapshot, 3101.0, false);
    require(advisory.reclaimEmptyKvExtents && !advisory.evictAllUnpinnedPrefixes &&
                advisory.targetBytes == 0,
            "system warning discarded live cache despite sufficient headroom");
    // The newest publication is what a follow-up resumes from; rebuilding it
    // costs a whole prefill, so a shrink nothing is waiting for leaves it and
    // takes the rest. A waiting request outranks it, and Critical takes all.
    policySnapshot.systemPressure = MemoryPressure::Warning;
    policySnapshot.hostHeadroomBytes = giB / 4;
    const auto speculative = policy.update(policySnapshot, 4101.0, false);
    require(speculative.targetBytes == giB && speculative.keepResumePoint,
            "a speculative shrink discarded the resume point");
    const auto demanded = policy.update(policySnapshot, 5101.0, true);
    require(demanded.targetBytes == giB && !demanded.keepResumePoint,
            "a waiting request could not reach the resume point");
    policySnapshot.pressure = MemoryPressure::Critical;
    auto criticalDirective = policy.update(policySnapshot, 2102.0, false);
    require(criticalDirective.reclaimEmptyKvExtents &&
                criticalDirective.evictAllUnpinnedPrefixes &&
                !criticalDirective.keepResumePoint,
            "critical pressure did not request aggressive reclaim");

    std::optional<uint64_t> elasticHostAvailable = 2ULL * 1024 * 1024 * 1024;
    MemoryGovernor hostGated(
        backend, backend.capabilities().recommendedMaxWorkingSetBytes,
        128ULL * 1024 * 1024,
        [&elasticHostAvailable] { return elasticHostAvailable; });
    kv::PageStorage hostGatedStorage(
        backend, hostGated.allocationAdmission(), kvLayout, 256, 128);
    if (hostGatedStorage.residentPages() != 128) {
        throw std::runtime_error(
            "elastic Q8 storage started with " +
            std::to_string(hostGatedStorage.residentPages()) +
            " resident blocks instead of 128");
    }
    elasticHostAvailable = 128ULL * 1024 * 1024;
    require(!hostGatedStorage.ensureResident(128) &&
                hostGatedStorage.residentPages() == 128,
            "host pressure did not reject the next KV extent transactionally");
    elasticHostAvailable = 4ULL * 1024 * 1024 * 1024;
    require(hostGatedStorage.ensureResident(128) &&
                hostGatedStorage.residentPages() == 256,
            "KV growth did not recover after host memory became available");

    MemoryGovernor governor(
        backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
    requireThrows<std::invalid_argument>(
        [&] { kv::PageStorage(backend, governor.allocationAdmission(), kvLayout, 192, 128); },
        "a pool of a part of an extent was accepted");
    requireThrows<std::invalid_argument>(
        [&] { kv::PageStorage(backend, governor.allocationAdmission(), kvLayout, 256, 64); },
        "an extent of a part of an alignment unit was accepted");

    metal::MetalBuffer table = backend.allocateBuffer(4 * sizeof(SplashKvPage));
    metal::MetalBuffer word = backend.allocateBuffer(sizeof(uint32_t));
    const uint64_t before = backend.memoryStats().allocatedBytes;
    kv::PageStorage storage(backend, governor.allocationAdmission(), kvLayout, 384, 128);
    const uint64_t extentBytes = 128 * kvLayout.bytesPerModelPage();
    require(uint64_t{storage.pageCount()} * storage.bytesPerPage() == 3 * extentBytes && storage.extentBytes() == extentBytes &&
                storage.actualAllocatedBytes() == extentBytes &&
                backend.memoryStats().allocatedBytes == before + extentBytes,
            "the runway extent was not allocated at exactly its size");
    require(storage.residentPages() == 128 && storage.isResident(127) && !storage.isResident(128),
            "initial Q8 runway residency is incorrect");
    const auto layer = storage.layer(15);
    require(layer.format == kv::Format::Int8 && layer.kv.extent_pages == 128 &&
                layer.kv.offset == 15 * 128 * kvLayout.bytesPerLayerPage(),
            "a layer's region does not follow the layers before it");

    const SplashKvPage runwayPage = storage.entry(5);
    require(runwayPage && (runwayPage & SPLASH_KV_PAGE_INDEX_MASK) == 5,
            "a page entry does not carry the page's index in its extent");
    requireThrows<std::logic_error>([&] { (void)storage.entry(200); },
                                    "an unbacked page received an entry");
    requireThrows<std::logic_error>([&] { (void)storage.spans(200); },
                                    "an unbacked page received host memory");
    requireSpansTileExtent(storage, 0);
    requireThrows<std::logic_error>(
        [&] { storage.writeEntries(std::array<uint32_t, 1>{200}, table); },
        "a table was written with an unbacked page");
    requireThrows<std::logic_error>(
        [&] { storage.writeEntries(std::array<uint32_t, 5>{0, 1, 2, 3, 4}, table); },
        "a table too small for its entries was written");

    const uint64_t generation = storage.generation();
    require(storage.ensureResident(200) && storage.generation() == generation + 1 &&
                storage.residentPages() == 256 &&
                storage.actualAllocatedBytes() == 2 * extentBytes &&
                backend.memoryStats().allocatedBytes == before + 2 * extentBytes,
            "growth did not add exactly one extent");
    storage.writeEntries(std::array<uint32_t, 4>{200, 5, 255, 128}, table);
    const auto *entries = static_cast<const SplashKvPage *>(table.contents());
    // Pages 200, 255 and 128 share the second extent, at indices 72, 127, 0.
    require(entries[0] == storage.entry(200) && (entries[0] & SPLASH_KV_PAGE_INDEX_MASK) == 72 &&
                entries[1] == runwayPage && entries[2] == entries[0] + 55 &&
                entries[3] == entries[0] - 72 && (runwayPage & ~uint64_t{SPLASH_KV_PAGE_INDEX_MASK}) !=
                                                     (entries[3] & ~uint64_t{SPLASH_KV_PAGE_INDEX_MASK}),
            "a page table does not hold the entries of its pages");

    // A command reaches extents through its tables without retaining them:
    // none is released while one is in flight.
    {
        const metal::ComputeDispatch kick{"residency_kick", {{0, word}}, {}, {1, 1, 1}, {1, 1, 1}};
        auto ticket = backend.submitAsync(kick);
        requireThrows<std::logic_error>(
            [&] { (void)storage.releaseBackingForPage(200); },
            "an extent was released while a command was in flight");
        require(storage.isResident(200) && storage.entry(200) == entries[0] &&
                    storage.generation() == generation + 1,
                "a refused release changed the extent");
        (void)ticket.wait();
    }
    require(storage.releaseBackingForPage(200) && !storage.isResident(200) &&
                storage.generation() == generation + 2 && storage.residentPages() == 128 &&
                backend.memoryStats().allocatedBytes == before + extentBytes,
            "a released extent did not return its memory at once");
    require(!storage.releaseBackingForPage(200), "an unbacked extent was released twice");
    require(storage.ensureResident(255) && storage.generation() == generation + 3 &&
                (storage.entry(255) & SPLASH_KV_PAGE_INDEX_MASK) == 127,
            "a released extent could not be allocated again");

    kv::PageStorage compactStorage(
        backend, governor.allocationAdmission(), compactLayout, 1024, 512);
    require(compactStorage.layer(9).kv.extent_pages == 512 &&
                compactStorage.layer(9).kv.offset ==
                    9 * 512 * compactLayout.bytesPerLayerPage() &&
                compactStorage.residentPages() == 512 &&
                compactStorage.actualAllocatedBytes() ==
                    512 * compactLayout.bytesPerModelPage(),
            "model-provided compact Q8 geometry was not honored");

    for (const auto layout : {bf16Layout, compactBf16Layout}) {
        const uint32_t extent = layout.minimumExtentPages();
        kv::PageStorage bf16(backend, governor.allocationAdmission(), layout,
                             2 * extent, extent);
        const auto bf16Layer = bf16.layer(layout.attentionLayers - 1);
        require(bf16Layer.format == kv::Format::BFloat16 &&
                    bf16Layer.kv.offset == (layout.attentionLayers - 1) * extent * 2 *
                                               layout.dataBytesPerLayerPage(),
                "BF16 regions hold quantization scales or misplace a layer");
        require(bf16.residentPages() == extent && !bf16.isResident(extent) &&
                    bf16.actualAllocatedBytes() == extent * layout.bytesPerModelPage(),
                "BF16 initial residency escaped its admitted extent");
        requireSpansTileExtent(bf16, 0);
        require(bf16.ensureResident(extent) && bf16.residentPages() == 2 * extent &&
                    bf16.actualAllocatedBytes() == uint64_t{bf16.pageCount()} * bf16.bytesPerPage(),
                "BF16 growth did not account for both extents");
        require(bf16.releaseBackingForPage(extent), "BF16 extent release failed");
        require(!bf16.isResident(extent) && bf16.ensureResident(extent),
                "BF16 extent could not be allocated again after release");
    }
    std::cout << "KV page storage tests passed\n";
}

}  // namespace

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "usage: q8_page_storage_test METALLIB\n";
        return EXIT_FAILURE;
    }
    try {
        run(argv[1]);
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "q8 page storage test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
