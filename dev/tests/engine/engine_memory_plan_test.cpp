#include "engine/MemoryPlan.hpp"
#include "TestModel.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

DeviceCapabilities device(uint64_t workingSet = 12 * kGiB) {
  DeviceCapabilities result;
  result.deviceName = "test";
  result.appleGpuFamily = 9;
  result.macosMajor = 26;
  result.macosMinor = 4;
  result.physicalMemoryBytes = 16 * kGiB;
  result.recommendedMaxWorkingSetBytes = workingSet;
  result.maxBufferLengthBytes = 8 * kGiB;
  result.maxThreadgroupMemoryBytes = 32 * 1024;
  result.maxThreadgroupWidth = 1024;
  result.hasUnifiedMemory = true;
  return result;
}

ModelMemoryProfile model() {
  return test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB);
}

// The pages the budget holds for one request's KV beside its state cell.
uint64_t budgetPages(const EngineMemoryBreakdown &budget) {
  return (budget.dynamicBudgetBytes - budget.activeStateCellBytes) / budget.kvPageBytes;
}

void testUnifiedElasticBudget() {
  EngineMemoryPlan plan = requireEngineMemoryPlan(device(), model());
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.maximumBatchWidth == 4 &&
              budget.kvExtentPages == 128,
          "execution geometry did not reach memory planning");
  require(budget.fixedRuntimeBytes == model().fixedRuntimeBytes(),
          "active state was incorrectly precharged as fixed memory");
  require(budget.dynamicBudgetBytes ==
              budget.hardBudgetBytes - budget.fixedRuntimeBytes,
          "state and KV do not share one dynamic budget");
  require(budget.minimumDynamicBytes ==
                  budget.activeStateCellBytes + budget.kvExtentBytes &&
              budget.minimumRequiredBytes ==
                  budget.fixedRuntimeBytes + budget.minimumDynamicBytes,
          "minimum B1 plus one physical extent is incorrect");
  require(budget.kvCapacityPages % 128 == 0 && budget.kvCapacityPages >= 128 &&
              budget.kvCapacityPages == budgetPages(budget) - budgetPages(budget) % 128 &&
              budget.kvCapacityBytes ==
                  uint64_t{budget.kvCapacityPages} * budget.kvPageBytes &&
              budget.kvCapacityTokens == uint64_t{budget.kvCapacityPages} * 32,
          "the KV pool is not the budget's whole extents");
  require(plan.maximumContextTokens() ==
              std::min<uint64_t>(model().maximumContextTokens,
                                 budget.kvCapacityTokens -
                                     model::ExecutionLimits::speculativeScratchTokens),
          "advertised context exceeds elastic KV capacity");
  const std::string json = plan.toStatusJson();
  require(json.find("\"dynamic_budget_bytes\"") != std::string::npos &&
              json.find("\"kv_extent_pages\":128") != std::string::npos,
          "elastic state/KV budget is missing from memory status");
}

void testBf16BudgetAndStatus() {
  auto profile = model();
  profile.targetKvLayout.format = kv::Format::BFloat16;
  const auto bf16 = requireEngineMemoryPlan(device(), profile);
  const auto int8 = requireEngineMemoryPlan(device(), model());
  const auto &budget = bf16.breakdown();
  require(budget.kvPageBytes == profile.targetKvLayout.bytesPerModelPage() &&
              budget.kvPageBytes > int8.breakdown().kvPageBytes &&
              budget.kvExtentPages ==
                  profile.targetKvLayout.extentPagesFor(budgetPages(budget)) &&
              budget.kvCapacityPages % budget.kvExtentPages == 0,
          "BF16 planning did not use its payload size and its pool's extent size");
  require(budget.kvCapacityBytes <= budget.dynamicBudgetBytes &&
              budget.kvCapacityPages < int8.breakdown().kvCapacityPages,
          "BF16 virtual capacity exceeded the shared budget");
  const auto json = bf16.toStatusJson();
  require(json.find("\"kv_format\":\"bf16\"") != std::string::npos &&
              json.find("\"kv_scale_value_bytes\":0") != std::string::npos &&
              json.find("\"q8_page_bytes\"") == std::string::npos,
          "BF16 memory status reported INT8 scales or pages");
  const uint64_t minimum = budget.minimumRequiredBytes;
  require(!evaluateEngineMemoryPlan(device(), profile, minimum - 1).plan,
          "BF16 startup admitted less than its minimum resident footprint");
}

void testUserCeilingAndFailure() {
  EngineMemoryPlan automatic = requireEngineMemoryPlan(device(), model());
  const uint64_t ceiling =
      automatic.breakdown().minimumRequiredBytes + 64 * kMiB;
  EngineMemoryPlan limited =
      requireEngineMemoryPlan(device(), model(), ceiling);
  require(limited.breakdown().hardBudgetBytes == ceiling,
          "explicit memory ceiling was ignored");
  require(requireEngineMemoryPlan(device(), model(), 16 * kGiB)
                  .breakdown().hardBudgetBytes ==
              automatic.breakdown().hardBudgetBytes,
          "explicit memory ceiling overrode the safe working set");

  auto failed = evaluateEngineMemoryPlan(
      device(), model(), automatic.breakdown().minimumRequiredBytes - 1);
  require(!failed.plan &&
              failed.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "budget smaller than B1 plus one extent was accepted");
}

// The disk tier's KV pages stage through a ring of Metal memory, which the
// governor charges beside the weights. The plan sets it aside, so one lone
// request can still map every KV page the advertised context promises.
void testDiskTierKvStagingIsBudgeted() {
  const EngineMemoryPlan without = requireEngineMemoryPlan(device(), model());
  ModelMemoryProfile tiered = model();
  // 128 16 KiB-aligned page slots plus the copy table rounded up to 16 KiB.
  const uint64_t ring =
      128 * tiered.targetKvLayout.bytesPerModelPage() + 16 * 1024;
  tiered.footprint.kvStagingBytes = ring;
  const EngineMemoryPlan with = requireEngineMemoryPlan(device(), tiered);
  const auto &budget = with.breakdown();
  require(without.breakdown().fixedRuntimeBytes + ring +
                  budget.activeStateCellBytes + budget.kvCapacityBytes <=
              budget.hardBudgetBytes,
          "the advertised context cannot be mapped beside the KV staging ring");
  require(with.maximumContextTokens() < without.maximumContextTokens(),
          "a budget-limited context did not shrink by the KV staging ring");
  require(budget.kvStagingBytes == ring &&
              budget.fixedRuntimeBytes ==
                  without.breakdown().fixedRuntimeBytes + ring,
          "KV staging was not planned as fixed runtime memory");
  // A budget that fits everything but the ring is refused by the plan, not
  // by a warmup allocation.
  const auto tight = evaluateEngineMemoryPlan(
      device(), tiered, without.breakdown().minimumRequiredBytes);
  require(!tight.plan &&
              tight.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "a budget without room for the KV staging ring was accepted");
  const std::string staging = "\"kv_staging_bytes\":" + std::to_string(ring);
  require(with.toStatusJson().find(staging + ",\"fixed_runtime_bytes\"") !=
                  std::string::npos &&
              with.toStatusJson().find(staging + "}}") != std::string::npos &&
              budget.describe().find("disk tier KV staging: " +
                                     std::to_string(ring)) != std::string::npos,
          "KV staging is missing from the memory plan status");
  require(without.breakdown().kvStagingBytes == 0 &&
              without.toStatusJson().find("\"kv_staging_bytes\":0,") !=
                  std::string::npos,
          "a plan without the disk tier reported KV staging");
}

void testHardBudgetBoundaries() {
  require(EngineMemoryPolicy::hardBudgetBytes(12 * kGiB) == 11 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 8 * kGiB) ==
                  8 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 16 * kGiB) ==
                  11 * kGiB,
          "preflight ceiling disagrees with automatic or explicit policy");
  require(EngineMemoryPolicy::hardBudgetBytes(0) == 0 &&
              EngineMemoryPolicy::hardBudgetBytes(kGiB, 1) == 0,
          "insufficient working set underflowed the preflight ceiling");
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
  require(EngineMemoryPolicy::hardBudgetBytes(maximum, maximum) ==
              maximum - EngineMemoryPolicy::workingSetMarginBytes(maximum),
          "maximum working set overflowed the preflight ceiling");
}

// What memory holds below the plan's budget, where the host has less: the
// plan made there, within the configured limit, and nothing where one request
// does not fit.
void testContextTokensWithin() {
  const EngineMemoryPlan plan = requireEngineMemoryPlan(device(), model());
  const auto &budget = plan.breakdown();
  const uint64_t ceiling = budget.minimumRequiredBytes + 64 * kMiB;
  const EngineMemoryPlan limited = requireEngineMemoryPlan(device(), model(), ceiling);
  require(plan.contextTokensWithin(16 * kGiB) == plan.maximumContextTokens() &&
              plan.contextTokensWithin(ceiling) == limited.maximumContextTokens() &&
              limited.maximumContextTokens() < plan.maximumContextTokens() &&
              limited.contextTokensWithin(16 * kGiB) == limited.maximumContextTokens() &&
              !plan.contextTokensWithin(budget.minimumRequiredBytes - 1) &&
              !plan.contextTokensWithin(0),
          "the context memory holds is not the plan's within the host's memory");
}

void testModelProvidedKvGeometry() {
  ModelMemoryProfile compact = model();
  compact.name = "compact-test-model";
  compact.targetKvLayout = {10, 2, 256};
  EngineMemoryPlan plan = requireEngineMemoryPlan(device(), compact);
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.kvPageBytes == 332'800 &&
              budget.kvExtentPages ==
                  compact.targetKvLayout.extentPagesFor(budgetPages(budget)) &&
              budget.kvExtentPages % 256 == 0,
          "memory plan ignored model-provided Q8 geometry");
  require(plan.toStatusJson().find("\"attention_layers\":10") !=
              std::string::npos &&
              plan.toStatusJson().find("\"kv_heads\":2") !=
                  std::string::npos,
          "model-provided Q8 geometry is missing from status");
}

// The pool's extents leave the fewest of the budget's pages unused, the size
// nearest the 128 MiB target on a tie, and a budget below the smallest
// extent holds no pool.
void testExtentSizeFollowsThePool() {
  ModelMemoryProfile compact = model();
  compact.targetKvLayout = {10, 2, 256};
  const auto &reference = requireEngineMemoryPlan(device(), compact).breakdown();
  const auto planFor = [&](uint64_t pages) {
    return evaluateEngineMemoryPlan(
        device(), compact,
        reference.fixedRuntimeBytes + reference.activeStateCellBytes +
            pages * reference.kvPageBytes);
  };
  for (const auto [pages, extent] :
       std::array<std::array<uint64_t, 2>, 4>{{{10'240, 512}, {10'496, 256},
                                              {10'751, 256}, {511, 256}}}) {
    const auto result = planFor(pages);
    require(result.plan && result.plan->breakdown().kvExtentPages == extent &&
                result.plan->breakdown().kvCapacityPages == pages - pages % extent &&
                result.plan->breakdown().kvExtentBytes == extent * 332'800,
            "the extent size does not leave the fewest pool pages over");
  }
  const auto tooSmall = planFor(255);
  require(!tooSmall.plan && tooSmall.status.code == BudgetErrorCode::KvPoolDoesNotFit &&
              tooSmall.status.breakdown.minimumDynamicBytes ==
                  reference.activeStateCellBytes + 256 * 332'800,
          "a budget below the smallest extent was accepted");
}

} // namespace

void testDeviceValidationNamesTheMacosFloor() {
  require(!device().validationError(),
          "the reference device reported a validation error");
  DeviceCapabilities older = device();
  older.macosMinor = 3;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "macOS 26.3 was not refused with the macOS reason");
  // The operating system is checked before the device.
  older.appleGpuFamily = 8;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "an older macOS did not take precedence over the device's reason");
  require(older.macosVersion() == "26.3.0",
          "the macOS version string is not major.minor.patch");
  DeviceCapabilities unknown = device();
  unknown.macosMajor = 0;
  unknown.macosMinor = 0;
  require(unknown.validationError().value_or("") == "macos_26_4_required",
          "an unknown macOS version was accepted");
  DeviceCapabilities newer = device();
  newer.macosMajor = 27;
  newer.macosMinor = 0;
  require(!newer.validationError(), "a newer macOS major was refused");
}

void testDeviceValidationMessageNamesWhatTheMacHas() {
  require(!device().validationMessage(),
          "the reference device has a validation message");
  const std::string needs =
      "Splash needs Apple GPU family 9 or newer (M3 or later) on macOS 26.4 "
      "or newer; this Mac has ";
  DeviceCapabilities m2 = device();
  m2.deviceName = "Apple M2 Max";
  m2.appleGpuFamily = 8;
  m2.macosPatch = 1;
  require(m2.validationMessage().value_or("") ==
              needs + "Apple M2 Max (Apple GPU family 8) on macOS 26.4.1 "
                      "(apple_gpu_family_9_required)",
          "a family-8 GPU was not named against the family required");
  DeviceCapabilities older = device();
  older.macosMinor = 3;
  require(older.validationMessage().value_or("") ==
              needs + "test (Apple GPU family 9) on macOS 26.3.0 "
                      "(macos_26_4_required)",
          "an older macOS was not named against the macOS required");
}

int main() {
  try {
    testUnifiedElasticBudget();
    testBf16BudgetAndStatus();
    testUserCeilingAndFailure();
    testDiskTierKvStagingIsBudgeted();
    testHardBudgetBoundaries();
    testContextTokensWithin();
    testModelProvidedKvGeometry();
    testExtentSizeFollowsThePool();
    testDeviceValidationNamesTheMacosFloor();
    testDeviceValidationMessageNamesWhatTheMacHas();
    std::cout << "elastic memory plan tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "elastic memory plan tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
