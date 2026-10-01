#pragma once

#include "engine/KvCache.hpp"
#include "engine/KvPool.hpp"
#include "engine/StateCache.hpp"
#include "model/Model.hpp"
#include "model/SlotFile.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace splash::engine {

struct CacheLookup final {
  uint32_t kvBoundary = 0;
  std::optional<CompositeStateLease> state;
  // A matched block deeper than the state once held a reusable state that
  // is gone from both tiers.
  bool lostState = false;

  [[nodiscard]] uint32_t resumeBoundary() const noexcept {
    return state ? state->boundary() : 0;
  }
  [[nodiscard]] uint32_t junctionBoundary() const noexcept {
    return kvBoundary > resumeBoundary() ? kvBoundary : 0;
  }
};

class Cache;

class CacheProbe final {
public:
  [[nodiscard]] uint32_t cachedTokens() const noexcept { return cachedTokens_; }

private:
  friend class Cache;

  std::vector<uint64_t> blocks_;
  // Only pages examined by matchedBlocks can affect this probe's result.
  std::vector<uint32_t> checkedTokens_;
  std::vector<ImageSpan> images_;
  const Cache *owner_ = nullptr;
  size_t promptSize_ = 0;
  uint64_t kvGeneration_ = 0;
  uint32_t cachedTokens_ = 0;
};

struct CacheLookupSnapshot final {
  uint64_t lookups = 0;
  uint64_t kvHitTokens = 0;
  uint64_t stateHitTokens = 0;
  uint64_t lazyJunctions = 0;
  // Lookups that matched KV where a reusable state used to be.
  uint64_t lostStateMisses = 0;
};

struct KvTierSnapshot final {
  // The disk quota shared by KV pages and states, and its current use.
  uint64_t capacityBytes = 0;
  uint64_t usedBytes = 0;
  uint64_t readBytes = 0;
  uint64_t writtenBytes = 0;
  uint64_t demotions = 0;
  uint64_t demotionFailures = 0;
  // Demotions the tier had no room for; the leaf stayed and the requester
  // waited.
  uint64_t demotionsRefused = 0;
  uint64_t restores = 0;
  uint64_t restoreFailures = 0;
  // Pages whose demotion is in flight.
  uint32_t pendingPages = 0;
  uint32_t diskBlocks = 0;
  uint64_t diskBytes = 0;
};

struct CacheSnapshot final {
  KvPoolSnapshot pool;
  KvCache::Snapshot kvCache;
  StateCacheSnapshot stateCache;
  KvTierSnapshot kvTier;
  CacheLookupSnapshot lookup;
  uint32_t activeRequests = 0;
};

struct TokenAdmission final {
  KvPageAcquireFailure failure = KvPageAcquireFailure::None;
  uint32_t additionalPages = 0;
  uint32_t availablePages = 0;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept {
    return failure == KvPageAcquireFailure::None;
  }
};

struct PageTableView final {
  std::span<const uint32_t> pages;
  uint64_t revision = 0;
};

struct CacheReclaimResult final {
  bool madeProgress = false;
  uint64_t reclaimedBytes = 0;
  // Nothing was reclaimed, but a transfer in flight (a KV demotion, a KV
  // restore or the one state write) holds what the next reclaim needs.
  // Retry when it lands rather than treating the cache as empty.
  bool pending = false;
};

enum class CacheReclaimMode { KeepExtents, ReleaseExtents };

// What one step of room for a state's snapshot gave: a recycled state's
// buffers, which the snapshot takes as they are, or a released extent's
// bytes, which the snapshot has to allocate and which may be fewer than a
// state needs.
struct StateRoom final {
  bool made = false;
  bool extent = false;
  explicit operator bool() const noexcept { return made; }
};

// KV restores a request waits for before it can run.
enum class KvRestoreStatus : uint8_t { None, Pending, Failed };

// Owns active KV page leases, the content-addressed KV graph and cached
// composite states. Physical recurrent-state cells remain model-owned.
// A state in RAM always sits on a resident KV block: reclaim takes the state
// before the block's page, and endRequest, pollTransfers and freeDiskSpace
// leave such a block its page.
class Cache final {
public:
  // The disk budget is the quota the states' file shares with the KV tier;
  // the states' file can run on it without the tier.
  Cache(KvPool &pool, CacheNamespace cacheNamespace, model::KvTier *kvTier = nullptr,
        std::shared_ptr<const model::DiskBudget> diskBudget = nullptr);
  Cache(const Cache &) = delete;
  Cache &operator=(const Cache &) = delete;

  void setCompletionNotifier(std::function<void()> notifier) {
    completionNotifier_ = std::move(notifier);
  }
  // Consumes finished transfers: a written state or KV page frees its RAM, a
  // restored block becomes usable, and restores the tier had no room for
  // start.
  [[nodiscard]] bool pollTransfers();
  void discardState(uint64_t block, const CompositeState *state) {
    states_.invalidate(block, state);
  }
  void beginRequest(uint64_t requestId);
  void endRequest(uint64_t requestId);

  // Scheduling probe only: does not pin, touch recency, or count a hit.
  [[nodiscard]] CacheProbe
  probe(std::span<const uint32_t> prompt,
        std::span<const ImageSpan> images = {}) const;

  // Pin the usable prefix before potentially evicting for active allocations.
  // Accounting is separate: failed admission retries are not extra samples.
  [[nodiscard]] CacheLookup lookup(
      std::span<const uint32_t> prompt,
      std::span<const ImageSpan> images = {},
      const CacheProbe *probe = nullptr);
  void recordLookup(const CacheLookup &lookup);
  void promoteState(const CacheLookup &lookup, StateRestore &transfer);
  // Gives the request the matched chain up to its state. Disk-only blocks
  // get fresh pages and start their restores; the request waits on
  // kvRestoreStatus() before it runs. Pages are admitted like ensureTokens().
  [[nodiscard]] TokenAdmission restoreRequest(uint64_t requestId,
                                              const CacheLookup &lookup);
  [[nodiscard]] KvRestoreStatus kvRestoreStatus(uint64_t requestId) const;

  [[nodiscard]] TokenAdmission ensureTokens(uint64_t requestId,
                                            uint64_t tokenCount);
  [[nodiscard]] PageTableView pageTable(uint64_t requestId) const;

  // Canonicalizes every newly complete Page32 block. Duplicate content swaps
  // the request to the existing immutable page after the writer command has
  // completed; no active command ever aliases a writable page.
  [[nodiscard]] uint64_t publishCommittedBlocks(
      uint64_t requestId, std::span<const uint32_t> exactTokens,
      uint32_t committedTokens, std::span<const ImageSpan> images = {});
  [[nodiscard]] uint64_t blockAt(uint64_t requestId, uint32_t boundary) const;
  [[nodiscard]] bool reuseCompositeState(uint64_t kvBlock, bool checkpoint = false);
  // Reuses the state at this block in either tier, as reuseCompositeState()
  // does a RAM copy.
  [[nodiscard]] bool reuseStoredState(uint64_t kvBlock, bool checkpoint = false);
  void publishCompositeState(uint64_t kvBlock,
                             std::shared_ptr<const CompositeState> state,
                             bool checkpoint = false);
  // Publishes the state of the lane at this block straight to disk, for a
  // state no cache slot can hold: `write` starts the write from the lane.
  // False when the tier cannot take the state now; nothing is published then.
  [[nodiscard]] bool publishStateToDisk(uint64_t kvBlock, const StateWriter &write,
                                        bool checkpoint = false);
  // The request holding the handle is unfinished and its conversation
  // resumes from the state at this block: see StateCache::use.
  [[nodiscard]] StateUse useState(uint64_t kvBlock) { return states_.use(kvBlock); }
  [[nodiscard]] StateCheckpoint checkpointState(uint64_t kvBlock) const noexcept;
  // The state at this block has a RAM copy.
  [[nodiscard]] bool stateResident(uint64_t kvBlock) const noexcept;
  // False only while this exact disposable publication is pinned.
  bool retireCheckpointState(StateCheckpoint checkpoint) noexcept;

  // One cache reclaimer for memory growth and pressure warnings. Free pages
  // return first: empty extents, then the free pages scattered over the
  // others once they cover the extent that holds the fewest pages, whose
  // pages move to them (compactExtent). Only then is anything evicted:
  // disposable checkpoints first, then ordinary states and resident KV
  // leaves, which share one oldest-first access order. States that
  // unfinished requests use and the KV they restore through follow in their
  // own such order, once no transfer in flight can return what is needed
  // first. A chosen state keeps its disk copy when it has one, is written
  // when the tier admits it and dropped otherwise; its RAM is free when the
  // call returns. A chosen KV leaf frees its page at once when a disk copy
  // exists, is dropped when nothing depends on it, and is otherwise written
  // first: its page returns when the copy has landed, which ensureTokens()
  // reports as Pending so callers wait instead of evicting more. A full disk
  // quota replaces the oldest redundant copy of either kind, then the oldest
  // copy that is the only one. The only copies of states in use, and the KV
  // they restore through, make room only for a copy that is itself in use.
  // Active requests and pinned restores are never selected. A pass releases
  // every extent it empties; one that evicts everything moves pages only
  // once it has, so that nothing is copied and then evicted.
  // keepResumePoint stops short of the newest state
  // publication. A shrink that no request is waiting for gains the one cell
  // that publication holds and costs the next request a replay of its whole
  // prompt, because a hybrid model cannot resume from cached KV without the
  // recurrent state. Empty extents, older publications and state-free KV are
  // still reclaimed. keepRunway leaves one empty extent allocated, for the
  // next request.
  [[nodiscard]] uint64_t reclaimCache(uint64_t targetBytes, bool evictAll,
                                      bool keepResumePoint = false,
                                      bool keepRunway = false);
  // reclaimCache's stop rule: releasedBytes and the pages whose copies are
  // being written meet targetBytes. A pass with evictAll has no target.
  [[nodiscard]] bool reclaimMet(uint64_t releasedBytes, uint64_t targetBytes,
                                bool evictAll) const noexcept;
  // A KV demotion, a KV restore or the one state write is in flight, so
  // memory or quota returns by itself and its completion wakes the engine.
  [[nodiscard]] bool transfersInFlight() const noexcept;
  // One bounded reclaim step for an allocation retry: one empty extent, one
  // extent emptied of its pages, one state or one KV leaf, so a denied
  // allocation frees only what it needs. Progress is distinct from released
  // bytes because evicting a KV reference can make a page reusable without
  // emptying its extent, and the extent a step empties may be the runway
  // it keeps.
  [[nodiscard]] CacheReclaimResult reclaimOne(
      CacheReclaimMode mode = CacheReclaimMode::ReleaseExtents,
      bool keepResumePoint = false, bool keepRunway = false);
  // One step of room for the snapshot of a state to publish at forBlock,
  // taking nothing of a higher class than that publication; the disk tier
  // keeps a state it admits. An optional publication (checkpointsOnly) takes
  // one checkpoint, an ordinary one a checkpoint, else its oldest ordinary
  // state. A publication in use takes what reclaimOne() takes, in its order:
  // an extent of free pages, a checkpoint, the oldest ordinary state or KV,
  // and last the oldest state in use, which is never dropped for a busy
  // write slot: while the write in flight holds it, nothing goes. KV frees
  // memory only as a whole extent, so leaves go until a state goes or the
  // free pages fill an extent, which is released before the call returns:
  // the snapshot follows at once, since the lane's state moves on with its
  // next command. Like every release, that needs no command in flight; the
  // engine publishes a lane's states between commands.
  // After a step that gave an extent the caller steps again while the
  // snapshot does not fit; after one that gave a state it does not, so a
  // snapshot denied for another reason costs one state at most. While the
  // engine may not grow (`growth` false) an extent's bytes are of no use to
  // a snapshot: extents and KV stay, and a publication in use takes states
  // alone, like every other.
  [[nodiscard]] StateRoom reclaimOneState(bool checkpointsOnly = false, uint64_t forBlock = 0,
                                          bool growth = true);
  // Recycles one unpinned state, preferring checkpoints, for a lane that
  // takes the state's buffers. The lane is running work: a state in use goes
  // once no other state is left and no transfer in flight can return what is
  // needed first. A state the tier could take once the write in flight has
  // finished stays and is reported pending, as in reclaimOne.
  // evictableStates() are those it can take.
  [[nodiscard]] CacheReclaimResult reclaimStateForLane();
  [[nodiscard]] uint32_t evictableStates() const noexcept {
    return states_.evictable();
  }
  [[nodiscard]] CacheSnapshot snapshot() const;

private:
  [[nodiscard]] std::vector<uint64_t>
  matchedBlocks(std::span<const uint32_t> prompt,
                std::span<const ImageSpan> images) const;

  struct Request final {
    std::vector<uint32_t> pages;
    std::vector<uint64_t> cachedBlocks;
    uint64_t pageTableRevision = 0;
    uint32_t pendingRestores = 0;
    bool restoreFailed = false;
  };
  struct Demotion final {
    uint64_t block = 0;
    std::unique_ptr<model::KvTransfer> transfer;
  };
  struct Restore final {
    // Null until the tier has room for it.
    std::unique_ptr<model::KvTransfer> transfer;
    std::vector<uint64_t> waiters;
  };

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] const Request &request(uint64_t requestId) const;
  // Pages whose demotion is in flight; each keeps its page until the copy
  // lands.
  [[nodiscard]] uint32_t pendingPages() const noexcept {
    return static_cast<uint32_t>(demotions_.size());
  }
  // Pending, in every verdict below and in the results above, means the
  // same thing: a transfer in flight holds what this needs, and it comes
  // back when the transfer lands. Only transfersInFlight() may report it:
  // a caller told to wait for nothing would wait for ever.
  enum class LeafReclaim : uint8_t {
    Started,
    // The tier's room, the quota or the state write's staging buffer is held
    // by transfers in flight.
    Pending,
    // reclaimKvLeaf: the leaf stays for now and scans move on to the next
    // one, because a state on it is pinned, a disk subtree depends on it
    // and the tier has no room that a transfer in flight will free, or a
    // disk subtree below it cannot drop yet (active, in transfer, or a
    // state write in flight). demoteKv: the leaf cannot be written, and may
    // go without a copy, because the tier takes no writes, has no such
    // room, or making room took the states the leaf was kept for.
    Impossible,
  };

  // What one reclaim step gave up: a state, its RAM free at once, or a KV
  // leaf, whose memory returns with the extent it leaves empty.
  struct Victim final {
    bool kv = false;
    uint64_t reclaimedBytes = 0;
  };

  [[nodiscard]] TokenAdmission admitPages(uint32_t count,
                                          std::vector<uint32_t> &pages);
  // One eviction: checkpoints first, then the shared recency order, then
  // what is in use.
  [[nodiscard]] CacheReclaimResult evictOne(bool keepResumePoint);
  // Reclaims a chosen state; with waitForWrite only the write in flight may
  // keep it.
  [[nodiscard]] StateEviction reclaimState(uint64_t block, bool waitForWrite);
  // Gives up the oldest of one class's states and resident KV leaves. A
  // state whose write must wait for the one in flight stays when
  // waitForWrite, and is dropped otherwise; a KV leaf the tier cannot take
  // now stays. The other kind may still give. Nothing once all stay.
  [[nodiscard]] std::optional<Victim> reclaimOldest(bool inUse, bool keepResumePoint,
                                                    bool waitForWrite);
  // Oldest resident KV leaf after `after` whose state, if any, is not in RAM,
  // and whose KV a state in use needs exactly when inUse.
  [[nodiscard]] std::optional<CacheEvictionCandidate> oldestKvLeaf(uint64_t after,
                                                                   bool inUse) const;
  // Frees the RAM of one resident KV leaf: through its disk copy when it has
  // one, by demotion when a state on it or below it depends on it, by
  // erasure otherwise, with any disk copies below it. Pending when the tier
  // cannot take it right now. A leaf that a disk subtree depends on is
  // dropped only once a failed write has closed the tier, together with
  // that subtree.
  [[nodiscard]] LeafReclaim reclaimKvLeaf(uint64_t block);
  [[nodiscard]] LeafReclaim demoteKv(uint64_t block);
  // A failed write closes the tier; existing copies stay readable.
  [[nodiscard]] bool kvTierWritable() const noexcept { return tier_ && tier_->writable(); }
  // Only a state restores a disk-only chain, through its own block and every
  // block above.
  [[nodiscard]] bool kvNeededByState(uint64_t block) const {
    return states_.contains(block) || kv_.stateBelow(block);
  }
  // The same for a state in use: such KV is in use too.
  [[nodiscard]] bool kvNeededByStateInUse(uint64_t block) const {
    return (states_.inUse(block) && states_.contains(block)) || kv_.stateInUseBelow(block);
  }
  // Erases the disk-only subtree below a resident leaf and the states on it,
  // in use or not; false, erasing nothing, while a block of it is in transfer
  // or active (a lookup holding a state keeps its block active) or a state
  // write is in flight.
  [[nodiscard]] bool dropDiskSubtree(uint64_t block);
  // Nothing below a block whose read failed matches any more. Once none of
  // it is in transfer or active, it is erased with the states on it, in use
  // or not, and the poisoned block leaves with its last user.
  void dropPoisoned();
  // A slot for a new KV copy, replacing older copies while the quota is full;
  // inUse when a state in use needs the copy.
  [[nodiscard]] std::shared_ptr<model::KvDiskSlot> acquireDiskSlot(bool inUse);
  // Gives up one disk copy for a new copy, in use or not: the oldest
  // redundant one, KV or state, else the oldest that is the only copy, never
  // the KV of a state in RAM. The only copy of a state in use, and the KV it
  // restores through, go only for a copy in use, and after every other.
  // False when the disk holds nothing the new copy may displace.
  [[nodiscard]] bool freeDiskSpace(bool inUse);
  void startRestore(uint64_t block);
  [[nodiscard]] uint64_t pendingBytes() const noexcept;
  [[nodiscard]] uint64_t
  reclaimEmptyExtents(bool keepRunway,
                      uint32_t limit = std::numeric_limits<uint32_t>::max());
  // Returns one extent of free pages: an empty one, or the one compactExtent
  // empties. Zero when the free pages fill none.
  [[nodiscard]] uint64_t releaseExtent();
  // Empties one extent that still holds pages (KvPool::compactExtent); the
  // blocks and requests on its pages follow them. A page a transfer reads or
  // writes stays where it is until the transfer has landed. False when the
  // free pages cover no extent.
  [[nodiscard]] bool compactExtent();

  KvPool &pool_;
  model::KvTier *tier_;
  std::shared_ptr<const model::DiskBudget> diskBudget_;
  CacheRecency recency_;
  KvCache kv_;
  StateCache states_;
  DiskRoom makeRoom_;
  std::unordered_map<uint64_t, Request> requests_;
  std::vector<Demotion> demotions_;
  // Block IDs increase from parent to child. Restores start in that order so
  // cancellation can discard an unread suffix without stranding its parents.
  std::map<uint64_t, Restore> restores_;
  // Blocks whose read failed, until they have left.
  std::vector<uint64_t> poisoned_;
  KvTierSnapshot kvTier_;
  CacheLookupSnapshot lookup_;
  std::function<void()> completionNotifier_;
};

} // namespace splash::engine
