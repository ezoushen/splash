#include "model/KvPageTier.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

namespace splash::model {

// A demotion or a restore of one page, until the tier retires its IO.
struct KvPageTier::Transfer final {
  bool demotion = false;
  std::shared_ptr<SlotFile::Operation> io;
  bool ready = false;
  bool success = false;
};

class KvPageTier::Ticket final : public KvTransfer {
public:
  explicit Ticket(std::shared_ptr<Transfer> transfer) : transfer_(std::move(transfer)) {}
  bool ready() const noexcept override { return transfer_->ready; }
  bool finish() override { return transfer_->success; }

private:
  std::shared_ptr<Transfer> transfer_;
};

namespace {
std::shared_ptr<KvPageTier::DiskSlot> diskSlot(const std::shared_ptr<KvDiskSlot> &slot) {
  auto disk = std::dynamic_pointer_cast<KvPageTier::DiskSlot>(slot);
  if (!disk) throw std::invalid_argument("KV disk slot belongs to another tier");
  return disk;
}
} // namespace

uint64_t KvPageTier::slotBytesFor(const kv::PageStorage &pages) noexcept {
  const uint64_t unit = SlotFile::kAlignmentBytes;
  return (pages.bytesPerPage() + unit - 1) / unit * unit;
}

KvPageTier::KvPageTier(kv::PageStorage &pages, std::shared_ptr<SlotFile> file,
                       uint32_t transfers)
    : pages_(pages), file_(std::move(file)), slotBytes_(slotBytesFor(pages)),
      transferLimit_(transfers) {
  if (!file_ || file_->slotBytes() != slotBytes_)
    throw std::invalid_argument("KV slot file does not match the page size");
  if (!transferLimit_)
    throw std::invalid_argument("KV tier needs room for one transfer");
  demotionLimit_ = std::max<uint32_t>(1, transferLimit_ / 2);
  restoreLimit_ = std::max<uint32_t>(1, transferLimit_ - transferLimit_ / 4);
  // Tracking a submitted IO must not allocate: the tier drains every IO it
  // tracks before it goes, and an IO it lost would outlive the page.
  inFlight_.reserve(transferLimit_);
}

// A transfer in flight reads or writes its page in place, and the cache that
// held the page for it is gone before the tier is: the IO worker must be done
// with every page first. Nothing will read the slots a cancelled write leaves
// behind, because the tier that maps pages to them is going away too.
KvPageTier::~KvPageTier() {
  for (const std::shared_ptr<Transfer> &transfer : inFlight_)
    transfer->io->drain();
}

uint64_t KvPageTier::slotBytes() const noexcept { return slotBytes_; }

bool KvPageTier::writable() const noexcept { return file_->writable(); }

bool KvPageTier::canDemote() const noexcept {
  return demotions_ < demotionLimit_ && inFlight_.size() < transferLimit_ && writable();
}

std::shared_ptr<KvDiskSlot> KvPageTier::acquireSlot() {
  auto slot = file_->acquire();
  if (!slot) return {};
  return std::make_shared<DiskSlot>(std::move(slot));
}

std::unique_ptr<KvTransfer> KvPageTier::demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot,
                                               std::function<void()> completion) {
  auto disk = diskSlot(slot);
  if (!canDemote())
    return {};
  const std::vector<std::span<std::byte>> bytes = pages_.spans(page);
  auto transfer = std::make_shared<Transfer>();
  transfer->demotion = true;
  auto ticket = std::make_unique<Ticket>(transfer);
  try {
    transfer->io = file_->write(disk->slot, {bytes.begin(), bytes.end()}, std::move(completion));
  } catch (const std::logic_error &) {
    // A write that failed since canDemote() has closed the file, which
    // refuses this one as canDemote() does from now on.
    if (writable()) throw;
    return {};
  }
  ++demotions_;
  inFlight_.push_back(std::move(transfer));
  return ticket;
}

std::unique_ptr<KvTransfer> KvPageTier::restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
                                                std::function<void()> completion) {
  auto disk = diskSlot(slot);
  if (restores_ >= restoreLimit_ || inFlight_.size() >= transferLimit_)
    return {};
  auto transfer = std::make_shared<Transfer>();
  auto ticket = std::make_unique<Ticket>(transfer);
  transfer->io = file_->read(disk->slot, pages_.spans(page), std::move(completion));
  ++restores_;
  inFlight_.push_back(std::move(transfer));
  return ticket;
}

void KvPageTier::poll() {
  for (size_t index = 0; index < inFlight_.size();) {
    if (!inFlight_[index]->io->ready()) {
      ++index;
      continue;
    }
    const std::shared_ptr<Transfer> transfer = std::move(inFlight_[index]);
    inFlight_.erase(inFlight_.begin() + static_cast<std::ptrdiff_t>(index));
    --(transfer->demotion ? demotions_ : restores_);
    transfer->success = transfer->io->wait();
    transfer->io.reset();
    transfer->ready = true;
  }
}

} // namespace splash::model
