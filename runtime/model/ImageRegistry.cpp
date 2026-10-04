#include "model/ImageRegistry.hpp"
#include "StderrLine.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

// A key is a SHA-256 in lowercase hex, which also names its lock file.
constexpr size_t kKeyLength = 64;
// How long a process waits for a holder's answer: a holder answers from
// memory, so only a stopped or overloaded holder takes this long. The asker
// then goes on as if it did not hold the image, and passes over it until its
// next pass of loads or restores (ImageRegistry::newPass).
constexpr std::chrono::milliseconds kHolderTimeout{2000};
// How long one lookup of an image among all the holders may take.
constexpr std::chrono::milliseconds kLookupTimeout{5000};
// How often a process marks its holder file and socket as in use: files of
// the temporary directory that no one changed for days may be removed.
constexpr std::chrono::milliseconds kTouchInterval = std::chrono::hours(1);
// How often a process that waits for an image's lock tries it again.
constexpr std::chrono::milliseconds kLockPoll{10};
// How long a holder waits for a request on a connection it accepted, shorter
// than an asker waits for it, so that an asker that stops does not make the
// holder look stopped to others.
constexpr std::chrono::milliseconds kServeTimeout{1000};

timeval timeoutValue(std::chrono::microseconds timeout) {
  return {static_cast<time_t>(timeout.count() / 1000000), static_cast<suseconds_t>(timeout.count() % 1000000)};
}

// Sets how long the socket's sends and receives wait.
bool setTimeout(int socket, std::chrono::microseconds timeout) {
  const timeval value = timeoutValue(timeout);
  return setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &value, sizeof(value)) == 0 &&
         setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &value, sizeof(value)) == 0;
}

[[noreturn]] void fail(const std::string &operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

// A descriptor this object closes.
class Descriptor final {
public:
  Descriptor() = default;
  explicit Descriptor(int descriptor) noexcept : descriptor_(descriptor) {}
  Descriptor(Descriptor &&other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}
  Descriptor &operator=(Descriptor &&other) noexcept {
    if (this != &other) {
      reset();
      descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
  }
  ~Descriptor() { reset(); }
  [[nodiscard]] int get() const noexcept { return descriptor_; }
  [[nodiscard]] explicit operator bool() const noexcept { return descriptor_ >= 0; }
  int release() noexcept { return std::exchange(descriptor_, -1); }
  void reset() noexcept {
    if (descriptor_ >= 0) close(descriptor_);
    descriptor_ = -1;
  }

private:
  int descriptor_ = -1;
};

void closeOnExec(int descriptor) {
  if (fcntl(descriptor, F_SETFD, FD_CLOEXEC) == -1) fail("cannot set close-on-exec on an image");
}

uint64_t pageRounded(uint64_t bytes) {
  const uint64_t page = static_cast<uint64_t>(getpagesize());
  return (bytes + page - 1) / page * page;
}

std::string randomHex(size_t bytes) {
  std::vector<uint8_t> random(bytes);
  arc4random_buf(random.data(), random.size());
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : random) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

// An image's shared memory as this process maps it, with its descriptor,
// which it can hand to another process. The buffer over it holds it.
struct Region final {
  Descriptor descriptor;
  void *address = nullptr;
  uint64_t bytes = 0;
  ~Region() {
    if (address) munmap(address, pageRounded(bytes));
  }
};

// Maps descriptor's shared memory, which holds an image of bytes.
std::shared_ptr<Region> mapRegion(Descriptor descriptor, uint64_t bytes, bool writable) {
  auto region = std::make_shared<Region>();
  void *address = mmap(nullptr, pageRounded(bytes), writable ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED,
                       descriptor.get(), 0);
  if (address == MAP_FAILED) fail("cannot map a shared weight image");
  region->descriptor = std::move(descriptor);
  region->address = address;
  region->bytes = bytes;
  return region;
}

// New shared memory for an image of bytes, which has no name: its name is
// removed at once.
std::shared_ptr<Region> createRegion(uint64_t bytes) {
  Descriptor descriptor;
  for (int attempt = 0; !descriptor; ++attempt) {
    const std::string name = "/splash-" + randomHex(8);
    descriptor = Descriptor(shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600));
    if (!descriptor && (errno != EEXIST || attempt == 8)) fail("cannot create a shared weight image");
    if (descriptor) shm_unlink(name.c_str());
  }
  closeOnExec(descriptor.get());
  if (ftruncate(descriptor.get(), static_cast<off_t>(pageRounded(bytes))) == -1)
    fail("cannot size a shared weight image");
  return mapRegion(std::move(descriptor), bytes, true);
}

// The memory a buffer is made over, which holds region until Metal lets the
// buffer go.
metal::MappedMemory mappedMemory(std::shared_ptr<Region> region) {
  void *address = region->address;
  const uint64_t bytes = region->bytes;
  return {address, bytes, [region = std::move(region)]() mutable { region.reset(); }};
}

sockaddr_un socketAddress(const std::filesystem::path &path) {
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  const std::string text = path.string();
  if (text.size() >= sizeof(address.sun_path))
    throw std::runtime_error("image registry path is too long for a socket: " + text);
  std::memcpy(address.sun_path, text.c_str(), text.size() + 1);
  return address;
}

Descriptor newSocket(std::chrono::microseconds timeout) {
  Descriptor result(socket(AF_UNIX, SOCK_STREAM, 0));
  if (!result) fail("cannot create an image registry socket");
  closeOnExec(result.get());
  const int on = 1;
  if (setsockopt(result.get(), SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on)) == -1 || !setTimeout(result.get(), timeout))
    fail("cannot configure an image registry socket");
  return result;
}

bool sendAll(int socket, const void *data, size_t bytes) {
  const auto *at = static_cast<const char *>(data);
  while (bytes) {
    const ssize_t sent = send(socket, at, bytes, 0);
    if (sent < 0 && errno == EINTR) continue;
    if (sent <= 0) return false;
    at += sent;
    bytes -= static_cast<size_t>(sent);
  }
  return true;
}

bool receiveAll(int socket, void *data, size_t bytes) {
  auto *at = static_cast<char *>(data);
  while (bytes) {
    const ssize_t received = recv(socket, at, bytes, 0);
    if (received < 0 && errno == EINTR) continue;
    if (received <= 0) return false;
    at += received;
    bytes -= static_cast<size_t>(received);
  }
  return true;
}

// The answer to a request: the image's length, zero when it is not held,
// and with a length its descriptor.
bool sendAnswer(int socket, uint64_t bytes, int descriptor) {
  iovec data{&bytes, sizeof(bytes)};
  msghdr message{};
  message.msg_iov = &data;
  message.msg_iovlen = 1;
  alignas(cmsghdr) std::array<char, CMSG_SPACE(sizeof(int))> control{};
  if (descriptor >= 0) {
    message.msg_control = control.data();
    message.msg_controllen = control.size();
    cmsghdr *header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(header), &descriptor, sizeof(int));
  }
  ssize_t sent;
  do sent = sendmsg(socket, &message, 0);
  while (sent < 0 && errno == EINTR);
  return sent == static_cast<ssize_t>(sizeof(bytes));
}

// The answer's length and descriptor; an empty descriptor when the holder
// does not hold the image or the answer is malformed.
std::pair<uint64_t, Descriptor> receiveAnswer(int socket) {
  uint64_t bytes = 0;
  iovec data{&bytes, sizeof(bytes)};
  msghdr message{};
  message.msg_iov = &data;
  message.msg_iovlen = 1;
  alignas(cmsghdr) std::array<char, CMSG_SPACE(sizeof(int))> control{};
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  ssize_t received;
  do received = recvmsg(socket, &message, 0);
  while (received < 0 && errno == EINTR);
  Descriptor descriptor;
  for (cmsghdr *header = received > 0 ? CMSG_FIRSTHDR(&message) : nullptr; header;
       header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS) continue;
    const size_t count = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    for (size_t i = 0; i < count; ++i) {
      int passed;
      std::memcpy(&passed, CMSG_DATA(header) + i * sizeof(int), sizeof(int));
      if (!descriptor) descriptor = Descriptor(passed);
      else close(passed);
    }
  }
  if (received != static_cast<ssize_t>(sizeof(bytes)) || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)))
    return {0, Descriptor()};
  return {bytes, std::move(descriptor)};
}

// Holds the lock of the file at path until destroyed, or none when another
// process held it for all of timeout. The file is created if it is missing.
class FileLock final {
public:
  FileLock(const std::filesystem::path &path, std::chrono::milliseconds timeout)
      : descriptor_(open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600)) {
    if (!descriptor_) fail("cannot open the image lock " + path.string());
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (flock(descriptor_.get(), LOCK_EX | LOCK_NB) == -1) {
      if (errno == EINTR) continue;
      if (errno != EWOULDBLOCK) fail("cannot lock the image lock " + path.string());
      if (std::chrono::steady_clock::now() >= deadline) {
        descriptor_.reset();
        return;
      }
      std::this_thread::sleep_for(kLockPoll);
    }
  }
  [[nodiscard]] explicit operator bool() const noexcept { return bool(descriptor_); }

private:
  Descriptor descriptor_;
};

void requirePrivateDirectory(const std::filesystem::path &directory) {
  if (mkdir(directory.c_str(), 0700) == -1 && errno != EEXIST)
    fail("cannot create the image registry " + directory.string());
  struct stat state{};
  if (lstat(directory.c_str(), &state) == -1) fail("cannot inspect the image registry " + directory.string());
  if (!S_ISDIR(state.st_mode) || state.st_uid != geteuid() || (state.st_mode & 077))
    throw std::runtime_error("refusing the image registry " + directory.string() +
                             ": it must be a directory of this user's that no one else may use");
}

} // namespace

struct ImageRegistry::Impl {
  std::filesystem::path directory;
  std::chrono::milliseconds writeWait{};
  // This process's name in the registry: its holder file and socket.
  std::string name;
  Descriptor holder;
  Descriptor listener;
  std::array<Descriptor, 2> wake;
  std::thread server;
  std::mutex mutex;
  // The holders that did not answer in time during this pass, which it
  // passes over. Only the thread that acquires uses it.
  std::set<std::string> unanswered;
  // The images this process holds, by key: a live region can be served.
  std::map<std::string, std::vector<std::weak_ptr<Region>>, std::less<>> held;

  [[nodiscard]] std::filesystem::path holderPath(std::string_view holder) const {
    return directory / (std::string(holder) + ".holder");
  }
  [[nodiscard]] std::filesystem::path socketPath(std::string_view holder) const {
    return directory / (std::string(holder) + ".socket");
  }
  [[nodiscard]] std::filesystem::path stagedPath(std::string_view holder) const {
    return directory / (std::string(holder) + ".new");
  }

  // Whether the process of that name has ended. A process holds the lock of
  // its holder file for its lifetime, from before it binds its socket, when
  // the file is staged under .new, and through its rename to .holder; a file
  // of its that this process can lock, or none, means it has ended. The
  // staged name is tried first, as the rename only moves it to the other.
  [[nodiscard]] bool ended(std::string_view other) const {
    for (const std::filesystem::path &path : {stagedPath(other), holderPath(other)}) {
      Descriptor file(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
      if (!file && errno == ENOENT) continue;
      return file && flock(file.get(), LOCK_EX | LOCK_NB) == 0;
    }
    return true;
  }

  // A region this process holds for key, if any.
  std::shared_ptr<Region> heldRegion(std::string_view key) {
    std::lock_guard lock(mutex);
    const auto found = held.find(key);
    if (found == held.end()) return nullptr;
    for (const auto &weak : found->second)
      if (auto region = weak.lock()) return region;
    return nullptr;
  }

  void hold(const std::string &key, const std::shared_ptr<Region> &region) {
    std::lock_guard lock(mutex);
    auto &regions = held[key];
    std::erase_if(regions, [](const auto &weak) { return weak.expired(); });
    regions.push_back(region);
  }

  // Another mapping of the image under key that this process holds.
  std::shared_ptr<Region> heldMapping(std::string_view key, uint64_t bytes) {
    const auto region = heldRegion(key);
    if (!region || region->bytes != bytes) return nullptr;
    Descriptor copy(fcntl(region->descriptor.get(), F_DUPFD_CLOEXEC, 0));
    if (!copy) fail("cannot duplicate a shared weight image");
    return mapRegion(std::move(copy), bytes, false);
  }

  // The image under key from the holder of that name, or null, waiting up
  // to timeout for its answer; a holder that does not answer in time is
  // passed over until the next pass.
  std::shared_ptr<Region> askHolder(const std::string &holder, const std::string &key, uint64_t bytes,
                                    std::chrono::microseconds timeout) {
    Descriptor socket = newSocket(timeout);
    const sockaddr_un address = socketAddress(socketPath(holder));
    if (connect(socket.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) == -1) return nullptr;
    errno = 0;
    std::pair<uint64_t, Descriptor> answer;
    if (sendAll(socket.get(), key.data(), key.size())) answer = receiveAnswer(socket.get());
    if (errno == EAGAIN || errno == EWOULDBLOCK) unanswered.insert(holder);
    auto &[length, descriptor] = answer;
    if (!descriptor || length != bytes) return nullptr;
    struct stat state{};
    if (fstat(descriptor.get(), &state) == -1 || uint64_t(state.st_size) != pageRounded(bytes)) return nullptr;
    closeOnExec(descriptor.get());
    return mapRegion(std::move(descriptor), bytes, false);
  }

  // The image under key from a live holder, or null. Only the files of a
  // process that has ended are removed (ended): a socket may refuse a live
  // holder too (a full backlog). That includes a socket or a staged holder
  // file a process left when it ended while it joined.
  std::shared_ptr<Region> askHolders(const std::string &key, uint64_t bytes) {
    const auto deadline = std::chrono::steady_clock::now() + kLookupTimeout;
    std::error_code error;
    std::set<std::string> names;
    std::set<std::string> holders;
    for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
      const std::filesystem::path &path = entry.path();
      const std::string other = path.stem().string();
      if (other == name) continue;
      if (path.extension() == ".holder") holders.insert(other);
      if (path.extension() == ".holder" || path.extension() == ".socket" || path.extension() == ".new")
        names.insert(other);
    }
    if (error) throw std::system_error(error, "cannot list the image registry " + directory.string());
    for (const std::string &other : names) {
      if (ended(other)) {
        unlink(socketPath(other).c_str());
        unlink(holderPath(other).c_str());
        unlink(stagedPath(other).c_str());
        continue;
      }
      if (!holders.contains(other) || unanswered.contains(other)) continue;
      const auto left =
          std::chrono::duration_cast<std::chrono::microseconds>(deadline - std::chrono::steady_clock::now());
      if (left <= std::chrono::microseconds::zero()) break;
      if (auto region = askHolder(other, key, bytes, std::min<std::chrono::microseconds>(left, kHolderTimeout)))
        return region;
    }
    return nullptr;
  }

  // Answers one request on socket.
  void answer(int socket) {
    std::string key(kKeyLength, '\0');
    if (!receiveAll(socket, key.data(), key.size())) return;
    const auto region = heldRegion(key);
    if (region) sendAnswer(socket, region->bytes, region->descriptor.get());
    else sendAnswer(socket, 0, -1);
  }

  // Marks this process's holder file and socket as in use, so that a
  // cleaner of temporary files does not take them for old ones.
  void touch() const {
    futimens(holder.get(), nullptr);
    utimensat(AT_FDCWD, socketPath(name).c_str(), nullptr, AT_SYMLINK_NOFOLLOW);
  }

  void serve() {
    auto touched = std::chrono::steady_clock::now();
    for (;;) {
      const auto now = std::chrono::steady_clock::now();
      if (now - touched >= kTouchInterval) {
        touch();
        touched = now;
      }
      const auto untilTouch =
          std::chrono::duration_cast<std::chrono::milliseconds>(touched + kTouchInterval - now).count();
      std::array<pollfd, 2> waiting{{{listener.get(), POLLIN, 0}, {wake[0].get(), POLLIN, 0}}};
      if (poll(waiting.data(), waiting.size(), static_cast<int>(std::max<int64_t>(untilTouch, 1))) == -1) {
        if (errno == EINTR) continue;
        return;
      }
      if (waiting[1].revents) return;
      if (!(waiting[0].revents & POLLIN)) continue;
      Descriptor client(accept(listener.get(), nullptr, nullptr));
      if (!client) continue;
      const int on = 1;
      setsockopt(client.get(), SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
      setTimeout(client.get(), kServeTimeout);
      answer(client.get());
    }
  }
};

ImageRegistry::ImageRegistry(const std::filesystem::path &directory, std::chrono::milliseconds writeWait)
    : impl_(std::make_unique<Impl>()) {
  Impl &registry = *impl_;
  registry.directory = directory;
  registry.writeWait = writeWait;
  requirePrivateDirectory(directory);
  registry.name = std::to_string(getpid()) + "-" + randomHex(4);
  // The holder file is created locked, under a staged name, before the
  // socket is bound, so no process takes either for an ended process's
  // (Impl::ended); it is renamed to its name once the socket listens.
  const std::filesystem::path staged = registry.stagedPath(registry.name);
  const std::filesystem::path socketPath = registry.socketPath(registry.name);
  const sockaddr_un address = socketAddress(socketPath);
  registry.holder =
      Descriptor(open(staged.c_str(), O_RDWR | O_CREAT | O_EXCL | O_EXLOCK | O_CLOEXEC | O_NOFOLLOW, 0600));
  if (!registry.holder) fail("cannot create the image registry holder " + staged.string());
  try {
    registry.listener = newSocket(kServeTimeout);
    if (bind(registry.listener.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) == -1)
      fail("cannot bind the image registry socket " + socketPath.string());
    if (listen(registry.listener.get(), SOMAXCONN) == -1) fail("cannot listen on the image registry socket");
    if (rename(staged.c_str(), registry.holderPath(registry.name).c_str()) == -1)
      fail("cannot publish the image registry holder");
    int wake[2];
    if (pipe(wake) == -1) fail("cannot create the image registry's wake pipe");
    registry.wake = {Descriptor(wake[0]), Descriptor(wake[1])};
    closeOnExec(wake[0]);
    closeOnExec(wake[1]);
    registry.server = std::thread([&registry] { registry.serve(); });
  } catch (...) {
    unlink(registry.holderPath(registry.name).c_str());
    unlink(socketPath.c_str());
    unlink(staged.c_str());
    throw;
  }
}

ImageRegistry::~ImageRegistry() {
  Impl &registry = *impl_;
  const char stop = 0;
  while (write(registry.wake[1].get(), &stop, 1) == -1 && errno == EINTR) {
  }
  registry.server.join();
  unlink(registry.socketPath(registry.name).c_str());
  unlink(registry.holderPath(registry.name).c_str());
}

void ImageRegistry::newPass() { impl_->unanswered.clear(); }

bool ImageRegistry::acquire(const std::string &key, uint64_t bytes,
                            const std::function<void(metal::MappedMemory)> &attach,
                            const std::function<void()> &write) {
  if (key.size() != kKeyLength ||
      !std::all_of(key.begin(), key.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
    throw std::invalid_argument("a shared image key is a SHA-256 in lowercase hex");
  if (!bytes) throw std::invalid_argument("a shared image is empty");
  Impl &registry = *impl_;
  const auto find = [&] {
    auto region = registry.heldMapping(key, bytes);
    return region ? region : registry.askHolders(key, bytes);
  };
  std::shared_ptr<Region> region = find();
  if (!region) {
    // One writer per key: the others wait here, then find its image. A
    // writer that holds the lock too long, as a stopped one does, is not
    // waited for: this process writes a copy of its own, which it does not
    // serve, as only an image written under its lock is served.
    const FileLock writing(registry.directory / (key + ".lock"), registry.writeWait);
    region = find();
    if (!region) {
      region = createRegion(bytes);
      attach(mappedMemory(region));
      write();
      // Written, it is read-only here as in every process it is served to.
      if (mprotect(region->address, pageRounded(bytes), PROT_READ) == -1)
        fail("cannot make a shared weight image read-only");
      if (writing)
        registry.hold(key, region);
      else
        writeStderrLine("A weight image another process was still writing was written again, unshared (" +
                        key.substr(0, 16) + ")");
      return true;
    }
  }
  attach(mappedMemory(region));
  registry.hold(key, region);
  return false;
}

} // namespace splash::model
