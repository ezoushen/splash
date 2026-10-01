#include "engine/FdTransport.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <fcntl.h>
#include <mutex>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace splash::engine {
namespace {

class RestoreFdFlags final {
public:
  RestoreFdFlags(int fd, int flags) : fd_(fd), flags_(flags) {}
  ~RestoreFdFlags() { static_cast<void>(fcntl(fd_, F_SETFL, flags_)); }

  RestoreFdFlags(const RestoreFdFlags &) = delete;
  RestoreFdFlags &operator=(const RestoreFdFlags &) = delete;

private:
  int fd_;
  int flags_;
};

[[noreturn]] void throwIo(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

int pollTimeout(const NativeRuntime &loop) {
  auto delay = loop.millisecondsUntilNextWakeup();
  int timeout = -1;
  if (delay) {
    if (*delay <= 0.0)
      timeout = 0;
    else if (*delay >= double(INT_MAX))
      timeout = INT_MAX;
    else
      timeout = static_cast<int>(std::ceil(*delay));
  }
  return timeout;
}

NativeProcessExit loopFailure(const NativeRuntime &loop) {
  return loop.engineHealthy() ? NativeProcessExit::ProtocolFailure
                              : NativeProcessExit::EngineFailure;
}

} // namespace

struct FdTransport::CompletionWake {
  int readFd = -1;
  int writeFd = -1;
  std::atomic<bool> controlPending{false};
  std::atomic<bool> shutdownRequested{false};

  CompletionWake() {
    int descriptors[2];
    if (pipe(descriptors) < 0)
      throwIo("pipe(completion wake)");
    readFd = descriptors[0];
    writeFd = descriptors[1];
    auto makeNonBlocking = [&](int fd) {
      int flags = fcntl(fd, F_GETFL);
      if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        int saved = errno;
        close(readFd);
        close(writeFd);
        readFd = -1;
        writeFd = -1;
        errno = saved;
        throwIo("fcntl(completion wake)");
      }
    };
    makeNonBlocking(readFd);
    makeNonBlocking(writeFd);
  }

  ~CompletionWake() {
    if (readFd >= 0)
      close(readFd);
    if (writeFd >= 0)
      close(writeFd);
  }

  void notify() const noexcept {
    constexpr uint8_t byte = 1;
    while (writeFd >= 0) {
      ssize_t written = write(writeFd, &byte, sizeof(byte));
      if (written > 0 ||
          (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))) {
        return;
      }
      if (written < 0 && errno == EINTR)
        continue;
      return;
    }
  }

  void notifyControl() noexcept {
    controlPending.store(true, std::memory_order_release);
    notify();
  }

  bool takeControl() noexcept {
    return controlPending.exchange(false, std::memory_order_acq_rel);
  }

  void drain() const noexcept {
    std::array<uint8_t, 64> bytes{};
    while (readFd >= 0) {
      ssize_t count = read(readFd, bytes.data(), bytes.size());
      if (count > 0)
        continue;
      if (count < 0 && errno == EINTR)
        continue;
      return;
    }
  }
};

// Reads the input on a thread of its own and queues it for the loop, which
// takes it between ticks. The bytes, then the input's end or a read error,
// reach the loop in the order they were read. Destruction stops the thread,
// whether it waits for input or for room in the queue, and joins it.
class FdTransport::InputReader final {
public:
  struct Input final {
    std::vector<uint8_t> bytes;
    // Set once the input has ended after these bytes: 0 at its end,
    // otherwise the error that ended reading.
    std::optional<int> end;
  };

  InputReader(int fd, size_t queueBytes, std::shared_ptr<CompletionWake> wake)
      : fd_(fd), queueBytes_(queueBytes), wake_(std::move(wake)) {
    int descriptors[2];
    if (pipe(descriptors) < 0)
      throwIo("pipe(input reader stop)");
    stopRead_ = descriptors[0];
    stopWrite_ = descriptors[1];
    try {
      thread_ = std::thread([this] { read(); });
    } catch (...) {
      close(stopRead_);
      close(stopWrite_);
      throw;
    }
  }

  ~InputReader() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    room_.notify_all();
    constexpr uint8_t byte = 1;
    while (write(stopWrite_, &byte, sizeof(byte)) < 0 && errno == EINTR) {
    }
    thread_.join();
    close(stopRead_);
    close(stopWrite_);
  }

  InputReader(const InputReader &) = delete;
  InputReader &operator=(const InputReader &) = delete;

  // Everything read since the last call, with the input's end once nothing
  // read before it is left.
  [[nodiscard]] Input take() {
    Input input;
    {
      std::lock_guard lock(mutex_);
      input.bytes.swap(queue_);
      input.end = end_;
    }
    room_.notify_all();
    return input;
  }

private:
  void read() noexcept {
    try {
      std::vector<uint8_t> chunk(64 * 1024);
      while (true) {
        {
          std::unique_lock lock(mutex_);
          room_.wait(lock, [&] { return stopping_ || queue_.size() < queueBytes_; });
          if (stopping_)
            return;
        }
        std::array<pollfd, 2> descriptors{pollfd{fd_, POLLIN, 0},
                                          pollfd{stopRead_, POLLIN, 0}};
        if (poll(descriptors.data(), descriptors.size(), -1) < 0) {
          if (errno == EINTR)
            continue;
          return finish(errno);
        }
        if (descriptors[1].revents)
          return;
        if (descriptors[0].revents & POLLNVAL)
          return finish(EBADF);
        if (!descriptors[0].revents)
          continue;
        const ssize_t count = ::read(fd_, chunk.data(), chunk.size());
        if (count > 0) {
          {
            std::lock_guard lock(mutex_);
            queue_.insert(queue_.end(), chunk.begin(), chunk.begin() + count);
          }
          wake_->notify();
          continue;
        }
        if (count == 0)
          return finish(0);
        if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
          return finish(errno);
      }
    } catch (const std::bad_alloc &) {
      finish(ENOMEM);
    }
  }

  void finish(int error) noexcept {
    {
      std::lock_guard lock(mutex_);
      end_ = error;
    }
    wake_->notify();
  }

  int fd_ = -1;
  size_t queueBytes_ = 0;
  std::shared_ptr<CompletionWake> wake_;
  int stopRead_ = -1;
  int stopWrite_ = -1;
  std::mutex mutex_;
  std::condition_variable room_;
  std::vector<uint8_t> queue_;
  std::optional<int> end_;
  bool stopping_ = false;
  std::thread thread_;
};

FdTransport::FdTransport(int inputFd, int outputFd, size_t inputQueueBytes)
    : inputFd_(inputFd), outputFd_(outputFd), inputQueueBytes_(inputQueueBytes),
      completionWake_(std::make_shared<CompletionWake>()) {
  if (inputFd_ < 0 || outputFd_ < 0) {
    throw std::invalid_argument("native transport requires valid fds");
  }
  if (!inputQueueBytes_)
    throw std::invalid_argument("native transport requires an input queue");
}

NativeRuntime::ByteSink FdTransport::outputSink() {
  return [this](std::span<const uint8_t> bytes) { writeAll(bytes); };
}

std::function<void()> FdTransport::controlNotifier() {
  std::shared_ptr<CompletionWake> wake = completionWake_;
  return [wake] { wake->notifyControl(); };
}

void FdTransport::setControlHandler(ControlHandler handler) {
  controlHandler_ = std::move(handler);
}

NativeProcessExit FdTransport::run(NativeRuntime &loop) {
  std::shared_ptr<CompletionWake> completionWake = completionWake_;
  loop.setCompletionNotifier([completionWake] { completionWake->notify(); });
  int originalFlags = fcntl(inputFd_, F_GETFL);
  if (originalFlags < 0)
    throwIo("fcntl(F_GETFL)");
  if (fcntl(inputFd_, F_SETFL, originalFlags | O_NONBLOCK) < 0) {
    throwIo("fcntl(F_SETFL)");
  }
  RestoreFdFlags restore(inputFd_, originalFlags);
  // Stopped and joined on every way out of run(), before the flags return.
  InputReader reader(inputFd_, inputQueueBytes_, completionWake);
  bool deferredControl = false;

  while (!loop.connectionMustClose()) {
    if (shutdownRequested())
      return NativeProcessExit::CleanEof;
    const InputReader::Input input = reader.take();
    if (!input.bytes.empty() && !loop.receive(input.bytes))
      return loopFailure(loop);
    if (input.end) {
      if (*input.end)
        return NativeProcessExit::IoFailure;
      return loop.finishInput() ? NativeProcessExit::CleanEof
                                : loopFailure(loop);
    }

    const auto inputRead = std::chrono::steady_clock::now();
    deferredControl = completionWake->takeControl() || deferredControl;
    if (deferredControl && !loop.commandInFlight()) {
      deferredControl = loop.runControl(controlHandler_);
    }

    const bool progressed = loop.tick();
    const double tickMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - inputRead).count();
    if (tickMilliseconds > maxTickMilliseconds_.load(std::memory_order_relaxed))
      maxTickMilliseconds_.store(tickMilliseconds, std::memory_order_relaxed);
    if (progressed)
      continue;
    if (loop.connectionMustClose())
      return loopFailure(loop);

    // The reader wakes the loop through the completion pipe as well.
    pollfd descriptor{completionWake->readFd, POLLIN, 0};
    int result;
    do {
      result = poll(&descriptor, 1, pollTimeout(loop));
    } while (result < 0 && errno == EINTR && !shutdownRequested());
    if (result < 0 && errno == EINTR)
      return NativeProcessExit::CleanEof;
    if (result < 0)
      return NativeProcessExit::IoFailure;
    if (descriptor.revents & POLLIN)
      completionWake->drain();
    // A zero result is a deadline or resource-retry wake-up; tick() at the
    // top of the next iteration performs the transition.
  }
  return loopFailure(loop);
}

bool FdTransport::shutdownRequested() const noexcept {
  return completionWake_->shutdownRequested.load(std::memory_order_acquire);
}

double FdTransport::maxTickMilliseconds() const noexcept {
  return maxTickMilliseconds_.load(std::memory_order_relaxed);
}

void FdTransport::requestShutdown() noexcept {
  completionWake_->shutdownRequested.store(true, std::memory_order_release);
  completionWake_->notify();
}

void FdTransport::writeAll(std::span<const uint8_t> bytes) const {
  size_t offset = 0;
  while (offset < bytes.size()) {
    ssize_t count =
        write(outputFd_, bytes.data() + offset, bytes.size() - offset);
    if (count > 0) {
      offset += static_cast<size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR)
      continue;
    if (count == 0) {
      throw std::runtime_error("native output accepted zero bytes");
    }
    throwIo("write(native output)");
  }
}

} // namespace splash::engine
