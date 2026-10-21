#pragma once

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

namespace splash::model {

// Creates directory 0700, or takes the one that exists, only if it is this
// user's own: a directory, not a symbolic link, of this user, which no one
// else may read, write or search. The files shared weights keep there (ImageRegistry,
// TensorDigests) are then this user's alone.
inline void requirePrivateDirectory(const std::filesystem::path &directory) {
  if (mkdir(directory.c_str(), 0700) == -1 && errno != EEXIST)
    throw std::system_error(errno, std::generic_category(), "cannot create " + directory.string());
  struct stat state{};
  if (lstat(directory.c_str(), &state) == -1)
    throw std::system_error(errno, std::generic_category(), "cannot inspect " + directory.string());
  if (!S_ISDIR(state.st_mode) || state.st_uid != geteuid() || (state.st_mode & 077))
    throw std::runtime_error("refusing " + directory.string() +
                             ": it must be a directory of this user's that no one else may use");
}

} // namespace splash::model
