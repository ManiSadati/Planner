#pragma once

#include <cstddef>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace PtoTestCommon {

inline bool ReadFile(const std::string &path, size_t &size, void *buffer,
                     size_t capacity) {
  struct stat status;
  if (stat(path.c_str(), &status) == -1 || !S_ISREG(status.st_mode)) {
    return false;
  }
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  std::filebuf *fileBuffer = file.rdbuf();
  size_t fileSize = fileBuffer->pubseekoff(0, std::ios::end, std::ios::in);
  if (fileSize == 0 || fileSize > capacity) {
    return false;
  }
  fileBuffer->pubseekpos(0, std::ios::in);
  fileBuffer->sgetn(static_cast<char *>(buffer), fileSize);
  size = fileSize;
  return true;
}

inline bool WriteFile(const std::string &path, const void *buffer,
                      size_t size) {
  if (buffer == nullptr) {
    return false;
  }
  int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
  if (fd < 0) {
    return false;
  }
  ssize_t written = write(fd, buffer, size);
  (void)close(fd);
  return written == static_cast<ssize_t>(size);
}

} // namespace PtoTestCommon
