#pragma once

#include <cstddef>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace PtoTestCommon {

inline bool ReadFile(const std::string &path, size_t &fileSize, void *buffer,
                     size_t bufferSize) {
    struct stat status;
    if (stat(path.c_str(), &status) == -1 || !S_ISREG(status.st_mode))
        return false;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        return false;

    std::filebuf *fileBuffer = file.rdbuf();
    size_t size = fileBuffer->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0 || size > bufferSize)
        return false;
    fileBuffer->pubseekpos(0, std::ios::in);
    fileBuffer->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    return true;
}

inline bool WriteFile(const std::string &path, const void *buffer, size_t size) {
    if (buffer == nullptr)
        return false;

    int descriptor = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC,
                          S_IRUSR | S_IWRITE);
    if (descriptor < 0)
        return false;
    ssize_t written = write(descriptor, buffer, size);
    (void)close(descriptor);
    return written == static_cast<ssize_t>(size);
}

} // namespace PtoTestCommon
