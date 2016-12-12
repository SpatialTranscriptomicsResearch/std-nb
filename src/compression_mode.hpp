#ifndef COMPRESSION_MODE_HPP
#define COMPRESSION_MODE_HPP

#include <string>

enum class CompressionMode {
  none,
  gzip,
  bzip2
};

std::string to_string(CompressionMode mode);

std::ostream &operator<<(std::ostream &os, CompressionMode mode);
std::istream &operator>>(std::istream &is, CompressionMode &mode);

#endif
