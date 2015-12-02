/* =====================================================================================
 * Copyright (c) 2012, Jonas Maaskola
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =====================================================================================
 *
 *       Filename:  io.hpp
 *
 *    Description:  Routines for reading and writing possibly compressed files
 *
 *        Created:  Thu May 31 06:47:48 2012 +0200
 *
 *         Author:  Jonas Maaskola <jonas@maaskola.de>
 *
 * =====================================================================================
 */

#ifndef COMPRESSION_HPP
#define COMPRESSION_HPP

#include <fstream>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filtering_stream.hpp>

namespace Exception {
namespace File {
struct Existence : public std::runtime_error {
  Existence(const std::string& path)
      : std::runtime_error("Error: file does not exist: '" + path + "'."){};
};
struct NoRegularFile : public std::runtime_error {
  NoRegularFile(const std::string& path)
      : std::runtime_error("Error: not a regular file: '" + path + "'."){};
};
struct Access : public std::runtime_error {
  Access(const std::string& path)
      : std::runtime_error("Error: file access failed: '" + path + "'."){};
};
struct Reading : public std::runtime_error {
  Reading(const std::string& path)
      : std::runtime_error("Error: reading from file failed: '" + path +
                           "'."){};
};
struct Parsing : public std::runtime_error {
  Parsing(const std::string& path)
      : std::runtime_error("Error: parsing file failed: '" + path + "'."){};
};
}
}

template <typename T, typename X, typename... Args>
T parse_file(const std::string& p, X fnc, Args&... args) {
  using namespace boost::filesystem;
  if (not exists(p))
    throw Exception::File::Existence(p);

  bool use_gzip = path(p).extension() == ".gz";
  bool use_bzip2 = path(p).extension() == ".bz2";
  std::ios_base::openmode flags = std::ios_base::in;
  if (use_gzip or use_bzip2) flags |= std::ios_base::binary;

  std::ifstream file(p, flags);
  if (not file) throw Exception::File::Access(p);
  boost::iostreams::filtering_stream<boost::iostreams::input> in;
  if (use_gzip) in.push(boost::iostreams::gzip_decompressor());
  if (use_bzip2) in.push(boost::iostreams::bzip2_decompressor());
  in.push(file);

  T return_value = fnc(in, args...);
  if (in.bad()) throw Exception::File::Reading(p);
  // if (in.fail())
  //   throw Exception::File::Parsing(p);
  return return_value;
};

#endif
