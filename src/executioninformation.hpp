/* =====================================================================================
 * Copyright (c) 2011, Jonas Maaskola
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
 *       Filename:  executioninformation.hpp
 *
 *    Description:  Store information about the current program execution
 *
 *        Created:  Thu Aug 4 22:12:31 2011 +0200
 *
 *         Author:  Jonas Maaskola <jonas@maaskola.de>
 *
 * =====================================================================================
 */

#ifndef EXECUTIONINFORMATION_HPP
#define EXECUTIONINFORMATION_HPP

#include <string>
#include "verbosity.hpp"

struct ExecutionInformation {
  ExecutionInformation();
  ExecutionInformation(const std::string &name, const std::string &version,
                       const std::string &git_branch, const std::string &build,
                       int argc, const char **argv);
  ExecutionInformation(const std::string &name, const std::string &version,
                       const std::string &git_branch, const std::string &build,
                       const std::string &cmdline);
  std::string program_name;
  std::string program_version;
  std::string git_branch;
  std::string build_type;
  std::string cmdline;
  std::string datetime;
  std::string directory;
  std::string name_and_version() const;
};

std::string generate_random_label(const std::string &prefix,
                                  size_t n_rnd_char = 5);

#endif
