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
 *       Filename:  verbosity.hpp
 *
 *    Description:  Represent different levels of output noisyness
 *
 *        Created:  Thu Aug 4 22:12:31 2011 +0200
 *
 *         Author:  Jonas Maaskola <jonas@maaskola.de>
 *
 * =====================================================================================
 */

#ifndef VERBOSITY_HPP
#define VERBOSITY_HPP

#include <string>
#include <iostream>

enum class Verbosity {
  fatal = 0,
  error = 1,
  warning = 2,
  info = 3,
  verbose = 4,
  debug = 5,
  trace = 6,
  everything = 7
};

extern Verbosity verbosity;

std::string to_string(Verbosity verbosity);
std::ostream &operator<<(std::ostream &os, Verbosity verbosity);
std::istream &operator>>(std::istream &is, Verbosity &verbosity);

#endif
