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
 *       Filename:  timer.hpp
 *
 *    Description:  Some timing routines.
 *
 *        Created:  05/30/2012 06:42:25 PM
 *
 *         Author:  Jonas Maaskola <jonas@maaskola.de>
 *
 * =====================================================================================
 */

#ifndef TIMER_HPP
#define TIMER_HPP

#include <ctime>
#include <sys/time.h>

class Timer {
private:
  struct timeval start;

public:
  /** Initialize and start timer. */
  Timer();

  /** (Re-)Start the timer. */
  void tick();

  /** Time in micro seconds since tick(). */
  double tock() const;
};

#endif /* ----- #ifndef TIMER_HPP  ----- */
