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
 *       Filename:  timer.cpp
 *
 *    Description:  Some timing routines.
 *
 *        Created:  05/30/2012 06:42:25 PM
 *
 *         Author:  Jonas Maaskola <jonas@maaskola.de>
 *
 * =====================================================================================
 */

#include <sys/time.h>
#include "timer.hpp"

Timer::Timer() { tick(); }

void Timer::tick() { gettimeofday(&start, NULL); }

/** Return time in micro seconds since tick(). */
double Timer::tock() const {
  struct timeval end;
  gettimeofday(&end, NULL);
  double time = (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec
                - start.tv_usec;
  return time;
}
