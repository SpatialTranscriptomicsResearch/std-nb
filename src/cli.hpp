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
 *       Filename:  main.cpp
 *
 *    Description:  Executable for the HMM package
 *
 *        Created:  Thu Aug 4 22:12:31 2011 +0200
 *
 *         Author:  Jonas Maaskola <jonas@maaskola.de>
 *
 * =====================================================================================
 */

#ifndef CLI_HPP
#define CLI_HPP

#include <string>
#include <boost/program_options.hpp>
#include "executioninformation.hpp"

boost::program_options::options_description gen_generic_options(
    std::string &config_path, size_t cols);

static const int PROCESSING_SUCCESSFUL = 2;

/** Parse and process command line arguments.
 * Return value is either EXIT_SUCCESS, EXIT_FAILURE, or PROCESSING_SUCCESSFUL.
 */
int process_cli_options(
    int argc, const char **argv, Verbosity &verbosity,
    ExecutionInformation &exec_info, const std::string &usage_string,
    boost::program_options::options_description &cli_options,
    bool use_positional_options,
    boost::program_options::positional_options_description &positional_options);

#endif
