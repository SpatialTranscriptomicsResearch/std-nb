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

#include "cli.hpp"
#include "log.hpp"
#include <git_config.hpp>
#include <iostream>
#include <fstream>
#include "terminal.hpp"

using namespace std;

namespace po = boost::program_options;

const std::string default_error_msg
    = "Please inspect the command line help with -h or --help.";

po::options_description gen_generic_options(string &config_path, size_t cols) {
  po::options_description generic_options("Generic options", cols);
  generic_options.add_options()
    ("config", po::value(&config_path), "Read options from a configuration file. ")
    ("help,h", "Produce help message. Combine with -v or -V for additional commands.")
    ("version", "Print out the version. Also show git SHA1 with -v.")
    ("verbose,v", "Be verbose about the progress.")
    ("noisy,V", "Be very verbose about the progress.")
    ;
  return generic_options;
}

int process_cli_options(
    int argc, const char **argv, ExecutionInformation &exec_info,
    const std::string &usage_string,
    boost::program_options::options_description &cli_options,
    bool use_positional_options,
    boost::program_options::positional_options_description
        &positional_options) {
  namespace po = boost::program_options;
  exec_info = ExecutionInformation(argv[0], GIT_DESCRIPTION, GIT_BRANCH,
                                   BUILD_TYPE, argc, argv);

  const size_t MIN_COLS = 60;
  const size_t MAX_COLS = 80;
  size_t cols = get_terminal_width();
  if (cols < MIN_COLS)
    cols = MIN_COLS;
  if (cols > MAX_COLS)
    cols = MAX_COLS;

  po::variables_map vm;
  try {
    if (not use_positional_options)
      po::store(po::command_line_parser(argc, argv).options(cli_options).run(),
                vm);
    else
      po::store(po::command_line_parser(argc, argv)
                    .options(cli_options)
                    .positional(positional_options)
                    .run(),
                vm);
  } catch (po::unknown_option &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "Option " << e.get_option_name() << " not known.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::ambiguous_option &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "Option " << e.get_option_name() << " is ambiguous.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::multiple_values &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "Option " << e.get_option_name() << " was specified multiple times.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::multiple_occurrences &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "Option " << e.get_option_name() << " was specified multiple times.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::invalid_option_value &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "The value specified for option " << e.get_option_name()
               << " has an invalid format.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::too_many_positional_options_error &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "Too many positional options were specified.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::invalid_command_line_syntax &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "Invalid command line syntax.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::invalid_command_line_style &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "There is a programming error related to command line style.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::reading_file &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "The config file can not be read.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::validation_error &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "Validation of option " << e.get_option_name() << " failed.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (po::error &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "No further information as to the nature of this error is "
                  "available, please check your command line arguments.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  } catch (std::exception &e) {
    LOG(fatal) << "An error occurred while parsing command line options.";
    LOG(fatal) << e.what();
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  }

  if (vm.count("verbose"))
    verbosity = Verbosity::verbose;
  if (vm.count("noisy"))
    verbosity = Verbosity::debug;

  if (vm.count("version") and not vm.count("help")) {
    std::cout << exec_info.name_and_version() << std::endl;
    if (verbosity >= Verbosity::verbose)
      std::cout << GIT_SHA1 << std::endl;
    return EXIT_SUCCESS;
  }

  if (vm.count("help")) {
    std::cout << exec_info.program_name << " " << exec_info.program_version
              << std::endl;
    std::cout
        << "Copyright (C) 2015 Jonas Maaskola\n"
           "Provided under GNU General Public License Version 3 or later.\n"
           "See the file COPYING provided with this software for details of "
           "the license.\n"
        << std::endl;
    std::cout << usage_string << std::endl << std::endl;
    std::cout << cli_options << std::endl;
    return EXIT_SUCCESS;
  }

  try {
    po::notify(vm);
  } catch (po::required_option &e) {
    LOG(fatal) << "Error while parsing command line options:";
    LOG(fatal) << "The required option " << e.get_option_name() << " was not specified.";
    LOG(fatal) << default_error_msg;
    return EXIT_FAILURE;
  }

  if (vm.count("config")) {
    std::string config_path = vm["config"].as<std::string>();
    std::ifstream ifs(config_path.c_str());
    if (!ifs) {
      LOG(fatal) << "Error: can not open config file: " << config_path;
      LOG(fatal) << default_error_msg;
      return EXIT_FAILURE;
    } else {
      try {
        store(parse_config_file(ifs, cli_options), vm);
      } catch (po::multiple_occurrences &e) {
        LOG(fatal) << "Error while parsing config file:";
        LOG(fatal) << "Option " << e.get_option_name() << " was specified multiple times.";
        LOG(fatal) << default_error_msg;
        return EXIT_FAILURE;
      } catch (po::unknown_option &e) {
        LOG(fatal) << "Error while parsing config file:";
        LOG(fatal) << "Option " << e.get_option_name() << " not known.";
        LOG(fatal) << default_error_msg;
        return EXIT_FAILURE;
      } catch (po::invalid_option_value &e) {
        LOG(fatal) << "Error while parsing config file:";
        LOG(fatal) << "The value specified for option " << e.get_option_name() << " has an invalid format.";
        LOG(fatal) << default_error_msg;
        return EXIT_FAILURE;
      }
      notify(vm);
    }
  }

  return PROCESSING_SUCCESSFUL;
}
