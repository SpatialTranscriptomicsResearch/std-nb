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

#include <iostream>
#include <fstream>
#include "terminal.hpp"
#include <git_config.hpp>
#include "cli.hpp"

using namespace std;

namespace po = boost::program_options;

static const string default_error_msg =
    "Please inspect the command line help with -h or --help.";

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

ExecutionInformation process_cli_options(
    int argc, const char **argv, Verbosity &verbosity,
    const string &usage_string, po::options_description &cli_options,
    bool use_positional_options,
    po::positional_options_description &positional_options) {
  ExecutionInformation exec_info(argv[0], GIT_DESCRIPTION, GIT_BRANCH, argc,
                                 argv);

  static const size_t MIN_COLS = 60;
  static const size_t MAX_COLS = 80;
  size_t cols = get_terminal_width();
  if (cols < MIN_COLS) cols = MIN_COLS;
  if (cols > MAX_COLS) cols = MAX_COLS;

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
    std::cout << "Error while parsing command line options:" << std::endl
              << "Option " << e.get_option_name() << " not known." << std::endl
              << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::ambiguous_option &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "Option " << e.get_option_name() << " is ambiguous."
              << std::endl << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::multiple_values &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "Option " << e.get_option_name()
              << " was specified multiple times." << std::endl
              << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::multiple_occurrences &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "Option " << e.get_option_name()
              << " was specified multiple times." << std::endl
              << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::invalid_option_value &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "The value specified for option " << e.get_option_name()
              << " has an invalid format." << std::endl << default_error_msg
              << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::too_many_positional_options_error &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "Too many positional options were specified." << std::endl
              << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::invalid_command_line_syntax &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "Invalid command line syntax." << std::endl
              << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::invalid_command_line_style &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "There is a programming error related to command line style."
              << std::endl << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::reading_file &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "The config file can not be read." << std::endl
              << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::validation_error &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "Validation of option " << e.get_option_name() << " failed."
              << std::endl << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (po::error &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "No further information as to the nature of this error is "
                 "available, please check your command line arguments."
              << std::endl << default_error_msg << std::endl;
    exit(EXIT_FAILURE);
  } catch (exception &e) {
    std::cout << "An error occurred while parsing command line options."
              << std::endl << e.what() << std::endl << default_error_msg
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (vm.count("verbose")) verbosity = Verbosity::Verbose;
  if (vm.count("noisy")) verbosity = Verbosity::Debug;

  if (vm.count("version") and not vm.count("help")) {
    std::cout << exec_info.program_name << " " << exec_info.program_version
              << " [" << GIT_BRANCH << " branch]" << std::endl;
    if (verbosity >= Verbosity::Verbose) std::cout << GIT_SHA1 << std::endl;
    exit(EXIT_SUCCESS);
  }

  if (vm.count("help")) {
    std::cout << exec_info.program_name << " " << exec_info.program_version
              << std::endl;
    std::cout
        << "Copyright (C) 2015 Jonas Maaskola\n"
           "Provided under GNU General Public License Version 3 or later.\n"
           "See the file COPYING provided with this software for details of "
           "the license.\n" << std::endl;
    std::cout << usage_string << std::endl << std::endl;
    std::cout << cli_options << std::endl;
    // std::cout << visible_options << std::endl;
    /*
    switch (verbosity) {
      case Verbosity::Nothing:
      case Verbosity::Error:
      case Verbosity::Info:
        std::cout
            << "Advanced and hidden options not shown. Use -hv or -hV to show "
               "them." << std::endl;
        break;
      case Verbosity::Verbose:
        std::cout << common_options << std::endl;
        std::cout << "Hidden options not shown. Use -hV to show them."
                  << std::endl;
        break;
      case Verbosity::Debug:
      case Verbosity::Everything:
        std::cout << common_options << std::endl;
        std::cout << hidden_options << std::endl;
        break;
    }
    */
    exit(EXIT_SUCCESS);
  }

  try {
    po::notify(vm);
  } catch (po::required_option &e) {
    std::cout << "Error while parsing command line options:" << std::endl
              << "The required option " << e.get_option_name()
              << " was not specified." << std::endl << default_error_msg
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (vm.count("config")) {
    string config_path = vm["config"].as<string>();
    ifstream ifs(config_path.c_str());
    if (!ifs) {
      std::cout << "Error: can not open config file: " << config_path
                << std::endl << default_error_msg << std::endl;
      exit(EXIT_FAILURE);
    } else {
      try {
        store(parse_config_file(ifs, cli_options), vm);
      } catch (po::multiple_occurrences &e) {
        std::cout << "Error while parsing config file:" << std::endl
                  << "Option " << e.get_option_name()
                  << " was specified multiple times." << std::endl
                  << default_error_msg << std::endl;
        exit(EXIT_FAILURE);
      } catch (po::unknown_option &e) {
        std::cout << "Error while parsing config file:" << std::endl
                  << "Option " << e.get_option_name() << " not known."
                  << std::endl << default_error_msg << std::endl;
        exit(EXIT_FAILURE);
      } catch (po::invalid_option_value &e) {
        std::cout << "Error while parsing config file:" << std::endl
                  << "The value specified for option " << e.get_option_name()
                  << " has an invalid format." << std::endl << default_error_msg
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      notify(vm);
    }
  }

  return exec_info;
}
