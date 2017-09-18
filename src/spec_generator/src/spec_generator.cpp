#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "aux.hpp"
#include "spec_parser/Driver.hpp"

#define OPTION(_id, _name, _shorthand, _default_value)          \
  namespace option {                                            \
  namespace _id {                                               \
  static constexpr char name[] = #_name;                        \
  static constexpr char shorthand[] = #_shorthand;              \
  static constexpr char declaration[] = #_name "," #_shorthand; \
  static constexpr char default_value[] = _default_value;       \
  }                                                             \
  }

using namespace spec_parser;

namespace po = boost::program_options;

constexpr char rate_variable[] = "rate";
constexpr char odds_variable[] = "odds";

OPTION(help, help, h, "")

OPTION(rate, rate_formula, r, "(gene + spot) * type + 1")
OPTION(odds, odds_formula, o, "gene * type + 1")
OPTION(dist, dist, d, "Lognormal(1,1)")

struct Options {
  std::string rate_formula;
  std::string odds_formula;
  std::string dist;
};

Options parse_command_line(int argc, char** argv) {
  po::options_description options("Program Options");
  options.add_options()
    (option::rate::declaration,
     po::value<std::string>()
       ->default_value(option::rate::default_value)
       ->required(),
     "the rate formula to use")
    (option::odds::declaration,
     po::value<std::string>()
       ->default_value(option::odds::default_value)
       ->required(),
     "the odds formula to use")
    (option::dist::declaration,
     po::value<std::string>()->default_value(option::dist::default_value),
     "distribution to use for coefficients")
    (option::help::declaration,
     "view this help message")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  if (vm.count(option::help::name)) {
    std::cout << options;
    exit(EXIT_SUCCESS);
  }
  po::notify(vm);
  return {
      vm[option::rate::name].as<std::string>(),
      vm[option::odds::name].as<std::string>(),
      vm[option::dist::name].as<std::string>(),
  };
}

void write(const std::string& rate_formula, const std::string& odds_formula,
           const std::string& dist) {
  Driver driver;
  driver.parse(rate_variable + (":=" + rate_formula));
  driver.parse(odds_variable + (":=" + odds_formula));
  for (auto& x : driver.random_variables) {
    driver.parse(x.second.full_id() + "~" + dist);
  }

  std::cout << "# Regression formulas" << std::endl;
  std::cout << "# rate := " << rate_formula << std::endl;
  std::cout << "# odds := " << odds_formula << std::endl;

  std::cout << std::endl;

  std::cout << "# Regression equations" << std::endl;
  for (auto& x : driver.regression_equations) {
    std::cout
        << x.first + "="
               + intercalate<std::vector<std::string>::iterator, std::string>(
                     x.second.variables.begin(), x.second.variables.end(), "*")
        << std::endl;
  }

  std::cout << std::endl;

  std::cout << "# Distribution specifications" << std::endl;
  // construct sorted variables map
  std::map<std::string, RandomVariable*> vmap;
  for (auto& x : driver.random_variables) {
    vmap.emplace(x.first, &x.second);
  }
  for (auto& x : vmap) {
    std::cout << to_string(*x.second) << std::endl;
  }
}

int main(int argc, char** argv) {
  try {
    Options options = parse_command_line(argc, argv);
    write(options.rate_formula, options.odds_formula, options.dist);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
