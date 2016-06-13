#ifndef LOG_HPP
#define LOG_HPP

// needed for dynamic linking to the Boost log library
#define BOOST_LOG_DYN_LINK 1

#include <boost/log/trivial.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <string>
#include "verbosity.hpp"

#define SEVERITY_THRESHOLD logging::trivial::info

// using severity_level = boost::log::trivial::severity_level;
using severity_level = Verbosity;

// register a global logger
BOOST_LOG_GLOBAL_LOGGER(logger,
                        boost::log::sources::severity_logger_mt<severity_level>)

#define LOG(severity) BOOST_LOG_SEV(logger::get(), severity_level::severity)
// #define LOG(x) BOOST_LOG_TRIVIAL(x)

void init_logging(const std::string &path);

#endif
