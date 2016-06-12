#ifndef LOG_HPP
#define LOG_HPP

// necessary when linking the boost_log library dynamically
#define BOOST_LOG_DYN_LINK 1

#include <boost/log/trivial.hpp>
#include <boost/log/sources/global_logger_storage.hpp>

#include <string>

#define SEVERITY_THRESHOLD logging::trivial::info

// register a global logger
BOOST_LOG_GLOBAL_LOGGER(
    logger, boost::log::sources::
                severity_logger_mt<boost::log::trivial::severity_level>)

#define LOG(severity) \
  BOOST_LOG_SEV(logger::get(), boost::log::trivial::severity)
// #define LOG(x) BOOST_LOG_TRIVIAL(x)

void init_logging(const std::string &path);

#endif
