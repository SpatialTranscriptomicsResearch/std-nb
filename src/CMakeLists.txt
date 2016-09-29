# add definitions, compiler switches, etc.
ADD_DEFINITIONS(-Wall -Wshadow -funroll-loops -march=native)

# Make it possible to #include generated header files for out-of-source builds
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

ADD_LIBRARY(multiScoop SHARED aux.cpp counts.cpp executioninformation.cpp
  PartialModel.cpp features.cpp mix.cpp verbosity.cpp
  hierarchical_kmeans.cpp
  log.cpp io.cpp metropolis_hastings.cpp parameters.cpp priors.cpp
  sampling.cpp target.cpp terminal.cpp timer.cpp pdist.cpp)

ADD_EXECUTABLE(enrich enrich.cpp cli.cpp entropy.cpp)
ADD_EXECUTABLE(multiScoop-bin main.cpp cli.cpp entropy.cpp)
ADD_EXECUTABLE(nhdp main-nhdp.cpp nhdp.cpp cli.cpp entropy.cpp)
ADD_EXECUTABLE(hdp main-hdp.cpp hdp.cpp cli.cpp entropy.cpp)
ADD_EXECUTABLE(test_sampling test_sampling.cpp entropy.cpp)
SET_TARGET_PROPERTIES(multiScoop-bin PROPERTIES OUTPUT_NAME multiScoop)

TARGET_LINK_LIBRARIES(multiScoop
  m
  -L${Boost_LIBRARY_DIRS}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_LOG_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${Boost_REGEX_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_DATE_TIME_LIBRARY}
  ${ARMADILLO_LIBRARIES}
  )

TARGET_LINK_LIBRARIES(multiScoop-bin multiScoop)
TARGET_LINK_LIBRARIES(test_sampling multiScoop)
TARGET_LINK_LIBRARIES(enrich multiScoop)
TARGET_LINK_LIBRARIES(nhdp multiScoop)
TARGET_LINK_LIBRARIES(hdp multiScoop)

IF(GOOGLEPERFTOOLS_FOUND)
  TARGET_LINK_LIBRARIES(multiScoop-bin ${PROFILER_LIBRARY} ${TCMALLOC_LIBRARY})
  TARGET_LINK_LIBRARIES(nhdp ${PROFILER_LIBRARY} ${TCMALLOC_LIBRARY})
  TARGET_LINK_LIBRARIES(hdp ${PROFILER_LIBRARY} ${TCMALLOC_LIBRARY})
  TARGET_LINK_LIBRARIES(enrich ${PROFILER_LIBRARY} ${TCMALLOC_LIBRARY})
ENDIF()

CONFIGURE_FILE(git_config.hpp.in git_config.hpp)

INSTALL(TARGETS enrich DESTINATION bin)
INSTALL(TARGETS nhdp DESTINATION bin)
INSTALL(TARGETS hdp DESTINATION bin)
INSTALL(TARGETS multiScoop-bin DESTINATION bin)
INSTALL(TARGETS multiScoop DESTINATION lib)
