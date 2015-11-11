# - Try to find LibGOOGLEPERFTOOLS
# Once done this will define
#  GOOGLEPERFTOOLS_FOUND - System has LibGOOGLEPERFTOOLS
#  GOOGLEPERFTOOLS_LIBRARIES - The libraries needed to use LibGOOGLEPERFTOOLS

find_library(TCMALLOC_LIBRARY NAMES tcmalloc)
find_library(PROFILER_LIBRARY NAMES profiler)

set(GOOGLEPERFTOOLS_LIBRARIES ${TCMALLOC_LIBRARY} ${PROFILER_LIBRARY})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBGOOGLEPERFTOOLS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(GOOGLEPERFTOOLS DEFAULT_MSG GOOGLEPERFTOOLS_LIBRARIES)


