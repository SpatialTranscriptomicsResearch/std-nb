# - Try to find LBFGS.h
# Once done this will define
#  LBFGS_FOUND - System has LBFGS.h
#  LBFGS_INCLUDE_DIRS - The LBFGS.h include directories

find_package(PkgConfig)

find_path(LBFGS_INCLUDE_DIR LBFGS.h)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LBFGS DEFAULT_MSG LBFGS_INCLUDE_DIR)

# mark_as_advanced(LIBXML2_INCLUDE_DIR LIBXML2_LIBRARY )

set(LBFGS_INCLUDE_DIRS ${LBFGS_INCLUDE_DIR} )
