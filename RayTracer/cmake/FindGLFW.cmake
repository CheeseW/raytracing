include_guard()

find_package(PkgConfig)
pkg_check_modules(GLFW QUIET glfw)

if (WIN32)
find_path(GLFW_INCLUDE_DIR 
	NAMES GLFW/glfw3.h
	PATHS
	${GLFW_ROOT_DIR}/include # the user defined one can override system setting
	$ENV{PROGRAMFILES}/include
	DOC "The directory where GLFW/glfw.h resides"
	PATH_SUFFIXES glfw
)

if (GLFW_USE_STATIC_LIBS)
	set (GLFW_LIBRARY_NAME glfw3)
else()
	set (GLFW_LIBRARY_NAME glfw3dll)

endif (GLFW_USE_STATIC_LIBS)

find_library(
	GLFW_LIBRARY
	NAMES ${GLFW_LIBRARY_NAME}
	PATH
	${GLFW_ROOT_DIR}/lib
	$ENV{PROGRAMFILES}/lib
)

unset(GLFW_LIBRARY_NAME)
else()
#TODO: handle find on other systems
endif(WIN32)

mark_as_advanced(GLFW_FOUND GLFW_LIBRARY GLFW_INCLUDE_DIR GLFW_LIBRARY)

include(FindPackageHANDLEStandardArgs)
find_package_handle_standard_args(GLFW 
	REQUIRED_VARS GLFW_LIBRARY GLFW_INCLUDE_DIR
)

if (GLFW_FOUND)
set(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})
set(GLFW_LIBRARIES ${GLFW_LIBRARY})

endif (GLFW_FOUND)

if (GLFW_FOUND AND NOT TARGET GLFW::GLFW)
if (WIN32)
if (GLFW_USE_STATIC_LIBS)
add_library(GLFW::GLFW STATIC IMPORTED)
set_target_properties(GLFW::GLFW PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}"
	IMPORTED_LOCATION "${GLFW_LIBRARY}"
	)
else()
add_library(GLFW::GLFW SHARED IMPORTED)
set_target_properties(GLFW::GLFW PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}"
	IMPORTED_IMPLIB "${GLFW_LIBRARY}"
	)
endif()
endif(WIN32)

endif()

