cmake_minimum_required(VERSION 2.8)
project( egbis )
set (CMAKE_CXX_STANDARD 14)

# Add the EGBIS algortihm
# add_subdirectory(egbis)

include_directories(/usr/include/python3.6/)
add_library( egbis SHARED egbis.h egbis.cpp egbis/segment-image.cpp egbis/segment-image.h egbis/segment-graph.cpp egbis/segment-graph.h egbis/disjoint-set.cpp egbis/disjoint-set.h egbis/image.h egbis/filter.cpp egbis/filter.h egbis/misc.cpp egbis/misc.h)
#add_executable(main egbis.h egbis.cpp egbis/segment-image.cpp egbis/segment-image.h egbis/segment-graph.cpp egbis/segment-graph.h egbis/disjoint-set.cpp egbis/disjoint-set.h egbis/image.h egbis/filter.cpp egbis/filter.h egbis/misc.cpp egbis/misc.h)
FIND_LIBRARY(PYTHON_LIBRARY python3.6 /usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/)
#find_package(PythonLibs 3.6 REQUIRED)
#find_package(Boost 1.67 REQUIRED COMPONENTS system )
message(STATUS mahdi: ${PythonLibs})

#target_link_libraries( main ${PYTHON_LIBRARY} )

