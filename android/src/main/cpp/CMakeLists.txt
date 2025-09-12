cmake_minimum_required(VERSION 3.10.2)
project("ultralytics")

add_library(
ultralytics  
SHARED
native-lib.cpp
)

find_library(
log-lib
log
)

target_link_libraries(
ultralytics
${log-lib}
)

