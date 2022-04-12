cmake_minimum_required (VERSION 3.10)
project(BPIMM VERSION 1.0.0
        DESCRIPTION "A boxplus interacting multiple model filter"
        LANGUAGES CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

#find_package(ADEKF CONFIG REQUIRED)
#find_package(ADEKF_VIZ CONFIG REQUIRED)
find_package(Boost 1.41 COMPONENTS system  REQUIRED)

add_subdirectory(ADEKF)
add_subdirectory(ADEKF_VIZ)

# Configure project for usage of Qt5
find_package(Qt5 COMPONENTS Core Gui Widgets PrintSupport Svg Xml  REQUIRED)

# include JKQTPlotter
find_package(JKQTCommonSharedLib REQUIRED)
find_package(JKQTMathTextSharedLib REQUIRED)
find_package(JKQTPlotterSharedLib REQUIRED)
#Include Threads to start ekf loop in other thread
find_package(Threads REQUIRED)
#find VTK
find_package(VTK REQUIRED)

# include all components

add_executable(RadarFlightExample MACOSX_BUNDLE immBodyDistance.cpp)
target_include_directories(RadarFlightExample PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ADEKF/include/misc ADEKF_VIZ/include ../Manifold-RTS-Smoother/include)
target_link_libraries(RadarFlightExample PUBLIC  ${Boost_LIBRARIES}  ADEKF ADEKF_VIZ ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(RadarFlightExample PUBLIC  cxx_std_17 )


add_executable(SmoothRadarFlightExample MACOSX_BUNDLE immBodyDistanceSmooth.cpp)
target_include_directories(SmoothRadarFlightExample PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ADEKF/include/misc ADEKF_VIZ/include ../Manifold-RTS-Smoother/include)
target_link_libraries(SmoothRadarFlightExample PUBLIC  ${Boost_LIBRARIES}  ADEKF ADEKF_VIZ ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(SmoothRadarFlightExample PUBLIC  cxx_std_17 )

add_executable(SmoothVectorFlightExample MACOSX_BUNDLE immVectorSmooth.cpp)
target_include_directories(SmoothVectorFlightExample PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ADEKF/include/misc ADEKF_VIZ/include ../Manifold-RTS-Smoother/include)
target_link_libraries(SmoothVectorFlightExample PUBLIC  ${Boost_LIBRARIES}  ADEKF ADEKF_VIZ ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(SmoothVectorFlightExample PUBLIC  cxx_std_17 )

add_executable(QuatMixing MACOSX_BUNDLE QuatMixing.cpp)
target_include_directories(QuatMixing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ADEKF/include/misc ADEKF_VIZ/include)
target_link_libraries(QuatMixing PUBLIC  ${Boost_LIBRARIES}  ADEKF ADEKF_VIZ ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(QuatMixing PUBLIC  cxx_std_17 )



add_executable(pdrIMM MACOSX_BUNDLE immPDR.cpp CSV_Reader.cpp)
target_include_directories(pdrIMM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ADEKF/include/misc ADEKF_VIZ/include ../Manifold-RTS-Smoother/include)
target_link_libraries(pdrIMM PUBLIC  ${Boost_LIBRARIES}  ADEKF ADEKF_VIZ ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(pdrIMM PUBLIC  cxx_std_17 )