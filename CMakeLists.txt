cmake_minimum_required (VERSION 3.7)
cmake_policy(VERSION 3.7...3.20)

project(DiffTetVR)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
#MESSAGE(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.c src/*.hpp src/*.h Data/Shaders/*.glsl)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)
include_directories(src)

file(GLOB_RECURSE MODULE_SOURCES src/Module/*.cpp src/Module/*.hpp)
# TODO: Undo when module is excluded.
foreach(MODULE_SOURCE ${MODULE_SOURCES})
    list(REMOVE_ITEM SOURCES ${MODULE_SOURCE})
endforeach()

file(GLOB_RECURSE SHADER_FILES Data/Shaders/*.glsl)
file(GLOB_RECURSE CUDA_MODULES Data/Shaders/*.cu)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders" PREFIX "Shaders" FILES ${SHADER_FILES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders" PREFIX "CUDA Modules" FILES ${CUDA_MODULES})

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    # Avoid warning 'IMPORTED_IMPLIB not set for imported target "sgl" configuration "MinSizeRel/RelWithDebInfo".'
    set(CMAKE_CONFIGURATION_TYPES Debug Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W3 /EHsc /Zc:__cplusplus")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

# Prefer GLVND by default when available.
if (${CMAKE_VERSION} VERSION_GREATER "3.11")
    cmake_policy(SET CMP0072 NEW)
endif()

if(VCPKG_TOOLCHAIN)
    if(EXISTS "${Z_VCPKG_ROOT_DIR}/triplets/${VCPKG_TARGET_TRIPLET}.cmake")
        include("${Z_VCPKG_ROOT_DIR}/triplets/${VCPKG_TARGET_TRIPLET}.cmake")
    elseif(EXISTS "${Z_VCPKG_ROOT_DIR}/triplets/community/${VCPKG_TARGET_TRIPLET}.cmake")
        include("${Z_VCPKG_ROOT_DIR}/triplets/community/${VCPKG_TARGET_TRIPLET}.cmake")
    else()
        message(FATAL_ERROR "Could not find a vcpkg triplet file for ${VCPKG_TARGET_TRIPLET}.")
    endif()
endif()

if (MSVC)
    set(DEFAULT_USE_TBB ON)
else()
    set(DEFAULT_USE_TBB OFF)
endif()

option(USE_STATIC_STD_LIBRARIES "Link with standard libraries statically (only supported on Linux for now)." OFF)
option(USE_GLIBCXX_DEBUG "Use the -D_GLIBCXX_DEBUG flag when compiling with GCC." OFF)
option(USE_TBB "Build with TBB threading support instead of using OpenMP." ${DEFAULT_USE_TBB})
option(USE_OVM "Use OpenVolumeMesh for the internal representation." ON)
option(SUPPORT_PYTORCH "Build the PyTorch module." ON)

set(DATA_PATH "${CMAKE_SOURCE_DIR}/Data" CACHE PATH "Location of folder 'Data'")
add_definitions(-DDATA_PATH=\"${DATA_PATH}\")

if ((UNIX OR MSYS OR MINGW) AND (NOT APPLE OR NOT VCPKG_TOOLCHAIN))
    find_package(PkgConfig REQUIRED)
endif()

# CUDA nvcc does not support MSYS/MinGW.
if(MSYS OR MINGW OR (${CMAKE_GENERATOR} STREQUAL "MinGW Makefiles") OR (${CMAKE_GENERATOR} STREQUAL "MSYS Makefiles"))
    set(SUPPORT_PYTORCH OFF)
endif()


if((MSYS OR MINGW OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")) AND ${USE_GLIBCXX_DEBUG})
    add_compile_definitions(_GLIBCXX_DEBUG)
endif()


set(FPHSA_NAME_MISMATCHED TRUE)
if(VCPKG_TOOLCHAIN)
    find_package(SDL2 CONFIG REQUIRED)
else()
    find_package(SDL2 REQUIRED)
endif()

if (VCPKG_TOOLCHAIN)
    find_package(Boost COMPONENTS system filesystem locale QUIET)
endif()

find_package(sgl QUIET)
if(NOT sgl_FOUND)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};third_party/sgl/install/lib/cmake/sgl)
    find_package(sgl REQUIRED)
endif()
if(VCPKG_TOOLCHAIN)
    find_package(tinyxml2 CONFIG REQUIRED)
    find_package(jsoncpp CONFIG REQUIRED)
else()
    find_package(TinyXML2 REQUIRED)
    find_package(jsoncpp QUIET)
    if (NOT jsoncpp_FOUND)
        pkg_check_modules(JSONCPP jsoncpp)
    endif()
endif()

get_target_property(SGL_INTERFACE_COMPILE_DEFINITIONS sgl INTERFACE_COMPILE_DEFINITIONS)
if (NOT "SUPPORT_VULKAN" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    message(FATAL_ERROR "sgl was not built with Vulkan support enabled.")
endif()

if (${SUPPORT_PYTORCH})
    find_package(Torch QUIET)
    find_package(Python3 COMPONENTS Development QUIET)
endif()
if (${SUPPORT_PYTORCH} AND NOT Torch_FOUND)
    MESSAGE(STATUS "PyTorch support was enabled, but PyTorch couldn't be found. Disabling PyTorch support.")
endif()

if ("SUPPORT_CUDA_INTEROP" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
        # https://cmake.org/cmake/help/git-stage/module/FindCUDA.html
        find_package(CUDA QUIET)
        if(${CUDA_FOUND})
            set(CUDA_ENABLED true)
        endif()
    else()
        # https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html
        find_package(CUDAToolkit QUIET)
        if(${CUDAToolkit_FOUND})
            set(CUDA_ENABLED true)
        endif()
    endif()
    if (NOT CUDA_FOUND AND NOT CUDAToolkit_FOUND)
        MESSAGE(STATUS "CUDA could not be found. Disabling tiny-cuda-nn an QuickMLP support.")
    endif()
else()
    MESSAGE(STATUS "sgl was not built with CUDA interoperability support.")
endif()

set(CUDA_HOST_COMPILER_COMPATIBLE true)
if (CUDA_ENABLED)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        MESSAGE(STATUS "CUDA was found and is compatible with the host compiler.")
    else()
        set(CUDA_ENABLED false)
        set(CUDA_HOST_COMPILER_COMPATIBLE false)
        MESSAGE(WARNING "CUDA was found, but the compiler cannot be enabled. Please check if CUDA is compatible with the used host compiler.")
    endif()
endif()

if (NOT ${USE_OVM})
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Tet/Loaders/OvmLoader.hpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Tet/Loaders/OvmLoader.cpp)
endif()


if(WIN32)
    add_executable(DiffTetVR WIN32 ${SOURCES} ${CUDA_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/CMake/winres.rc)
elseif(APPLE)
    add_executable(DiffTetVR MACOSX_BUNDLE ${SOURCES} ${CUDA_SOURCES})
else()
    add_executable(DiffTetVR ${SOURCES} ${CUDA_SOURCES})
endif()

if((MSYS OR MINGW OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")) AND ${USE_STATIC_STD_LIBRARIES})
    target_link_options(DiffTetVR PRIVATE -static-libgcc -static-libstdc++)
endif()

if (VCPKG_TOOLCHAIN AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND ${CMAKE_VERSION} VERSION_GREATER "3.12"
        AND "SGL_STATIC_LIBRARY" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    #target_link_options(DiffTetVR PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/CMake/symbols_linux_gcc.map")
    target_link_options(DiffTetVR PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/CMake/symbols_linux_gcc.map")
endif()

if (APPLE)
    set(BUNDLE_PATH ${CMAKE_CURRENT_BINARY_DIR}/DiffTetVR.app)
    set_target_properties(
            DiffTetVR PROPERTIES
            OUTPUT_NAME "DiffTetVR"
            MACOSX_BUNDLE_EXECUTABLE_NAME "DiffTetVR"
            MACOSX_BUNDLE_BUNDLE_NAME "DiffTetVR"
            FOLDER "DiffTetVR"
            MACOSX_BUNDLE_SHORT_VERSION_STRING ""
            MACOSX_BUNDLE_LONG_VERSION_STRING ""
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/CMake/MacOSXBundleInfo.plist.in")
endif()


# Hide console window in release build on Windows/MSVC.
# See: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/recipe/VSConfigSpecificSettings
if(MSVC)
    set_target_properties(DiffTetVR PROPERTIES LINK_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE")
    set_target_properties(DiffTetVR PROPERTIES COMPILE_DEFINITIONS_DEBUG "_CONSOLE")
    set_target_properties(DiffTetVR PROPERTIES LINK_FLAGS_RELWITHDEBINFO "/SUBSYSTEM:CONSOLE")
    set_target_properties(DiffTetVR PROPERTIES COMPILE_DEFINITIONS_RELWITHDEBINFO "_CONSOLE")
    set_target_properties(DiffTetVR PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:WINDOWS")
    set_target_properties(DiffTetVR PROPERTIES LINK_FLAGS_MINSIZEREL "/SUBSYSTEM:WINDOWS")
endif()

if(MSVC)
    #target_compile_definitions(DiffTetVR PRIVATE /wd4251)
    if (NOT CUDA_ENABLED)
        set_target_properties(DiffTetVR PROPERTIES COMPILE_FLAGS /wd4251)
    endif()
    target_compile_options(DiffTetVR PRIVATE
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>)
endif()

if (NOT MSVC)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(DiffTetVR PRIVATE NDEBUG)
    endif()
endif()

if(MSYS OR MINGW OR (${CMAKE_GENERATOR} STREQUAL "MinGW Makefiles") OR (${CMAKE_GENERATOR} STREQUAL "MSYS Makefiles"))
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mconsole")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mwindows")
    endif()
    target_link_libraries(DiffTetVR PUBLIC mingw32)
endif()
if(VCPKG_TOOLCHAIN)
    target_link_libraries(DiffTetVR PUBLIC SDL2::SDL2main)
else()
    target_link_libraries(DiffTetVR PUBLIC SDL2::Main)
endif()

target_link_libraries(DiffTetVR PRIVATE sgl)
target_include_directories(DiffTetVR PRIVATE ${sgl_INCLUDES})

if(VCPKG_TOOLCHAIN)
    target_link_libraries(DiffTetVR PRIVATE tinyxml2::tinyxml2)
else()
    target_link_libraries(DiffTetVR PRIVATE ${TINYXML2_LIBRARIES})
endif()
target_include_directories(DiffTetVR PRIVATE ${sgl_INCLUDES} ${TINYXML2_INCLUDE_DIR})

if(CUDA_ENABLED)
    enable_language(CUDA)
    set_target_properties(DiffTetVR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(DiffTetVR PRIVATE CUDA_ENABLED)
endif()
if(CUDAToolkit_FOUND)
    # https://en.wikipedia.org/wiki/CUDA
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability
    if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
        set_property(TARGET DiffTetVR PROPERTY CUDA_ARCHITECTURES 86 90)
    elseif (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.2")
        set_property(TARGET DiffTetVR PROPERTY CUDA_ARCHITECTURES 86)
    else()
        message(FATAL_ERROR "Found CUDA toolkit version ${CUDAToolkit_VERSION} is too old. CUDA 11.2 or greater is required.")
    endif()
    #if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
    #    set_property(TARGET DiffTetVR PROPERTY CUDA_ARCHITECTURES 61 75 86 90)
    #elseif (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.1")
    #    set_property(TARGET DiffTetVR PROPERTY CUDA_ARCHITECTURES 61 75 86)
    #elseif (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "10.0")
    #    set_property(TARGET DiffTetVR PROPERTY CUDA_ARCHITECTURES 61 75)
    #else()
    #    set_property(TARGET DiffTetVR PROPERTY CUDA_ARCHITECTURES 61)
    #endif()
    set_target_properties(DiffTetVR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    MESSAGE(STATUS "CUDA toolkit version: " ${CUDAToolkit_VERSION})
endif()

if (JSONCPP_LIBRARIES)
    target_link_libraries(DiffTetVR PRIVATE ${JSONCPP_LIBRARIES})
    target_link_directories(DiffTetVR PRIVATE ${JSONCPP_LIBRARY_DIRS})
    target_include_directories(DiffTetVR PRIVATE ${JSONCPP_INCLUDE_DIRS})
endif()
if (TARGET jsoncpp_lib)
    get_target_property(jsoncpp_INCLUDES jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
    target_link_libraries(DiffTetVR PRIVATE jsoncpp_lib)
elseif(TARGET jsoncpp_static)
    get_target_property(jsoncpp_INCLUDES jsoncpp_static INTERFACE_INCLUDE_DIRECTORIES)
    target_link_libraries(DiffTetVR PRIVATE jsoncpp_static)
endif()
if (TARGET jsoncpp_object)
    target_link_libraries(DiffTetVR PRIVATE jsoncpp_object)
endif()
target_include_directories(DiffTetVR PRIVATE ${jsoncpp_INCLUDES})

# sgl provides Vulkan headers in its interface, which are provided if Vulkan was not found during its build.
find_package(Vulkan QUIET)
if (${Vulkan_FOUND} AND NOT "USE_SGL_VULKAN_HEADERS" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS
        AND ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.21.0")
    target_link_libraries(DiffTetVR PUBLIC Vulkan::Headers)
endif()

if (${SUPPORT_PYTORCH} AND Torch_FOUND AND Python3_FOUND)
    target_link_libraries(DiffTetVR PRIVATE Python3::Python)
    target_link_libraries(DiffTetVR PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(DiffTetVR PRIVATE ${TORCH_INCLUDE_DIR})
    target_compile_definitions(DiffTetVR PRIVATE SUPPORT_PYTORCH)
    #target_compile_definitions(DiffTetVR PRIVATE BUILD_PYTHON_MODULE)   # TODO
endif()

# TODO: Remove when module is excluded.
#find_package(Python3 COMPONENTS Development REQUIRED)
#target_link_libraries(DiffTetVR PRIVATE Python3::Python)

#if(VCPKG_TOOLCHAIN)
#    find_package(CURL CONFIG REQUIRED)
#else()
#    find_package(CURL REQUIRED)
#endif()
#target_link_libraries(DiffTetVR PUBLIC ${CURL_LIBRARIES})  # CURL::libcurl, since CMake 3.12
#target_include_directories(DiffTetVR PUBLIC ${CURL_INCLUDES} ${CURL_INCLUDE_DIRS})

# TODO: Make this optional?
add_subdirectory(third_party/fuchsia_radix_sort)
target_link_libraries(DiffTetVR PRIVATE vk-radix-sort)
target_compile_definitions(DiffTetVR PRIVATE USE_FUCHSIA_RADIX_SORT_CMAKE)

if (${USE_OVM})
    add_subdirectory(third_party/OpenVolumeMesh)
    target_link_libraries(DiffTetVR PRIVATE OpenVolumeMesh::OpenVolumeMesh)
    target_compile_definitions(DiffTetVR PRIVATE USE_OPEN_VOLUME_MESH)
    if(MSYS OR MINGW OR (${CMAKE_GENERATOR} STREQUAL "MinGW Makefiles") OR (${CMAKE_GENERATOR} STREQUAL "MSYS Makefiles"))
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/Tet/Loaders/OvmLoader.cpp PROPERTIES COMPILE_FLAGS -Wa,-mbig-obj)
    endif()
endif()

if (${USE_TBB})
    find_package(TBB REQUIRED)
    target_link_libraries(DiffTetVR PRIVATE TBB::tbb)
    target_compile_definitions(DiffTetVR PRIVATE USE_TBB)
    if (USE_GTEST)
        target_link_libraries(LineVis_test PRIVATE TBB::tbb)
        target_compile_definitions(LineVis_test PRIVATE USE_TBB)
    endif()
else()
    # According to https://devblogs.microsoft.com/cppblog/improved-openmp-support-for-cpp-in-visual-studio/,
    # support for LLVM OpenMP was added with Visual Studio 2019 version 16.9. According to
    # https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170,
    # both Visual Studio 2019 version 16.8 and 16.9 have _MSC_VER / MSVC_VERSION 1928, so we need to check
    # for a minimum version 1929.
    if (APPLE)
        # There are some problems with Homebrew OpenMP as of 2022-10-20.
        find_package(OpenMP QUIET)
        if (OpenMP_FOUND)
            MESSAGE(STATUS "Found OpenMP version ${OpenMP_CXX_VERSION}. Enabling OpenMP threading support.")
        else()
            MESSAGE(STATUS "OpenMP not found. Disabling OpenMP threading support.")
        endif()
    else()
        find_package(OpenMP REQUIRED)
    endif()
    if(OpenMP_FOUND)
        if(MSVC)
            option(USE_MSVC_OPENMP_LLVM "Use LLVM OpenMP with MSVC (available since Visual Studio 2019 version 16.9)." ON)
            option(USE_MSVC_OPENMP_EXPERIMENTAL "Use experimental OpenMP with MSVC (available since Visual Studio 2019 version 16.9)." ON)

            if(${MSVC_VERSION} GREATER_EQUAL 1929 AND (USE_MSVC_OPENMP_LLVM OR USE_MSVC_OPENMP_EXPERIMENTAL))
                define_property(
                        SOURCE
                        PROPERTY COMPILE_FLAGS
                        INHERITED
                        BRIEF_DOCS "brief-doc"
                        FULL_DOCS "full-doc"
                )
                if(USE_MSVC_OPENMP_LLVM)
                    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp:llvm")
                    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")
                    set_directory_properties(PROPERTIES COMPILE_FLAGS "/openmp:llvm")
                endif()
                if(USE_MSVC_OPENMP_EXPERIMENTAL)
                    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp:experimental")
                    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
                    set_directory_properties(PROPERTIES COMPILE_FLAGS "/openmp:experimental")
                endif()
                if(USE_MSVC_OPENMP_LLVM AND USE_MSVC_OPENMP_EXPERIMENTAL)
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w44005")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44005")
                endif()
                set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/CMake/winres.rc" PROPERTIES COMPILE_FLAGS "")
            endif()
        elseif(NOT TARGET OpenMP::OpenMP_CXX)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        else()
            target_link_libraries(DiffTetVR PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()


function(compile_fatbin SRC OUTPUT)
    add_custom_command(
            OUTPUT "${OUTPUT}"
            COMMAND "${CMAKE_CUDA_COMPILER}" -fatbin -arch=all -Wno-deprecated-gpu-targets "${SRC}" -o "${OUTPUT}"
            MAIN_DEPENDENCY "${SRC}"
            VERBATIM
    )
endfunction()

if ("SUPPORT_CUDA_INTEROP" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS AND NOT MSYS AND NOT MINGW AND ${CUDA_HOST_COMPILER_COMPATIBLE})
    foreach(CUDA_MODULE IN LISTS CUDA_MODULES)
        get_filename_component(CUDA_MODULE_NAME "${CUDA_MODULE}" NAME_WE)
        set(DEST "${CUDA_MODULE_NAME}.fatbin")
        set(OUTPUT "${CMAKE_SOURCE_DIR}/Data/__cudacache__/${DEST}")
        compile_fatbin("${CUDA_MODULE}" "${OUTPUT}")
        list(APPEND CUDA_FATBIN_FILES "${OUTPUT}")
    endforeach()
    add_custom_target(COMPILE_CUDA_MODULES ALL DEPENDS ${CUDA_FATBIN_FILES})
    add_dependencies(${CMAKE_PROJECT_NAME} COMPILE_CUDA_MODULES)
endif()
