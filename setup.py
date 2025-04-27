# BSD 2-Clause License
#
# Copyright (c) 2024, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import os
import stat
import platform
import glob
import shutil
import subprocess
import urllib
import zipfile
import tarfile
import inspect
from pathlib import Path
from urllib.request import urlopen
import setuptools
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.dist import Distribution
from setuptools.command import bdist_egg
import torch
from torch.utils.cpp_extension import include_paths, library_paths, BuildExtension, \
    IS_WINDOWS, IS_HIP_EXTENSION, ROCM_VERSION, ROCM_HOME, CUDA_HOME
try:
    from torch.utils.cpp_extension import SYCL_HOME
except ImportError:
    SYCL_HOME = None

extra_compile_args = []
if IS_WINDOWS:
    extra_compile_args.append('/std:c++17')
    extra_compile_args.append('/Zc:__cplusplus')
    extra_compile_args.append('/openmp')
else:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp')


class EggInfoInstallLicense(egg_info):
    def run(self):
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)
        egg_info.run(self)


def TorchExtension(name, sources, *args, **kwargs):
    # The interface was changed between PyTorch versions.
    include_paths_arg_name = inspect.getfullargspec(include_paths).args[0]
    if include_paths_arg_name == 'cuda':
        device_type = CUDA_HOME is not None or ROCM_HOME is not None
    else:  # include_paths_arg_name == 'device_type'
        if CUDA_HOME is not None or ROCM_HOME is not None:
            device_type = 'cuda'
        elif SYCL_HOME is not None:
            device_type = 'xpu'
        else:
            device_type = 'cpu'
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths(device_type)
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(device_type)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    if IS_HIP_EXTENSION:  # (ROCM_HOME is not None) and (torch.version.hip is not None)
        assert ROCM_VERSION is not None
        libraries.append('amdhip64' if ROCM_VERSION >= (3, 5) else 'hip_hcc')
        libraries.append('c10_hip')
        libraries.append('torch_hip')
    elif CUDA_HOME is not None and torch.cuda.is_available():
        libraries.append('cudart')
        libraries.append('c10_cuda')
        libraries.append('torch_cuda')
    elif SYCL_HOME is not None:
        libraries.append('sycl')
        libraries.append('c10_xpu')
        libraries.append('torch_xpu')
        include_dirs.append(os.path.join(SYCL_HOME, 'include', 'sycl'))
    kwargs['libraries'] = libraries

    return setuptools.Extension(name, sources, *args, **kwargs)


def find_all_sources_in_dir(root_dir):
    source_files = []
    for root, subdirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.cpp') or filename.endswith('.cc') or filename.endswith('.c'):
                source_files.append(root + "/" + filename)
    return source_files


def get_cmake_exec():
    cmake_exec = 'cmake'
    if IS_WINDOWS:
        # CMake on Windows is usually not in the PATH. If it is not found, try to use the default location.
        cmake_exec = shutil.which('cmake')
        cmake_default_path = 'C:\\Program Files\\CMake\\bin\\cmake.exe'
        if cmake_exec is None and os.path.isfile(cmake_default_path):
            cmake_exec = cmake_default_path
        return cmake_exec, cmake_exec is not None
    else:
        return cmake_exec, shutil.which('cmake') is not None


def rmtree_ex(dir_path):
    if IS_WINDOWS:
        # https://bugs.python.org/issue43657
        def remove_readonly(func, path, exec_info):
            if func not in (os.unlink, os.rmdir) or exec_info[1].winerror != 5:
                raise exec_info[1]
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(dir_path, onerror=remove_readonly)
    else:
        shutil.rmtree(dir_path)


#sgl_sources = [ 'third_party/sgl/src/Graphics/Vulkan/Utils/Device.cpp' ]
#sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Buffers')
#sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Image')
#sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Render')
#sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Shader')

if not os.path.exists('third_party/sgl/src'):
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'])

include_dirs = [
    'src',
    'third_party',
    'third_party/sgl/src',
    'third_party/sgl/src/Graphics/Vulkan/libs',
    'third_party/sgl/src/Graphics/Vulkan/libs/Vulkan-Headers',
    'third_party/glm',
    'third_party/tinyxml2',
    'third_party/jsoncpp/include',
    'third_party/OpenVolumeMesh/src',
    'third_party/custom',
    'third_party/glslang',
    'third_party/fuchsia_radix_sort/include',
]
source_files = []
source_files += find_all_sources_in_dir('src/Module')
source_files += find_all_sources_in_dir('src/Renderer')
source_files += find_all_sources_in_dir('src/Tet')
source_files += [
    'third_party/sgl/src/Math/Geometry/MatrixUtil.cpp',
    'third_party/sgl/src/Math/Geometry/Plane.cpp',
    'third_party/sgl/src/Math/Geometry/AABB3.cpp',
    'third_party/sgl/src/Math/Geometry/Ray3.cpp',
    'third_party/sgl/src/Utils/Dialog.cpp',
    'third_party/sgl/src/Utils/StringUtils.cpp',
    'third_party/sgl/src/Utils/Env.cpp',
    'third_party/sgl/src/Utils/Convert.cpp',
    'third_party/sgl/src/Utils/AppSettings.cpp',
    'third_party/sgl/src/Utils/Timer.cpp',
    'third_party/sgl/src/Utils/XML.cpp',
    'third_party/sgl/src/Utils/File/Logfile.cpp',
    'third_party/sgl/src/Utils/File/FileUtils.cpp',
    'third_party/sgl/src/Utils/File/Execute.cpp',
    'third_party/sgl/src/Utils/File/FileLoader.cpp',
    'third_party/sgl/src/Utils/File/LineReader.cpp',
    'third_party/sgl/src/Utils/File/CsvParser.cpp',
    'third_party/sgl/src/Utils/File/PathWatch.cpp',
    'third_party/sgl/src/Utils/Events/EventManager.cpp',
    'third_party/sgl/src/Utils/Events/Stream/BinaryStream.cpp',
    'third_party/sgl/src/Utils/Events/Stream/StringStream.cpp',
    'third_party/sgl/src/Utils/Regex/Tokens.cpp',
    'third_party/sgl/src/Utils/Regex/TransformString.cpp',
    'third_party/sgl/src/Utils/Parallel/Histogram.cpp',
    'third_party/sgl/src/Utils/Parallel/Reduction.cpp',
    'third_party/sgl/src/Utils/Mesh/IndexMesh.cpp',
    'third_party/sgl/src/Graphics/Color.cpp',
    'third_party/sgl/src/Graphics/Scene/Camera.cpp',
    'third_party/sgl/src/Graphics/Scene/CameraHelper.cpp',
    'third_party/sgl/src/Graphics/Scene/RenderTarget.cpp',
    'third_party/sgl/src/Graphics/GLSL/PreprocessorGlsl.cpp',
    'third_party/sgl/src/Graphics/Vulkan/libs/volk/volk.c',
    'third_party/sgl/src/Graphics/Vulkan/libs/SPIRV-Reflect/spirv_reflect.c',
    'third_party/sgl/src/Graphics/Vulkan/Utils/VmaImpl.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Memory.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Status.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Instance.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Device.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Swapchain.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/SyncObjects.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Timer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Buffers/Buffer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Buffers/Framebuffer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Image/Image.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Shader/Shader.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Shader/ShaderManager.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/CommandBuffer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Renderer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Data.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Pipeline.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/ComputePipeline.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/GraphicsPipeline.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/RayTracingPipeline.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/AccelerationStructure.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Passes/Pass.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Passes/BlitRenderPass.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Passes/BlitComputePass.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Helpers.cpp',
    'third_party/sgl/src/ImGui/Widgets/TransferFunctionWindow.cpp',
]
source_files += [
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Attribs/InterfaceAttrib.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Attribs/OpenVolumeMeshStatus.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Attribs/StatusAttrib.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/FileManager/FileManager.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/FileManager/TypeNames.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/FileManager/Serializers.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Core/Handles.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Core/ResourceManager.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Core/BaseEntities.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Core/TopologyKernel.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Core/Iterators.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Core/detail/internal_type_name.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Core/Properties/PropertyStorageBase.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/enums.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/PropertyCodecs.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/BinaryIStream.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/BinaryFileReader.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/BinaryFileWriter.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/GeometryWriter.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/GeometryReader.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/Decoder.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/Encoder.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/ovmb_format.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/ovmb_codec.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/IO/detail/WriteBuffer.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Mesh/TetrahedralMeshIterators.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Mesh/HexahedralMeshIterators.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Mesh/TetrahedralMeshTopologyKernel.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Mesh/HexahedralMeshTopologyKernel.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Unstable/Topology/TetTopology.cc',
    'third_party/OpenVolumeMesh/src/OpenVolumeMesh/Unstable/Topology/TriangleTopology.cc',
]
source_files += [
    'third_party/tinyxml2/tinyxml2.cpp',
]
source_files += find_all_sources_in_dir('third_party/jsoncpp/src/lib_json')
source_files += find_all_sources_in_dir('third_party/glslang/SPIRV')
source_files += find_all_sources_in_dir('third_party/glslang/glslang/CInterface')
source_files += find_all_sources_in_dir('third_party/glslang/glslang/GenericCodeGen')
source_files += find_all_sources_in_dir('third_party/glslang/glslang/MachineIndependent')
source_files += find_all_sources_in_dir('third_party/glslang/glslang/ResourceLimits')
if IS_WINDOWS:
    source_files += find_all_sources_in_dir('third_party/glslang/glslang/OSDependent/Windows')
else:
    source_files += find_all_sources_in_dir('third_party/glslang/glslang/OSDependent/Unix')


data_files_all = []
data_files = ['src/Module/difftetvr.pyi']
libraries = []
extra_objects = []
defines = [
    ('USE_GLM',),
    ('SUPPORT_VULKAN',),
    ('SUPPORT_GLSLANG_BACKEND',),
    ('SUPPORT_TINYXML2',),
    ('DISABLE_IMGUI',),
    ('BUILD_PYTHON_MODULE',),
    ('USE_OPEN_VOLUME_MESH',),
    ('USE_FUCHSIA_RADIX_SORT_CMAKE',),
    # For glslang.
    ('ENABLE_SPIRV',),
]
# Change symbol visibility?
if IS_WINDOWS:
    defines.append(('DLL_OBJECT', ''))
    defines.append(('DISABLE_SINGLETON_BOOST_INTERPROCESS',))
    # According to https://learn.microsoft.com/en-us/windows/win32/api/shlwapi/nf-shlwapi-pathremovefilespecw,
    # shlwapi.lib and shlwapi.dll both exist. Maybe this should rather be an extra_objects file?
    libraries.append('shlwapi')
    libraries.append('shell32')
    libraries.append('user32')
    defines.append(('GLSLANG_OSINCLUDE_WIN32', ''))
else:
    defines.append(('DLL_OBJECT', ''))
    # extra_compile_args.append('-O0')  # For debugging tests.
    # extra_compile_args.append('-ggdb')  # For debugging tests.
    libraries.append('dl')
    defines.append(('GLSLANG_OSINCLUDE_UNIX', ''))


if IS_HIP_EXTENSION:  # (ROCM_HOME is not None) and (torch.version.hip is not None)
    defines.append(('SUPPORT_HIP_INTEROP',))
    defines.append(('SUPPORT_COMPUTE_INTEROP',))
    source_files.append('third_party/sgl/src/Graphics/Vulkan/Utils/InteropHIP.cpp')
    source_files.append('third_party/sgl/src/Graphics/Vulkan/Utils/InteropCompute.cpp')
elif CUDA_HOME is not None and torch.cuda.is_available():
    defines.append(('SUPPORT_CUDA_INTEROP',))
    defines.append(('SUPPORT_COMPUTE_INTEROP',))
    source_files.append('third_party/sgl/src/Graphics/Vulkan/Utils/InteropCuda.cpp')
    source_files.append('third_party/sgl/src/Graphics/Vulkan/Utils/InteropCompute.cpp')
elif SYCL_HOME is not None:
    defines.append(('SUPPORT_LEVEL_ZERO_INTEROP',))
    defines.append(('SUPPORT_SYCL_INTEROP',))
    defines.append(('SUPPORT_COMPUTE_INTEROP',))
    source_files.append('third_party/sgl/src/Graphics/Vulkan/Utils/InteropLevelZero.cpp')
    source_files.append('third_party/sgl/src/Graphics/Vulkan/Utils/InteropCompute.cpp')


if platform.machine() == 'x86_64' or platform.machine() == 'AMD64':
    os_arch = 'x86_64'
else:
    os_arch = 'aarch64'


glslang_validator_path = shutil.which('glslangValidator')
env_cmake = dict(os.environ)
if glslang_validator_path is None:
    # Vulkan SDK is missing!
    if IS_WINDOWS:
        # We should check in the future if automatic installation is also viable on desktop Windows...
        if os.getenv('GITHUB_ACTIONS') == 'true':
            vulkan_version = '1.4.304.0'
            if not os.path.isdir('C:\\VulkanSDK'):
                vulkan_installer_exe = f'VulkanSDK-{vulkan_version}-Installer.exe'
                vulkan_sdk_url = f'https://sdk.lunarg.com/sdk/download/1.4.304.0/windows/{vulkan_installer_exe}'
                opener = urllib.request.URLopener()
                opener.addheader('User-Agent', 'Mozilla/5.0')
                filename, headers = opener.retrieve(vulkan_sdk_url, f'third_party/{vulkan_installer_exe}')
                f'third_party/{vulkan_installer_exe}'
                subprocess.run([
                    f'third_party/{vulkan_installer_exe}',
                    '--accept-licenses', '--default-answer', '--confirm-command', 'install'], check=True)
                os.remove(f'third_party/{vulkan_installer_exe}')
            # https://github.com/python/cpython/issues/105889
            # https://docs.python.org/3/library/subprocess.html#popen-constructor
            if 'PATH' in env_cmake:
                env_cmake['PATH'] += f';C:\\VulkanSDK\\{vulkan_version}\\Bin'
                os.environ['PATH'] += f';C:\\VulkanSDK\\{vulkan_version}\\Bin'
            else:
                env_cmake['PATH'] = f'C:\\VulkanSDK\\{vulkan_version}\\Bin'
                os.environ['PATH'] = f'C:\\VulkanSDK\\{vulkan_version}\\Bin'
            env_cmake['VULKAN_SDK'] = f'C:\\VulkanSDK\\{vulkan_version}'
            glslang_validator_path = shutil.which('glslangValidator', path=f'C:\\VulkanSDK\\{vulkan_version}\\Bin')
        else:
            raise RuntimeError('Missing Vulkan SDK. Please install it from https://vulkan.lunarg.com/sdk/home#windows.')
    if not IS_WINDOWS:
        if not os.path.isdir('third_party/VulkanSDK'):
            vulkan_sdk_url = 'https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz'
            # Gets "HTTP Error 403: Forbidden".
            # if not os.path.isdir(f'third_party/vulkan-sdk'):
            #     urllib.request.urlretrieve(vulkan_sdk_url, f'third_party/vulkan-sdk.tar.gz')
            # https://stackoverflow.com/questions/34957748/http-error-403-forbidden-with-urlretrieve
            opener = urllib.request.URLopener()
            opener.addheader('User-Agent', 'Mozilla/5.0')
            filename, headers = opener.retrieve(vulkan_sdk_url, 'third_party/vulkan-sdk.tar.gz')
            with tarfile.open('third_party/vulkan-sdk.tar.gz', 'r') as tar_ref:
                tar_ref.extractall('third_party/VulkanSDK')
            os.remove('third_party/vulkan-sdk.tar.gz')
            vulkan_sdk_path = os.path.join('third_party', 'VulkanSDK', os.listdir('third_party/VulkanSDK')[0])
            vulkan_sdk_root = os.path.abspath(os.path.join(vulkan_sdk_path, os_arch))
            if os_arch != "x86_64":
                subprocess.run([
                    os.path.join(vulkan_sdk_path, 'vulkansdk'),
                    '-j', f'{os.cpu_count()}', 'vulkan-loader', 'glslang', 'shaderc'], check=True)
            # Fix pkgconfig file.
            shaderc_pkgconfig_file = os.path.join(vulkan_sdk_root, 'lib', 'pkgconfig', 'shaderc.pc')
            if os.path.isfile(shaderc_pkgconfig_file):
                # subprocess.run([
                #     'sed', '-i', f"'3s;.*;prefix=\"'{prefix_path}'\";'", shaderc_pkgconfig_file], check=True)
                # subprocess.run([
                #     'sed', '-i', "'5s;.*;libdir=${prefix}/lib;'", shaderc_pkgconfig_file], check=True)
                with open(shaderc_pkgconfig_file, 'r') as pkgconf_file:
                    pkgconf_lines = pkgconf_file.readlines()
                    pkgconf_lines[3] = f'prefix="{vulkan_sdk_root}"\n'
                    pkgconf_lines[5] = 'libdir=${prefix}/lib\n'
                with open(shaderc_pkgconfig_file, 'w') as pkgconf_file:
                    pkgconf_file.writelines(pkgconf_lines)
        else:
            vulkan_sdk_root = os.path.abspath(os.path.join(
                'third_party', 'VulkanSDK', os.listdir('third_party/VulkanSDK')[0], os_arch))
        vulkan_bin_path = os.path.join(vulkan_sdk_root, 'bin')
        vulkan_lib_path = os.path.join(vulkan_sdk_root, 'lib')
        if 'PATH' in env_cmake:
            env_cmake['PATH'] += f':{vulkan_bin_path}'
        else:
            env_cmake['PATH'] = vulkan_bin_path
        if 'LD_LIBRARY_PATH' in env_cmake:
            env_cmake['LD_LIBRARY_PATH'] += f':{vulkan_lib_path}'
        else:
            env_cmake['LD_LIBRARY_PATH'] = vulkan_lib_path
        env_cmake['VULKAN_SDK'] = vulkan_sdk_root
        glslang_validator_path = shutil.which('glslangValidator', path=vulkan_bin_path)


def compile_shader(header_name, shader_path, out_dir, shader_defines, recompile=False):
    shader_var_name = Path(header_name).stem.replace('.', '_')
    out_file_path = os.path.join(out_dir, header_name)
    if os.path.isfile(out_file_path) and not recompile:
        return
    process_args = [
        glslang_validator_path, '--target-env', 'vulkan1.2', '--vn', f'{shader_var_name}_shader_binary',
        shader_path, '-o', out_file_path
    ] + shader_defines
    subprocess.run(process_args, check=True)


cmake_exec, cmake_found = get_cmake_exec()
if cmake_found:
    tmp_path = 'tmp/fuchsia_radix_sort'
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    if IS_WINDOWS:
        radix_sort_lib_path = f'{tmp_path}/build/Release/vk-radix-sort.lib'
    else:
        radix_sort_lib_path = f'{tmp_path}/build/libvk-radix-sort.a'
    extra_objects.append(radix_sort_lib_path)
    if not os.path.isfile(radix_sort_lib_path):
        volk_header_path = 'third_party/sgl/src/Graphics/Vulkan/libs/volk'
        volk_header_path = os.path.abspath(volk_header_path)
        if IS_WINDOWS:
            volk_header_path = volk_header_path.replace('\\', '/')
        subprocess.run([
            cmake_exec, '-S', 'third_party/fuchsia_radix_sort', '-B', f'{tmp_path}/build',
            # '-DCMAKE_BUILD_TYPE=DEBUG',  # For debugging purposes.
            # '-DCMAKE_VERBOSE_MAKEFILE=ON',
            '-DCMAKE_BUILD_TYPE=Release',
            f'-DVOLK_INCLUDE_DIR={volk_header_path}',
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON'], env=env_cmake, check=True)
        subprocess.run([cmake_exec, '--build', f'{tmp_path}/build', '--config', 'Release'], check=True)
else:
    # fuchsia_radix_sort
    include_dirs += [
        'third_party/sgl/src/Graphics/Vulkan/libs/volk',
        'third_party/fuchsia_radix_sort',
        'third_party/fuchsia_radix_sort/lib',
        'tmp/fuchsia_radix_sort',
    ]
    source_files += [
        'third_party/fuchsia_radix_sort/lib/radix_sort_vk.c',
        'third_party/fuchsia_radix_sort/lib/target.c',
        'third_party/fuchsia_radix_sort/lib/target_requirements.c',
        'third_party/fuchsia_radix_sort/common/vk/assert.c',
        'third_party/fuchsia_radix_sort/common/vk/barrier.c',
        'third_party/fuchsia_radix_sort/common/util.c',
    ]
    source_files += find_all_sources_in_dir('third_party/fuchsia_radix_sort/lib/targets/amd')
    source_files += find_all_sources_in_dir('third_party/fuchsia_radix_sort/lib/targets/intel')
    source_files += find_all_sources_in_dir('third_party/fuchsia_radix_sort/lib/targets/nvidia')
    source_files += find_all_sources_in_dir('third_party/fuchsia_radix_sort/lib/targets/arm')
    defines += ['VK_NO_PROTOTYPES', 'VOLK_INCLUDE_DIR']

    tmp_shaders_path = 'tmp/fuchsia_radix_sort'
    Path(tmp_shaders_path).mkdir(parents=True, exist_ok=True)
    for support in ['noi64', 'i64']:
        for keyval in ['u32', 'u64']:
            a = f'{support}_{keyval}_'
            d = []
            if support == 'noi64':
                d.append('-DRS_DISABLE_SHADER_INT64')
            if keyval == 'u32':
                d.append('-DRS_KEYVAL_DWORDS=1')
            else:
                d.append('-DRS_KEYVAL_DWORDS=2')
            compile_shader(
                f'{a}init.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/init.comp',
                tmp_shaders_path, d, recompile=False)
            compile_shader(
                f'{a}fill.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/fill.comp',
                tmp_shaders_path, d, recompile=False)
            compile_shader(
                f'{a}histogram.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/histogram.comp',
                tmp_shaders_path, d, recompile=False)
            compile_shader(
                f'{a}prefix.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/prefix.comp',
                tmp_shaders_path, d, recompile=False)
            compile_shader(
                f'{a}scatter.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/scatter.comp',
                tmp_shaders_path, d, recompile=False)


# fTetWild, according to https://github.com/wildmeshing/fTetWild, relies on GMP or MPIR.
# MPIR is a fork of GMP with better Windows support. They recommend installation via:
# - homebrew on macOS: brew install gmp
# - Package manager on Linux: sudo apt-get install gmp
# - Conda on Windows: conda install -c conda-forge mpir
# As we currently have no way to check whether CMake will be able to find gmp, we will try to find it manually.
# Example location on Ubuntu 22.04: /lib/x86_64-linux-gnu/libgmp.so.10
def get_gmp_lib_found():
    if IS_WINDOWS:
        gmp_lib_name = 'mpir.dll'
    else:
        gmp_lib_name = 'libgmp.so'
    if not IS_WINDOWS:
        # We will use ldconfig to check if libgmp.so is available globally.
        # Attention: According to https://unix.stackexchange.com/questions/282199/find-out-if-library-is-in-path,
        # LD_LIBRARY_PATH is NOT used by ldconfig by default. Thus, we only search for globally installed libraries.
        ldconfig_proc = subprocess.Popen(['ldconfig', '-N', '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (ldconfig_output, ldconfig_err) = ldconfig_proc.communicate()
        # ldconfig_proc_status = ldconfig_proc.wait()  # Can be 1 for some reason...
        ldconfig_proc.wait()
        ldconfig_stdout_string = ldconfig_output.decode('utf-8')

        # Additionally check pkg-config, as the GMP might not be installed in the dev version.
        pkgconfig_gmp_proc_status = 1
        pkgconfig_exec = shutil.which('pkg-config')
        if pkgconfig_exec is not None:
            pkgconfig_gmp_proc = subprocess.Popen(
                [pkgconfig_exec, '--exists', 'gmp'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # (pkgconfig_gmp_output, pkgconfig_gmp_err) = pkgconfig_gmp_proc.communicate()
            pkgconfig_gmp_proc.communicate()
            pkgconfig_gmp_proc_status = pkgconfig_gmp_proc.wait()

        if pkgconfig_gmp_proc_status == 0 and gmp_lib_name in ldconfig_stdout_string:
            return True, gmp_lib_name, None

    # On Windows, we will assume mpir.dll has been installed using conda via MPIR.
    for search_path in sys.path:
        if not os.path.isdir(search_path):
            continue
        for walked_root, _, walked_files in os.walk(search_path):
            for walked_file in walked_files:
                if gmp_lib_name in walked_file:
                    return True, gmp_lib_name, os.path.join(walked_root, walked_file)

    return False, gmp_lib_name, None


support_ftetwild, gmp_lib_name, gmp_path = get_gmp_lib_found()
if support_ftetwild:
    if os.path.isdir('third_party/fTetWild'):
        # We delete the directory if gmp_lib_name could not be found in it.
        ftetwild_files = os.listdir('third_party/fTetWild')
        found_gmp_lib = False
        for ftetwild_file in ftetwild_files:
            if gmp_lib_name in ftetwild_file:
                found_gmp_lib = True
                break
        if not found_gmp_lib:
            shutil.rmtree('third_party/fTetWild')
    if not os.path.isdir('third_party/fTetWild'):
        if os.path.isdir('third_party/fTetWild-src'):
            rmtree_ex('third_party/fTetWild-src')
        subprocess.run(['git', 'clone', 'https://github.com/chrismile/fTetWild.git', 'third_party/fTetWild-src'])
        Path('third_party/fTetWild-src/build').mkdir(exist_ok=True)
        ftetwild_build_options = []
        if not IS_WINDOWS:
            ftetwild_build_options.append('-DCMAKE_BUILD_TYPE=Release')
        env_cmake_ftetwild = dict(os.environ)
        if IS_WINDOWS:
            gmp_base_path = os.path.dirname(os.path.dirname(gmp_path))
            env_cmake_ftetwild['GMP_INC'] = os.path.join(gmp_base_path, 'include')
            env_cmake_ftetwild['GMP_LIB'] = os.path.join(gmp_base_path, 'lib')
        subprocess.run(
            [cmake_exec, '-S', 'third_party/fTetWild-src', '-B', 'third_party/fTetWild-src/build']
            + ftetwild_build_options, env=env_cmake_ftetwild, check=True)
        subprocess.run([cmake_exec, '--build', f'third_party/fTetWild-src/build', '--config', 'Release'], check=True)
        Path('third_party/fTetWild').mkdir(exist_ok=True)
        if IS_WINDOWS:
            shutil.copy(
                'third_party/fTetWild-src/build/Release/FloatTetwild_bin.exe',
                'third_party/fTetWild/FloatTetwild_bin.exe')
        else:
            shutil.copy(
                'third_party/fTetWild-src/build/FloatTetwild_bin',
                'third_party/fTetWild/FloatTetwild_bin')
        if gmp_path is not None:
            shutil.copy(gmp_path, 'third_party/fTetWild/')
        elif not IS_WINDOWS:
            ldd_proc = subprocess.Popen(
                ['ldd', 'third_party/fTetWild-src/build/FloatTetwild_bin'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (ldd_output, ldd_err) = ldd_proc.communicate()
            ldd_proc_status = ldd_proc.wait()
            if ldd_proc_status == 0:
                ldd_stdout_string = ldd_output.decode('utf-8')
                for ldd_line in ldd_stdout_string.splitlines():
                    if gmp_lib_name in ldd_line:
                        ldd_line_split = ldd_line.split()
                        gmp_path_local = ldd_line_split[2]
                        gmp_lib_name_local = ldd_line_split[0]
                        shutil.copy(gmp_path_local, f'third_party/fTetWild/{gmp_lib_name_local}')
                        break
elif platform.machine() == 'x86_64' or platform.machine() == 'AMD64':
    # Download precompiled binaries if fTetWild build is not supported, but we are on x86_64 Windows or Linux.
    support_ftetwild = True
    ftetwild_version = '0.1.0'
    if IS_WINDOWS:
        target = 'x86_64-windows'
    else:
        target = f'{os_arch}-linux'
    ftetwild_dir = f'fTetWild-v{ftetwild_version}-{target}'
    ftetwild_url = f'https://github.com/chrismile/fTetWild/releases/download/v{ftetwild_version}/{ftetwild_dir}.zip'
    if not os.path.isdir(f'third_party/{ftetwild_dir}'):
        urllib.request.urlretrieve(ftetwild_url, f'third_party/{ftetwild_dir}.zip')
        with zipfile.ZipFile(f'third_party/{ftetwild_dir}.zip', 'r') as zip_ref:
            zip_ref.extractall(f'third_party/{ftetwild_dir}')
        if not IS_WINDOWS:
            subprocess.run(['chmod', '+x', f'third_party/{ftetwild_dir}/bin/FloatTetwild_bin'], check=True)
    if not os.path.isdir(f'third_party/fTetWild'):
        Path('third_party/fTetWild').mkdir(exist_ok=True)
        for ftetwild_file in os.listdir(f'third_party/{ftetwild_dir}/bin'):
            shutil.copy(os.path.join(f'third_party/{ftetwild_dir}/bin', ftetwild_file), 'third_party/fTetWild/')


support_tetgen = True
if support_tetgen:
    tetgen_version = '1.6.0'
    if IS_WINDOWS:
        target = 'x86_64-windows-gnu'
    else:
        target = f'{os_arch}-linux'
    tetgen_dir = f'tetgen-v{tetgen_version}-{target}'
    tetgen_url = f'https://github.com/chrismile/tetgen/releases/download/v{tetgen_version}/{tetgen_dir}.zip'
    if not os.path.isdir(f'third_party/{tetgen_dir}'):
        urllib.request.urlretrieve(tetgen_url, f'third_party/{tetgen_dir}.zip')
        with zipfile.ZipFile(f'third_party/{tetgen_dir}.zip', 'r') as zip_ref:
            zip_ref.extractall(f'third_party/{tetgen_dir}')
        if not IS_WINDOWS:
            subprocess.run(['chmod', '+x', f'third_party/{tetgen_dir}/bin/tetgen'], check=True)


data_files_all.append(('.', data_files))


def update_data_files_recursive(data_files_all, directory):
    files_in_directory = []
    for filename in os.listdir(directory):
        abs_file = directory + "/" + filename
        if os.path.isdir(abs_file):
            update_data_files_recursive(data_files_all, abs_file)
        else:
            files_in_directory.append(abs_file)
    if len(files_in_directory) > 0:
        data_files_all.append((directory, files_in_directory))


update_data_files_recursive(data_files_all, 'docs')
update_data_files_recursive(data_files_all, 'Data/Shaders')
update_data_files_recursive(data_files_all, 'Data/TransferFunctions')

for define in defines:
    if IS_WINDOWS:
        if len(define) == 1:
            extra_compile_args.append('/D')
            extra_compile_args.append(f'{define[0]}')
        else:
            extra_compile_args.append('/D')
            extra_compile_args.append(f'{define[0]}={define[1]}')
    else:
        if len(define) == 1:
            extra_compile_args.append(f'-D{define[0]}')
        else:
            extra_compile_args.append(f'-D{define[0]}={define[1]}')

uses_pip = \
    ('_' in os.environ and (os.environ['_'].endswith('pip') or os.environ['_'].endswith('pip3'))) \
    or 'PIP_BUILD_TRACKER' in os.environ
if os.path.exists('difftetvr'):
    shutil.rmtree('difftetvr')
if uses_pip:
    Path('difftetvr/Data').mkdir(parents=True, exist_ok=True)
    shutil.copy('src/Module/difftetvr.pyi', 'difftetvr/__init__.pyi')
    shutil.copy('src/Module/pyutils.py', 'difftetvr/pyutils.py')
    shutil.copy('LICENSE', 'difftetvr/LICENSE')
    shutil.copytree('docs', 'difftetvr/docs')
    shutil.copytree('Data/Shaders', 'difftetvr/Data/Shaders')
    shutil.copytree('Data/TransferFunctions', 'difftetvr/Data/TransferFunctions')
    pkg_data = ['**/LICENSE']
    if IS_WINDOWS:
        if support_ftetwild or support_tetgen:
            pkg_data.append('**/*.exe')
        if support_ftetwild:
            shutil.copy('third_party/fTetWild/FloatTetwild_bin.exe', 'difftetvr/FloatTetwild_bin.exe')
        if support_tetgen:
            shutil.copy(f'third_party/{tetgen_dir}/bin/tetgen.exe', 'difftetvr/tetgen.exe')
    else:
        if support_ftetwild:
            pkg_data.append('**/FloatTetwild_bin')
            shutil.copy('third_party/fTetWild/FloatTetwild_bin', 'difftetvr/FloatTetwild_bin')
            patchelf_exec = shutil.which('patchelf')
            if patchelf_exec is not None:
                subprocess.run([patchelf_exec, '--set-rpath', '\'$ORIGIN\'', 'difftetvr/FloatTetwild_bin'], check=True)
        if support_tetgen:
            pkg_data.append('**/tetgen')
            shutil.copy(f'third_party/{tetgen_dir}/bin/tetgen', 'difftetvr/tetgen')
    if support_ftetwild:
        files_in_ftetwild_dir = os.listdir('third_party/fTetWild')
        for file_in_ftetwild_dir in files_in_ftetwild_dir:
            if gmp_lib_name in file_in_ftetwild_dir:
                gmp_lib_name_local = file_in_ftetwild_dir
                if IS_WINDOWS:
                    pkg_data.append(f'**/{gmp_lib_name_local}')
                else:
                    pkg_data.append(f'**/{gmp_lib_name_local}')
                shutil.copy(f'third_party/fTetWild/{gmp_lib_name_local}', f'difftetvr/{gmp_lib_name_local}')
                break
    ext_modules = [
        TorchExtension(
            'difftetvr.difftetvr',
            source_files,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_objects=extra_objects
        )
    ]
    dist = Distribution(attrs={'name': 'difftetvr', 'version': '0.0.0', 'ext_modules': ext_modules})
    bdist_egg_cmd = dist.get_command_obj('bdist_egg')
    build_cmd = bdist_egg_cmd.get_finalized_command('build_ext')
    difftetvr_so_file = ''
    for ext in build_cmd.extensions:
        fullname = build_cmd.get_ext_fullname(ext.name)
        filename = build_cmd.get_ext_filename(fullname)
        difftetvr_so_file = os.path.basename(filename)
    with open('difftetvr/__init__.py', 'w') as file:
        file.write('import torch\n\n')
        file.write('def __bootstrap__():\n')
        file.write('    global __bootstrap__, __loader__, __file__\n')
        file.write('    import sys, pkg_resources, importlib.util\n')
        file.write(f'    __file__ = pkg_resources.resource_filename(__name__, \'{difftetvr_so_file}\')\n')
        file.write('    __loader__ = None; del __bootstrap__, __loader__\n')
        file.write('    spec = importlib.util.spec_from_file_location(__name__,__file__)\n')
        file.write('    mod = importlib.util.module_from_spec(spec)\n')
        file.write('    spec.loader.exec_module(mod)\n')
        file.write('__bootstrap__()\n')
    setup(
        name='difftetvr',
        author='Christoph Neuhauser',
        ext_modules=ext_modules,
        packages=find_packages(include=['difftetvr', 'difftetvr.*']),
        package_data={'difftetvr': ['**/*.py', '**/*.pyi', '**/*.md', '**/*.txt', '**/*.xml', '**/*.glsl'] + pkg_data},
        #include_package_data=True,
        cmdclass={
            'build_ext': BuildExtension,
            'egg_info': EggInfoInstallLicense
        },
        license_files=('LICENSE',),
        include_dirs=include_dirs
    )
else:
    setup(
        name='difftetvr',
        author='Christoph Neuhauser',
        ext_modules=[
            TorchExtension(
                'difftetvr',
                source_files,
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                extra_objects=extra_objects
            )
        ],
        data_files=data_files_all,
        cmdclass={
            'build_ext': BuildExtension,
            'egg_info': EggInfoInstallLicense
        },
        license_files=('LICENSE',),
        include_dirs=include_dirs
    )
