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
import glob
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen
import setuptools
from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.dist import Distribution
from setuptools.command import bdist_egg
from torch.utils.cpp_extension import include_paths, library_paths, BuildExtension, IS_WINDOWS, IS_HIP_EXTENSION, ROCM_VERSION

extra_compile_args = []
if IS_WINDOWS:
    extra_compile_args.append('/std:c++17')
    extra_compile_args.append('/openmp')
else:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp')
    # '-fpermissive' is necessary for compiling C code with torch.utils.cpp_extension.BuildExtension.
    #extra_compile_args.append('-fpermissive')


class EggInfoInstallLicense(egg_info):
    def run(self):
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)
        egg_info.run(self)


def TorchExtension(name, sources, *args, **kwargs):
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths(cuda=True)
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    if IS_HIP_EXTENSION:
        assert ROCM_VERSION is not None
        libraries.append('amdhip64' if ROCM_VERSION >= (3, 5) else 'hip_hcc')
        libraries.append('c10_hip')
        libraries.append('torch_hip')
    else:
        libraries.append('cudart')
        libraries.append('c10_cuda')
        libraries.append('torch_cuda')
    kwargs['libraries'] = libraries

    return setuptools.Extension(name, sources, *args, **kwargs)


def find_all_sources_in_dir(root_dir):
    source_files = []
    for root, subdirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.cpp') or filename.endswith('.cc') or filename.endswith('.c'):
                source_files.append(root + "/" + filename)
    return source_files


#sgl_sources = [ 'third_party/sgl/src/Graphics/Vulkan/Utils/Device.cpp' ]
#sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Buffers')
#sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Image')
#sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Render')
#sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Shader')

if not os.path.exists('third_party/sgl'):
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
    # 'third_party/fuchsia_radix_sort',
    # 'third_party/fuchsia_radix_sort/lib',
    'third_party/fuchsia_radix_sort/include',
    # 'tmp/fuchsia_radix_sort',
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


# fuchsia_radix_sort
#source_files += [
#    'third_party/fuchsia_radix_sort/lib/radix_sort_vk.c',
#    'third_party/fuchsia_radix_sort/lib/target.c',
#    'third_party/fuchsia_radix_sort/lib/target_requirements.c',
#    'third_party/fuchsia_radix_sort/common/vk/assert.c',
#    'third_party/fuchsia_radix_sort/common/vk/barrier.c',
#    'third_party/fuchsia_radix_sort/common/util.c',
#]
#source_files += find_all_sources_in_dir('third_party/fuchsia_radix_sort/lib/targets/amd')
#source_files += find_all_sources_in_dir('third_party/fuchsia_radix_sort/lib/targets/intel')
#source_files += find_all_sources_in_dir('third_party/fuchsia_radix_sort/lib/targets/nvidia')
#source_files += find_all_sources_in_dir('third_party/fuchsia_radix_sort/lib/targets/arm')
#
#
#def compile_shader(header_name, shader_path, out_dir, shader_defines):
#    shader_var_name = Path(header_name).stem.replace('.', '_')
#    out_file_path = os.path.join(out_dir, header_name)
#    if os.path.isfile(out_file_path):
#        return
#    process_args = [
#        'glslangValidator', '--target-env', 'vulkan1.2', '--vn', f'{shader_var_name}_shader_binary',
#        shader_path, '-o', out_file_path
#    ] + shader_defines
#    subprocess.run(process_args, check=True)
#
#
#tmp_shaders_path = 'tmp/fuchsia_radix_sort'
#Path(tmp_shaders_path).mkdir(parents=True, exist_ok=True)
#for support in ['noi64', 'i64']:
#    for keyval in ['u32', 'u64']:
#        a = f'{support}_{keyval}_'
#        d = []
#        if support == 'noi64':
#            d.append('-DRS_DISABLE_SHADER_INT64')
#        if keyval == 'u32':
#            d.append('-DRS_KEYVAL_DWORDS=1')
#        else:
#            d.append('-DRS_KEYVAL_DWORDS=2')
#        compile_shader(
#            f'{a}init.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/init.comp', tmp_shaders_path, d)
#        compile_shader(
#            f'{a}fill.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/fill.comp', tmp_shaders_path, d)
#        compile_shader(
#            f'{a}histogram.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/histogram.comp', tmp_shaders_path, d)
#        compile_shader(
#            f'{a}prefix.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/prefix.comp', tmp_shaders_path, d)
#        compile_shader(
#            f'{a}scatter.comp.h', 'third_party/fuchsia_radix_sort/lib/shaders/scatter.comp', tmp_shaders_path, d)

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
    # TODO: Test on Windows.
    # According to https://learn.microsoft.com/en-us/windows/win32/api/shlwapi/nf-shlwapi-pathremovefilespecw,
    # shlwapi.lib and shlwapi.dll both exist. Maybe this should rather be a extra_objects file?
    libraries.append('shlwapi')
    libraries.append('shell32')
    libraries.append('user32')
    defines.append(('GLSLANG_OSINCLUDE_WIN32', ''))
else:
    defines.append(('DLL_OBJECT', ''))
    #extra_compile_args.append('-O0')  # For debugging tests.
    #extra_compile_args.append('-ggdb')  # For debugging tests.
    libraries.append('dl')
    defines.append(('GLSLANG_OSINCLUDE_UNIX', ''))


# TODO: Add support for not using CUDA.
defines.append(('SUPPORT_CUDA_INTEROP',))
source_files.append('third_party/sgl/src/Graphics/Vulkan/Utils/InteropCuda.cpp')


tmp_path = 'tmp/fuchsia_radix_sort'
Path(tmp_path).mkdir(parents=True, exist_ok=True)
if IS_WINDOWS:
    radix_sort_lib_path = f'{tmp_path}/build/Release/vk-radix-sort.lib'
else:
    radix_sort_lib_path = f'{tmp_path}/build/libvk-radix-sort.a'
extra_objects.append(radix_sort_lib_path)
#if IS_WINDOWS:
#    libraries.append('vulkan-1')
#else:
#    libraries.append('vulkan')
if not os.path.isfile(radix_sort_lib_path):
    volk_header_path = 'third_party/sgl/src/Graphics/Vulkan/libs/volk'
    volk_header_path = os.path.abspath(volk_header_path)
    cmake_exec = 'cmake'
    if IS_WINDOWS:
        volk_header_path = volk_header_path.replace('\\', '/')
        # CMake on Windows is usually not in the PATH. If it is not found, try to use the default location.
        cmake_exec = shutil.which('cmake')
        cmake_default_path = 'C:\\Program Files\\CMake\\bin\\cmake.exe'
        if cmake_exec is None and os.path.isfile(cmake_default_path):
            cmake_exec = cmake_default_path
    subprocess.run([
        cmake_exec, '-S', 'third_party/fuchsia_radix_sort', '-B', f'{tmp_path}/build',
        # '-DCMAKE_BUILD_TYPE=DEBUG',  # For debugging purposes.
        # '-DCMAKE_VERBOSE_MAKEFILE=ON',
        '-DCMAKE_BUILD_TYPE=Release',
        f'-DVOLK_INCLUDE_DIR={volk_header_path}',
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON'], check=True)
    subprocess.run([cmake_exec, '--build', f'{tmp_path}/build', '--config', 'Release'], check=True)


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
if uses_pip:
    if os.path.exists('difftetvr'):
        shutil.rmtree('difftetvr')
    Path('difftetvr/Data').mkdir(parents=True, exist_ok=True)
    shutil.copy('src/Module/difftetvr.pyi', 'difftetvr/__init__.pyi')
    shutil.copy('src/Module/pyutils.py', 'difftetvr/pyutils.py')
    shutil.copytree('docs', 'difftetvr/docs')
    shutil.copytree('Data/Shaders', 'difftetvr/Data/Shaders')
    shutil.copytree('Data/TransferFunctions', 'difftetvr/Data/TransferFunctions')
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
        package_data={'difftetvr': ['**/*.py', '**/*.pyi', '**/*.md', '**/*.txt', '**/*.xml', '**/*.glsl']},
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
