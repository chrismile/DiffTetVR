import sys
import os
from setuptools import setup
from setuptools.command.egg_info import egg_info
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, IS_WINDOWS, IS_MACOS

extra_compile_args = []
if IS_WINDOWS:
    extra_compile_args.append('/std:c++17')
    extra_compile_args.append('/openmp')
elif IS_MACOS:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp=libomp')
else:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp')

class EggInfoInstallLicense(egg_info):
    def run(self):
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)
        egg_info.run(self)

# Files TODO:
# - Vulkan headers from sgl -> include_dirs
# - Device.cpp, Instance.cpp, Logfile.cpp, Dialog.cpp, Execute.cpp, FileUtils.cpp, StringUtils.cpp, InteropCuda.cpp,
#   Memory.cpp, Shader.cpp, ShaderManager.cpp, IncluderInterface.cpp, ReflectHelpers.cpp, Swapchain.cpp, SyncObjects.cpp
# Everything in Graphics/Vulkan/Buffers, Graphics/Vulkan/Image, Graphics/Vulkan/Render, Graphics/Vulkan/Shader
# - Vulkan/libs/volk/volk.c, Vulkan/libs/SPIRV-Refllect/spirv_reflect.c
# For Instance.cpp: FileUtils::get()->getAppName().c_str() is used.
# For ShaderManager.cpp: Uses sgl::AppSettings::get()->getDataDirectory()

def find_all_sources_in_dir(root_dir):
    source_files = []
    for root, subdirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.cpp') or filename.endswith('.c'):
                source_files.append(root + "/" + filename)
    return source_files

sgl_sources = [ 'third_party/sgl/src/Graphics/Vulkan/Utils/Device.cpp' ]
sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Buffers')
sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Image')
sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Render')
sgl_sources += find_all_sources_in_dir('third_party/sgl/src/Graphics/Vulkan/Shader')

setup(
    name='difftetvr',
    author='Christoph Neuhauser',
    ext_modules=[
        CUDAExtension('difftetvr', [
            'src/Module/Module.cpp',
        ] + sgl_sources, libraries=['nvrtc'], extra_compile_args=extra_compile_args)
    ],
    data_files=[
        ( '.', ['src/Module/difftetvr.pyi'] )
    ],
    cmdclass={
        'build_ext': BuildExtension,
        'egg_info': EggInfoInstallLicense
    },
    license_files = ('LICENSE',),
    include_dirs=['third_party/glm', 'third_party/sgl/src', 'third_party/sgl/src/Graphics/Vulkan/libs/Vulkan-Headers']
)
