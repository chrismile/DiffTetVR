# DiffTetVR

DiffTetVR is a differentiable direct volume renderer for tetrahedral meshes.


## Building and running the program

The program consists of a visualization frontend and a Python module.

### Visualization frontend

Currently, there are multiple ways to compile the program frontend:
- Linux: Using the system package manager to install all dependencies (tested: apt on Ubuntu, pacman on Arch Linux, dnf/yum on Fedora).
- Linux & Windows: Installing all dependencies using [vcpkg](https://github.com/microsoft/vcpkg)  (by using the flag `./build.sh --vcpkg` on Linux or `build-msvc.bat` on Windows).
- Windows: Using [MSYS2](https://www.msys2.org/) to install all dependencies (by using `./build.sh` in a MSYS2 shell).
- Linux: Installing all dependencies with [conda](https://docs.conda.io/en/latest/) (by using the flag `./build.sh --conda`).
- Linux: Installing all dependencies with [Nix](https://nixos.org/) (by invoking `./build.sh` after calling `nix-shell`).

A build script `build.sh` is available in the project root directory that builds the application using the system
package manager on Linux and MSYS2 on Windows. Please download and install MSYS2 from https://www.msys2.org/ if you wish
to use this build script and run the script from an MSYS2 shell.
Alternatively, we recommend to use vcpkg if the users wants to compile the application with Microsoft Visual Studio.
A build script using MSVC and vcpkg, `build-msvc.bat`, is available in the project root directory.
Please note that the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) needs to be installed beforehand if using
Microsoft Visual Studio for compilation.

Under `Data/CloudDataSets/datasets.json`, loadable data sets can be specified. Additionally, the user can also open
arbitrary data sets using a file explorer via "File > Open Dataset..." (or using Ctrl+O).

Below, an example for a `Data/DataSets/datasets.json` file can be found.

```json
{
    "datasets": [
        { "name" : "Icosahedron", "filename": "mesh_ico.bintet" },
        { "name" : "Turbine", "filename": "turbine.ovm" }
    ]
}
```

These files then appear with their specified name in the menu "File > Datasets". All paths must be specified relative to
the folder `Data/DataSets/` (unless they are global, like `C:/path/file.bintet` or `/path/file.bintet`).

Supported formats currently are:
- .bintet and .txt files, which store cell indices, vertex positions and vertex colors in a binary or text-based format.
  For more information on this custom format, please refer to src/Tet/Loaders/{Bin,Txt}TetLoader.cpp.
- .ovm and .ovmb files, which are native to the library
  [OpenVolumeMesh](https://www.graphics.rwth-aachen.de/software/openvolumemesh/).
- .vtk files, which are also supported via OpenVolumeMesh.
- Gmsh .msh files (for more details see https://victorsndvg.github.io/FEconv/formats/gmshmsh.xhtml).
- MEDIT .mesh files (for more details see https://victorsndvg.github.io/FEconv/formats/gmshmsh.xhtml).

### Python module

Please note that the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) needs to be installed beforehand on
Windows.

To install the library as a Python module, first install a version of PyTorch matching your GPU.
Then, the following command must be called in the repository directory.
It seems like `--no-build-isolation` is only necessary when installing in a Python venv, not a conda environment.

```sh
pip install --no-build-isolation .
```

If the package should be installed in a conda environment, activate the corresponding environment first as follows.

```sh
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate <env-name>
```

PyTorch is a necessary dependency that needs to be installed in the active Python environment.
All packages necessary for DiffTetVR and the accompanying test scripts can be installed, e.g., via conda:

```sh
export CONDA_ALWAYS_YES="true"
conda create -n diffdvr python=3.12
conda activate diffdvr
pip install numpy sympy numba matplotlib tqdm scikit-image tensorboard opencv-python openexr setuptools
# Alternatively to the command above: Install the dependencies via conda.  
# conda install numpy sympy numba matplotlib tqdm scikit-image conda-forge::tensorboard conda-forge::opencv conda-forge::openexr-python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Examples how to use the Python module can be found in the directory `pytests/`.

For more details on how to install PyTorch, see: https://pytorch.org/get-started/locally/


### CUDA Detection

If setup.py is not able to find your CUDA installation on Linux, add the following lines to the end of `$HOME/.profile`
and log out of and then back into your user account.
`cuda-12.6` needs to be adapted depending on the CUDA version installed.

```sh
export CPATH=/usr/local/cuda-12.6/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
```

### Intel GPU Support

DiffTetVR has WIP support for Intel GPUs via the Python module. Please follow the steps below to install the Python
module. Please note that on Windows, only the cmd.exe-based "Anaconda Prompt" works, but **NOT** the "Anaconda
Powershell Prompt", as it does not seem to propagate the environment variables from the vars.bat scripts to Python.

- Step 1: Install the drivers and the "Intel Deep Learning Essentials" by following the steps at
  https://pytorch.org/docs/main/notes/get_start_xpu.html under the category "Software Prerequisite".
  Alternatively, it can be obtained from:
  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials
- Step 2: Open the "Anaconda Prompt" (Windows) or a terminal and activate conda (Linux).
- Step 3: Run the following commands (replace "2025.1" by the used oneAPI version).

```sh
# Windows (Intel Deep Learning Essentials/oneAPI only)
"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.1\env\vars.bat"
"C:\Program Files (x86)\Intel\oneAPI\ocloc\2025.1\env\vars.bat"
# Windows (release from https://github.com/intel/llvm/releases only; adapt paths where necessary)
set "CMPLR_ROOT=C:\Users\cneuhaus\Programming\Tools\sycl_windows_2025-04-25"
set "ONEAPI_ROOT=%CMPLR_ROOT%"
set "PATH=%CMPLR_ROOT%\bin;%PATH%"
set "CPATH=%CMPLR_ROOT%\include;%CPATH%"
set "INCLUDE=%CMPLR_ROOT%\include;%INCLUDE%"
set "LIB=%CMPLR_ROOT%\lib\clang\21\lib\windows;%CMPLR_ROOT%\lib;%LIB%"
call "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" amd64
# Windows (all)
set DISTUTILS_USE_SDK=1
set VSLANG=1033
set KMP_DUPLICATE_LIB_OK=TRUE
# Linux:
source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/umf/latest/env/vars.sh
source /opt/intel/oneapi/pti/latest/env/vars.sh
# All:
conda create -n diffdvr python=3.12
conda activate diffdvr
pip3 install numpy sympy numba matplotlib tqdm scikit-image tensorboard opencv-python openexr setuptools
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
cd <path-to-difftetvr>
pip install .
```

**TODOs**:
- It should be investigated why KMP_DUPLICATE_LIB_OK=TRUE is necessary on Windows.
  On Windows, I get the following error message when using the PyTorch module with SYCL:
  
  "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
  OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is
  dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that
  only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime
  in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable
  KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently
  produce incorrect results. For more information, please see http://www.intel.com/software/products/support/."
  
  While KMP_DUPLICATE_LIB_OK=TRUE seems to resolve the problem, it might cause some issues.
  Turning off OpenMP support in setup.py doesn't seem to resolve it, so it is probably an oneAPI vs PyTorch problem?
  Example of two files on my system:
  - C:\Users\chris\miniconda3\pkgs\intel-openmp-2023.1.0-h59b6b97_46320\Library\bin\libiomp5md.dll
  - C:\Program Files (x86)\Intel\oneAPI\compiler\2025.1\bin\libiomp5md.dll


### AMD GPU Support

DiffTetVR has WIP support for AMD GPUs via the Python module. However, AMD (as of 2025-03-30) only supports PyTorch
via ROCm on Linux and on Windows via WSL. I have not been able to get Vulkan-HIP interop to work under WSL so far.

```sh
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/6.3.4/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb
sudo apt install ./amdgpu-install_6.3.60304-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms rocm
# https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html
amdgpu-install -y --usecase=wsl,rocm,graphics --no-dkms

conda create -n diffdvr python=3.12
conda activate diffdvr
conda install numpy sympy numba matplotlib tqdm scikit-image conda-forge::tensorboard conda-forge::opencv conda-forge::openexr-python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
```


## How to report bugs

When [reporting a bug](https://github.com/chrismile/DiffTetVR/issues), please also attach the logfile generated by this
program. Below, the location of the logfile on different operating systems can be found.

- Linux: `~/.config/difftetvr/Logfile.html`
- Windows: `C:/Users/<USER>/AppData/Roaming/DiffTetVR/Logfile.html`
