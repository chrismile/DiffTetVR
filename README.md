# DiffTetVR
Differentiable volume renderer for tetrahedral meshes.

TODO:

This library implements correlation metrics using OpenMP on the CPU and CUDA on the GPU for use in PyTorch.

Currently, the computation of the following correlation metrics is supported.
- Pearson correlation coefficient.
- Spearman rank correlation coefficient.
- Kendall rank correlation coefficient (aka. Kendall's tau).
- A binned mutual information estimator.
- The mutual information estimator by Kraskov et al. as introduced in
> Alexander Kraskov, Harald Stögbauer, and Peter Grassberger: Estimating mutual information.
Phys. Rev. E, 69:066138, June 2004, https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138


## Install with setuptools

To install the library as a Python module, the following command must be called in the repository directory.

```sh
python setup.py install
```

If it should be installed in a Conda environment, activate the corresponding environment first as follows.

```sh
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate <env-name>
```


## CUDA Detection

If setup.py is not able to find your CUDA installation on Linux, add the following lines to the end of `$HOME/.profile`
and log out of and then back into your user account.
`cuda-11.5` needs to be adapted depending on the CUDA version installed.

```sh
export CPATH=/usr/local/cuda-11.5/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.5/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.5/bin:$PATH
```


## CMake Support

This library can also be built with CMake if it should not be installed globally in a Python environment.
In order for CMake to find your PyTorch/LibTorch installation, the following command can be used.
If you are using a virtual environment with pip or Conda, you may need to first activate the environment.

```sh
cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
```
