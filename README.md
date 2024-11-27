# DiffTetVR

Differentiable volume renderer for tetrahedral meshes.


## Install with setuptools

To install the library as a Python module, the following command must be called in the repository directory.

```sh
pip install .
```

If the package should be installed in a Conda environment, activate the corresponding environment first as follows.

```sh
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate <env-name>
```

Necessary packages can be installed, e.g., via conda:

```sh
conda install numpy pillow conda-forge::openexr-python
```

For more details on how to install PyTorch, see: https://pytorch.org/get-started/locally/


## CUDA Detection

If setup.py is not able to find your CUDA installation on Linux, add the following lines to the end of `$HOME/.profile`
and log out of and then back into your user account.
`cuda-12.4` needs to be adapted depending on the CUDA version installed.

```sh
export CPATH=/usr/local/cuda-12.4/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
```
