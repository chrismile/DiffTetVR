# BSD 2-Clause License
#
# Copyright (c) 2025, Christoph Neuhauser
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

# Fetches the submodules either via a call to Git or by using the entries in .gitmodules.
# NOTE: The .gitmodules file misses information about the specific commit of submodules.

import os
import sys
import shutil
import subprocess


def fetch_submodules_from_gitmodules_file(proj_dir):
    # First, parse the .gitmodules file.
    with open(os.path.join(proj_dir, '.gitmodules')) as gitmodules_file:
        submodule_entries = []
        current_entry = None
        for line in gitmodules_file:
            line = line.strip()
            if line.startswith('[submodule '):
                if current_entry is not None:
                    submodule_entries.append(current_entry)
                current_entry = {}
            elif len(line) > 0:
                assignment_pos = line.find('=')
                key = line[:assignment_pos].rstrip()
                value = line[assignment_pos + 1:].lstrip()
                current_entry[key] = value
        if current_entry is not None:
            submodule_entries.append(current_entry)

    # Secondly, download all submodules to the right paths.
    for submodule_entry in submodule_entries:
        if 'branch' in submodule_entry:
            additional_args = ['-b', submodule_entry['branch'], '--single-branch']
        else:
            additional_args = []
        submodule_abs_path = os.path.join(proj_dir, submodule_entry['path'])
        subprocess.run(['git', 'clone'] + additional_args + [submodule_entry['url'], submodule_abs_path])


def main():
    if len(sys.argv) == 2:
        proj_dir = sys.argv[1]
    else:
        proj_dir = os.getcwd()
    if not os.path.exists(proj_dir):
        raise RuntimeError(f'Project directory "{proj_dir}" does not exist.')
    if shutil.which('git') is None:
        raise RuntimeError(f'Could not find command "git".')
    if os.path.isdir(os.path.join(proj_dir, '.git')):
        subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'])
    elif not os.path.isfile(os.path.join(proj_dir, '.gitmodules')):
        raise RuntimeError(
            f'Neither the directory ".git" nor the file ".gitmodules" exists in the directory "{proj_dir}".')
    else:
        fetch_submodules_from_gitmodules_file(proj_dir)


if __name__ == '__main__':
    main()
