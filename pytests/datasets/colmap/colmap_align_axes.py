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

import numpy as np
try:
    import open3d as o3d  # pip install open3d
    can_use_o3d = True
except ImportError:
    o3d = None
    can_use_o3d = False


def normalize(v):
    norm = np.linalg.norm(v)
    if abs(norm) <= 1e-9:
       return v
    return v / norm


def align_axes_point_cloud(pcd, points):
    # Do PCA anaylsis of covariance matrix of point cloud.
    mean, cov = pcd.compute_mean_and_covariance()
    eigenvals, eigenvecs = np.linalg.eig(cov)
    sorted_indices = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[sorted_indices]
    #eigenvecs = eigenvecs[:, sorted_indices]
    axes = np.array([normalize(eigenvecs[:, i]) for i in range(3)])
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] = -axes[:, 2]

    # 'axes' transforms from world space to point cloud space.
    base_transform = np.linalg.inv(axes)
    points_aligned = np.dot(points, np.transpose(base_transform))
    return points_aligned, base_transform

