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

import math
import random

import difftetvr as d
import numpy as np
from scipy.spatial.transform import Rotation as Rotation


def vec_cross(v0, v1):
    return np.array([
        v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0],
    ])


def vec_length(p):
    return np.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])


def vec_normalize(p):
    l = vec_length(p)
    return np.array([p[0] / l, p[1] / l, p[2] / l])


def matrix_translation(t):
    return np.array([
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])


# https://stackoverflow.com/questions/1556260/convert-quaternion-rotation-to-rotation-matrix
def matrix_quaternion(q_vec):
    # We assume q_vec is already normalized and has the format (w, x, y, z).
    # Otherwise: q_vec /= np.linalg.norm(q_vec)
    qw = q_vec[0]
    qx = q_vec[1]
    qy = q_vec[2]
    qz = q_vec[3]
    return np.array(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qw * qz,
                2.0 * qz * qx + 2.0 * qw * qy,
                0.0
            ],
            [
                2.0 * qx * qy + 2.0 * qw * qz,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qw * qx,
                0.0
            ],
            [
                2.0 * qz * qx - 2.0 * qw * qy,
                2.0 * qy * qz + 2.0 * qw * qx,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
                0.0
            ],
            [0.0, 0.0, 0.0, 1.0]
        ]
    )


def convert_focal_length_to_fov(focal_length, pixel_size):
    return 2.0 * math.atan(0.5 * pixel_size / focal_length)


def sample_view_matrix_circle(aabb, uniform_r=False, r_min=None, r_max=None):
    global_up = np.array([0.0, 1.0, 0.0])
    theta = 2.0 * np.pi * random.random()
    phi = np.arccos(1.0 - 2.0 * random.random())
    if uniform_r:
        r = 1.0
    else:
        r_total = 0.5 * vec_length(np.array([aabb.max.x - aabb.min.x, aabb.max.y - aabb.min.y, aabb.max.z - aabb.min.z]))
        #if test_case == 'Cloud' or test_case == 'Cloud Fog':
        #    r = r_total * 1.7
        # main_energy.py: r = random.uniform(r_total * 1.35, r_total * 1.7)
        if r_min is None:
            r_min = 1.35
        if r_max is None:
            r_max = 1.8
        r = random.uniform(r_total * r_min, r_total * r_max)
    camera_position = np.array([r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)])
    camera_forward = vec_normalize(camera_position)
    camera_right = vec_normalize(vec_cross(global_up, camera_forward))
    camera_up = vec_normalize(vec_cross(camera_forward, camera_right))
    rotation_matrix = np.empty((4, 4))
    for i in range(4):
        rotation_matrix[i, 0] = camera_right[i] if i < 3 else 0.0
        rotation_matrix[i, 1] = camera_up[i] if i < 3 else 0.0
        rotation_matrix[i, 2] = camera_forward[i] if i < 3 else 0.0
        rotation_matrix[i, 3] = 0.0 if i < 3 else 1.0
    inverse_view_matrix = matrix_translation(camera_position).dot(rotation_matrix)
    view_matrix = np.linalg.inv(inverse_view_matrix)
    view_matrix_array = np.empty(16)
    for i in range(4):
        for j in range(4):
            view_matrix_array[i * 4 + j] = view_matrix[j, i]
    return view_matrix_array, view_matrix, inverse_view_matrix


def jitter_direction(camera_forward, jitter_rad):
    v1 = np.array([0.0, 0.0, 1.0])
    v2 = camera_forward
    # For more details see:
    # - https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    # - https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    q_xyz = np.cross(v1, v2)
    q_w = np.inner(v1, v1) * np.inner(v2, v2) + np.dot(v1, v2)
    transform = Rotation.from_quat(np.array([q_xyz[0], q_xyz[1], q_xyz[2], q_w]))
    phi = random.uniform(0.0, 2.0 * np.pi)
    theta = random.uniform(0, jitter_rad)
    vec_spherical = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    vec_new = transform.apply(vec_spherical)
    return vec_new


def sample_view_matrix_box(aabb):
    global_up = np.array([0.0, 1.0, 0.0])
    rx = 0.5 * (aabb.max.x - aabb.min.x)
    ry = 0.5 * (aabb.max.y - aabb.min.y)
    rz = 0.5 * (aabb.max.z - aabb.min.z)
    radii_sorted = sorted([rx, ry, rz])
    r_base = np.sqrt(radii_sorted[0] ** 2 + radii_sorted[1] ** 2)
    #if random.randint(0, 1) == 0:
    #    r = random.uniform(1.5 * r_base, r_base * 3.0)
    #else:
    #    r = random.uniform(0.5 * r_base, r_base * 1.5)  # TODO
    r = random.uniform(1.5 * r_base, r_base * 3.0)
    h = 2 * radii_sorted[2]
    hi = 0 if rx >= ry and rx >= rz else (1 if ry >= rz else 2)
    r0i = 0 if hi != 0 else 1
    r1i = 2 if hi != 2 else 1
    area_sphere = 4 * (r**2) * np.pi
    area_cylinder = 2 * np.pi * r * h
    area_total = area_sphere + area_cylinder
    pos_rand = random.random() * area_total
    theta = 2.0 * np.pi * random.random()
    camera_position = np.zeros(3)
    if pos_rand < area_cylinder:
        h_pos = (random.random() - 0.5) * h
        camera_position[r0i] = r * np.cos(theta)
        camera_position[r1i] = r * np.sin(theta)
        camera_forward = vec_normalize(camera_position)
        camera_position[hi] = h_pos
    else:
        phi = np.arccos(1.0 - 2.0 * random.random())
        #camera_position = np.array([r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)])
        camera_position[r0i] = r * np.sin(phi) * np.cos(theta)
        camera_position[r1i] = r * np.sin(phi) * np.sin(theta)
        camera_position[hi] = r * np.cos(phi)
        camera_forward = vec_normalize(camera_position)
        if np.cos(phi) > 0.0:
            camera_position[hi] += h / 2
        else:
            camera_position[hi] -= h / 2

    use_camera_jitter_closeup = True
    if use_camera_jitter_closeup and r / r_base < 1.5:
        jitter_rad = 30.0 / 180.0 * np.pi
        camera_forward = jitter_direction(camera_forward, jitter_rad)

    #camera_position[0] += aabb[0]
    #camera_position[1] += aabb[2]
    #camera_position[2] += aabb[4]
    camera_right = vec_normalize(vec_cross(global_up, camera_forward))
    camera_up = vec_normalize(vec_cross(camera_forward, camera_right))
    rotation_matrix = np.empty((4, 4))
    for i in range(4):
        rotation_matrix[i, 0] = camera_right[i] if i < 3 else 0.0
        rotation_matrix[i, 1] = camera_up[i] if i < 3 else 0.0
        rotation_matrix[i, 2] = camera_forward[i] if i < 3 else 0.0
        rotation_matrix[i, 3] = 0.0 if i < 3 else 1.0
    inverse_view_matrix = matrix_translation(camera_position).dot(rotation_matrix)
    view_matrix = np.linalg.inv(inverse_view_matrix)
    view_matrix_array = np.empty(16)
    for i in range(4):
        for j in range(4):
            view_matrix_array[i * 4 + j] = view_matrix[j, i]
    return view_matrix_array, view_matrix, inverse_view_matrix


def sample_random_view_parametrized(params):
    u = params['u']
    v = params['v']
    w = params['w']
    camera_position = np.array([params['cx'], params['cy'], params['cz']])
    h = Rotation.from_quat(np.array([
        np.sqrt(1-u) * np.sin(2*np.pi*v),
        np.sqrt(1-u) * np.cos(2*np.pi*v),
        np.sqrt(u) * np.sin(2*np.pi*w),
        np.sqrt(u) * np.cos(2*np.pi*w)
    ]))
    rotation_matrix = np.zeros((4, 4))
    rotation_matrix[:3, :3] = h.as_matrix()
    rotation_matrix[3, 3] = 1
    inverse_view_matrix = matrix_translation(camera_position).dot(rotation_matrix)
    view_matrix = np.linalg.inv(inverse_view_matrix)
    view_matrix_array = np.empty(16)
    for i in range(4):
        for j in range(4):
            view_matrix_array[i * 4 + j] = view_matrix[j, i]
    return view_matrix_array, view_matrix, inverse_view_matrix


def get_position_random_range(aabb):
    rx = 0.5 * (aabb.max.x - aabb.min.x)
    ry = 0.5 * (aabb.max.y - aabb.min.y)
    rz = 0.5 * (aabb.max.z - aabb.min.z)
    r_median = np.median([rx, ry, rz])
    return [(aabb[i * 2] - 2.0 * r_median, aabb[i * 2 + 1] + 2.0 * r_median) for i in range(3)]


def sample_random_view(aabb):
    # https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
    rvec = [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
    u = rvec[0]
    v = rvec[1]
    w = rvec[2]
    cam_pos_range = get_position_random_range(aabb)
    camera_position = np.array([random.uniform(cam_pos_range[i][0], cam_pos_range[i][1]) for i in range(3)])
    params = {
        'cx': camera_position[0], 'cy': camera_position[1], 'cz': camera_position[2],
        'u': u, 'v': v, 'w': w
    }
    view_matrix_array, view_matrix, inverse_view_matrix = sample_random_view_parametrized(params)
    return view_matrix_array, view_matrix, inverse_view_matrix, params


def iceil(a, b):
    return -(a // -b)


def build_projection_matrix(fovy, aspect):
    z_near = 0.001953125
    z_far = 80.0
    tan_half_fovy = np.tan(fovy / 2.0)
    result = np.zeros((4, 4))
    result[0, 0] = 1.0 / (aspect * tan_half_fovy)
    result[1, 1] = 1.0 / tan_half_fovy
    result[2, 2] = z_far / (z_near - z_far)
    result[3, 2] = -1.0
    result[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return result


def make_view_matrix(camera_position, camera_right, camera_up, camera_forward):
    rotation_matrix = np.zeros((4, 4))
    for i in range(4):
        rotation_matrix[i, 0] = camera_right[i] if i < 3 else 0.0
        rotation_matrix[i, 1] = camera_up[i] if i < 3 else 0.0
        rotation_matrix[i, 2] = camera_forward[i] if i < 3 else 0.0
        rotation_matrix[i, 3] = 0.0 if i < 3 else 1.0
    inverse_view_matrix = matrix_translation(camera_position).dot(rotation_matrix)
    view_matrix = np.linalg.inv(inverse_view_matrix)
    view_matrix_array = np.empty(16)
    for i in range(4):
        for j in range(4):
            view_matrix_array[i * 4 + j] = view_matrix[j, i]
    return view_matrix_array


def get_scale_factor(aabb: d.AABB3):
    dimensions = aabb.get_dimensions()
    max_dimension = np.max([dimensions.x, dimensions.y, dimensions.z])
    return 0.5 / max_dimension


def apply_scale_factor_aabb(aabb: d.AABB3, scale_factor: float):
    return d.AABB3(
        d.vec3(aabb.min.x * scale_factor, aabb.min.y * scale_factor, aabb.min.z * scale_factor),
        d.vec3(aabb.max.x * scale_factor, aabb.max.y * scale_factor, aabb.max.z * scale_factor))
