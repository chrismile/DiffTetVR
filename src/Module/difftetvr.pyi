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

from __future__ import annotations
import torch
import difftetvr
import typing
import enum

__all__ = [
    "forward",
]


class vec3:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float) -> None:
        pass

class vec4:
    x: float
    y: float
    z: float
    w: float

    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        pass

class uvec3:
    x: int
    y: int
    z: int

    def __init__(self, x: int, y: int, z: int) -> None:
        pass


class AABB3:
    min: vec3
    max: vec3

    @overload
    def __init__(self) -> None:
        pass
    @overload
    def __init__(self, min: vec3, max: vec3) -> None:
        pass

    def get_dimensions(self) -> vec3:
        pass
    def get_extent(self) -> vec3:
        pass
    def get_center(self) -> vec3:
        pass
    def get_minimum(self) -> vec3:
        pass
    def get_maximum(self) -> vec3:
        pass


class FTetWildParams:
    """ https://github.com/wildmeshing/fTetWild?tab=readme-ov-file#command-line-switches"""
    relative_ideal_edge_length: float = 0.05  # -l
    epsilon: float = 1e-3                     # -e
    skip_simplify: bool = False               # --skip-simlify
    coarsen: bool = False                     # --coarsen
    def __init__(self) -> None:
        pass

class TetGenParams:
    """ https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html """
    use_steiner_points: bool = True              # -q; to remove badly-shaped tetrahedra
    use_radius_edge_ratio_bound: bool = False
    radius_edge_ratio_bound: float = 1.2         # radius-edge ratio bound
    use_maximum_volume_constraint: bool = False  # -a
    maximum_tetrahedron_volume: float = 1.0
    coarsen: bool = False                        # -R
    maximum_dihedral_angle: float = 165.0        # -o/
    # -O; mesh optimization settings
    mesh_optimization_level: int = 2             # Between 0 and 10.
    use_edge_and_face_flips: bool = True
    use_vertex_smoothing: bool = True
    use_vertex_insertion_and_deletion: bool = True
    def __init__(self) -> None:
        pass


class TestCase(enum.Enum):
    SINGLE_TETRAHEDRON = enum.auto() # (= 0)

class SplitGradientType(enum.Enum):
    POSITION = enum.auto()     # (= 0)
    COLOR = enum.auto()        # (= 1)
    ABS_POSITION = enum.auto() # (= 2)
    ABS_COLOR = enum.auto()    # (= 3)

class TetMesh:
    def __init__(self) -> None:
        pass
    def set_use_gradients(self, use_gradients: bool = True) -> None:
        pass
    def load_test_data(self, test_case: TestCase) -> None:
        pass
    def load_from_file(self, file_path: str) -> bool:
        pass
    def save_to_file(self, file_path: str) -> bool:
        pass
    def get_bounding_box(self) -> AABB3:
        pass
    def set_vertices_changed_on_device(self, vertices_changed: bool = True) -> None:
        pass

    def set_force_use_ovm_representation(self) -> None:
        """ Coarse to fine strategy."""
        pass
    def set_hex_mesh_const(
            self,
            aabb: AABB3,
            xs: int,
            ys: int,
            zs: int,
            const_color: vec4
    ) -> None:
        """ Initialize with tetrahedralized hex mesh with constant color. """
        pass
    def set_tetrahedralized_grid_ftetwild(
            self,
            aabb: AABB3,
            xs: int,
            ys: int,
            zs: int,
            const_color: vec4,
            params: FTetWildParams
    ) -> bool:
        """ Initialize with constant color tet mesh tetrahedralized from a grid using fTetWild. """
        pass
    def set_tetrahedralized_grid_tetgen(
            self,
            aabb: AABB3,
            xs: int,
            ys: int,
            zs: int,
            const_color: vec4,
            params: TetGenParams
    ) -> bool:
        """ Initialize with constant color tet mesh tetrahedralized from a grid using TetGen. """
        pass

    # Get mesh information.
    def get_num_cells(self) -> int:
        pass
    def get_num_vertices(self) -> int:
        pass

    def get_vertex_positions(self) -> torch.Tensor:
        pass
    def get_vertex_colors(self) -> torch.Tensor:
        pass
    def get_vertex_boundary_bit_tensor(self) -> torch.Tensor:
        pass

    def check_is_any_tet_degenerate(self) -> bool:
        """ Returns whether any tetrahedral element is degenerate (i.e., has a volume <= 0). """
        pass

    def unlink_tets(self) -> None:
        """ Removes the links between all tets, i.e., a potentially used shared index representation is reversed. """
        pass
    def split_by_largest_gradient_magnitudes(
            self, renderer: Renderer, split_gradient_type: SplitGradientType, splits_ratio: float) -> None:
        pass


class RendererType(enum.Enum):
    PPLL = enum.auto() # (= 0)
    PROJECTION = enum.auto() # (= 1)
    INTERSECTION = enum.auto() # (= 2)

class Renderer:
    def __init__(self, renderer_type: RendererType = RendererType.PPLL) -> None:
        pass
    def get_renderer_type(self) -> RendererType:
        pass
    def set_tet_mesh(self, tet_mesh: TetMesh) -> None:
        pass
    def get_tet_mesh(self) -> TetMesh:
        pass
    def get_attenuation(self) -> float:
        pass
    def set_attenuation(self, attenuation_coefficient: float) -> None:
        pass
    def set_coarse_to_fine_target_num_tets(self, target_num_tets: int) -> None:
        pass
    def set_use_early_ray_termination(self, use_early_ray_termination: bool) -> None:
        pass
    def set_early_ray_out_thresh(self, threshold: float) -> None:
        pass
    def set_early_ray_out_alpha(self, alpha: float) -> None:
        pass
    def set_clear_color(self, color: vec4) -> None:
        pass
    def set_viewport_size(self, image_width: int, image_height: int, recreate_swapchain: bool = True) -> None:
        pass
    def reuse_intermediate_buffers_from(self, renderer_other: Renderer) -> None:
        pass
    def set_camera_fovy(self, fovy: float) -> None:
        pass
    def set_view_matrix(self, view_matrix_array: list[float]) -> None:
        pass
    def render(self) -> torch.Tensor:
        pass
    def render_adjoint(self, image_adjoint: torch.Tensor) -> None:
        pass


class TetRegularizer:
    def __init__(self, tet_mesh: TetMesh, reg_lambda: float, softplus_beta: float) -> None:
        pass
    def compute_grad(self) -> RendererType:
        pass


class TetMeshVtkWriter:
    """
    For logging purposes (e.g., writing gradients).
    """
    def __init__(self, file_path: str, is_binary: bool = True) -> None:
        pass
    def write_next_time_step(self, tet_mesh: TetMesh):
        pass


class RegularGrid:
    def __init__(self) -> None:
        pass
    def load_from_file(self, file_path: str) -> bool:
        pass
    def get_grid_size_x(self) -> int:
        pass
    def get_grid_size_y(self) -> int:
        pass
    def get_grid_size_z(self) -> int:
        pass
    def get_bounding_box(self) -> AABB3:
        pass


class RegularGridInterpolationMode(enum.Enum):
    NEAREST = enum.auto() # (= 0)
    LINEAR = enum.auto() # (= 1)

class RegularGridRenderer:
    def __init__(self) -> None:
        pass
    def set_regular_grid(self, regular_grid: RegularGrid) -> None:
        pass
    def get_regular_grid(self) -> RegularGrid:
        pass
    def get_attenuation(self) -> float:
        pass
    def set_attenuation(self, attenuation_coefficient: float) -> None:
        pass
    def load_transfer_function_from_file(self, file_path: str) -> bool:
        pass
    def set_clear_color(self, color: vec4) -> None:
        pass
    def get_step_size(self) -> float:
        pass
    def set_step_size(self, step_size: float) -> None:
        pass
    def set_viewport_size(self, image_width: int, image_height: int, recreate_swapchain: bool = True) -> None:
        pass
    def set_camera_fovy(self, fovy: float) -> None:
        pass
    def set_view_matrix(self, view_matrix_array: list[float]) -> None:
        pass
    def render(self) -> torch.Tensor:
        pass


class OptimizerType(enum.Enum):
    SGD = enum.auto()  # (= 0)
    ADAM = enum.auto() # (= 1)

class LossType(enum.Enum):
    L1 = enum.auto() # (= 0)
    L2 = enum.auto() # (= 1)

class OptimizerSettings:
    # SGD & Adam.
    learning_rate: float = 0.4
    lr_decay_rate: float = 0.999
    # Adam.
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    def __init__(self) -> None:
        pass

class TetRegularizerSettings:
    # Regularizer loss weight (0 means turned off).
    lambda_: float = 0.1
    # Softplus parameter.
    beta: float = 100.0
    def __init__(self) -> None:
        pass

class InitGridType(enum.Enum):
    DECOMPOSED_HEX_MESH = enum.auto() # (= 0)
    MESHING_FTETWILD = enum.auto()    # (= 1)
    MESHING_TETGEN = enum.auto()      # (= 2)

class OptimizationSettings:
    optimizer_type: OptimizerType = OptimizerType.ADAM
    loss_type: LossType = LossType.L2
    optimize_positions: bool = True
    optimize_colors: bool = True
    optimizer_settings_positions: OptimizerSettings = OptimizerSettings()
    optimizer_settings_colors: OptimizerSettings = OptimizerSettings()
    tet_regularizer_settings: TetRegularizerSettings = TetRegularizerSettings()
    max_num_epochs: int = 200
    fix_boundary: bool = False
    # DVR.
    image_width: int = 512
    image_height: int = 512
    attenuation_coefficient: float = 100.0
    sample_random_view: bool = True
    # Selected file name.
    data_set_file_name_gt: str
    # Selected file name.
    data_set_file_name_opt: str
    # Coarse to fine.
    use_coarse_to_fine: bool = False
    use_constant_init_grid: bool = False
    init_grid_type: InitGridType = InitGridType.DECOMPOSED_HEX_MESH
    init_grid_resolution: uvec3 = uvec3(16, 16, 16)
    ftetwild_params: FTetWildParams = FTetWildParams()
    tetgen_params: TetGenParams = TetGenParams()
    max_num_tets: int = 1320000
    num_splits_ratio: float = 0.1
    split_gradient_type: SplitGradientType = SplitGradientType.ABS_COLOR
    # Export position gradient field.
    export_position_gradients: bool = False
    export_file_name_gradient_field: str
    is_binary_vtk: bool = True
    def __init__(self) -> None:
        pass


class CameraSettings:
    def __init__(self) -> None:
        pass
    def set_intrinsics(self, img_width: int, img_height: int, fovy: float, near: float, far: float) -> None:
        pass
    def set_view_matrix(self, view_matrix_array: list[float]) -> None:
        pass

class VoxelCarvingType(enum.Enum):
    DENSE_CPU = enum.auto()  # (= 0)

class VoxelCarving:
    def __init__(
            self, grid_bounding_box: AABB3, grid_resolution: uvec3,
            voxel_carving_type: VoxelCarvingType = VoxelCarvingType.DENSE_CPU) -> None:
        pass
    def process_next_frame(self, input_image: torch.Tensor, camera_settings: CameraSettings) -> None:
        pass
    def compute_non_empty_bounding_box(self) -> AABB3:
        pass


def render(X: torch.Tensor) -> torch.Tensor:
    """
    Forward rendering pass.
    """
