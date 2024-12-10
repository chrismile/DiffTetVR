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
        """ Initialize with tetrahedralized tet mesh with constant color. """
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
    def __init__(
            self,
            learning_rate: float = 0.4,
            lr_decay_rate: float = 0.999,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8
    ) -> None:
        """Auto-generated default constructor with named params"""
        pass

class TetRegularizerSettings:
    # Regularizer loss weight (0 means turned off).
    lambda_: float = 0.1
    # Softplus parameter.
    beta: float = 100.0
    def __init__(self, lambda_: float = 0.1, beta: float = 100.0) -> None:
        """Auto-generated default constructor with named params"""
        pass

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
    init_grid_resolution: uvec3 = uvec3(16, 16, 16)
    max_num_tets: int = 1320000
    num_splits_ratio: float = 0.1
    split_gradient_type: SplitGradientType = SplitGradientType.ABS_COLOR
    # Export position gradient field.
    export_position_gradients: bool = False
    export_file_name_gradient_field: str
    is_binary_vtk: bool = True
    def __init__(
            self,
            optimizer_type: OptimizerType = OptimizerType.ADAM,
            loss_type: LossType = LossType.L2,
            optimize_positions: bool = True,
            optimize_colors: bool = True,
            optimizer_settings_positions: OptimizerSettings = OptimizerSettings(),
            optimizer_settings_colors: OptimizerSettings = OptimizerSettings(),
            tet_regularizer_settings: TetRegularizerSettings = TetRegularizerSettings(),
            max_num_epochs: int = 200,
            fix_boundary: bool = False,
            image_width: int = 512,
            image_height: int = 512,
            attenuation_coefficient: float = 100.0,
            sample_random_view: bool = True,
            data_set_file_name_gt: str = "",
            data_set_file_name_opt: str = "",
            use_coarse_to_fine: bool = False,
            use_constant_init_grid: bool = False,
            init_grid_resolution: uvec3 = uvec3(16, 16, 16),
            max_num_tets: int = 1320000,
            num_splits_ratio: float = 0.1,
            split_gradient_type: SplitGradientType = SplitGradientType.ABS_COLOR,
            export_position_gradients: bool = False,
            export_file_name_gradient_field: str = "",
            is_binary_vtk: bool = True
    ) -> None:
        """Auto-generated default constructor with named params"""
        pass


def render(X: torch.Tensor) -> torch.Tensor:
    """
    Forward rendering pass.
    """
