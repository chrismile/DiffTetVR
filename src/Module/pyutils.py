import torch
import difftetvr as d


class DifferentiableRenderingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, renderer, tet_regularizer, use_abs_grad, vertex_positions, vertex_colors):
        image = renderer.render()
        ctx.save_for_backward(image)
        ctx.renderer = renderer
        ctx.tet_regularizer = tet_regularizer
        ctx.use_abs_grad = use_abs_grad
        return image

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, image_adj):
        image, = ctx.saved_tensors
        if ctx.tet_regularizer is not None:
            ctx.tet_regularizer.compute_grad()
        ctx.renderer.render_adjoint(image_adj, ctx.use_abs_grad)
        d_vertex_positions = ctx.renderer.get_tet_mesh().get_vertex_positions().grad
        d_vertex_colors = ctx.renderer.get_tet_mesh().get_vertex_colors().grad
        del ctx.renderer
        del ctx.tet_regularizer
        del ctx.use_abs_grad
        # We CANNOT return d_vertex_positions, d_vertex_colors, as we automatically do addition of new gradients
        # onto .grad entry on the Vulkan side.
        # return None, None, None, d_vertex_positions, d_vertex_colors
        return None, None, None, None, None


class DifferentiableRenderer(torch.nn.Module):
    def __init__(self, renderer, use_tet_regularizer: bool, tet_reg_lambda: float, tet_reg_softplus_beta: float):
        super(DifferentiableRenderer, self).__init__()
        self.renderer = renderer
        self.use_abs_grad = False
        self.tet_regularizer = None
        if use_tet_regularizer:
            self.tet_regularizer = d.TetRegularizer(tet_reg_lambda, tet_reg_softplus_beta)

    def set_use_abs_grad(self, use_abs_grad):
        self.use_abs_grad = use_abs_grad

    def forward(self):
        tet_mesh = self.renderer.get_tet_mesh()
        vertex_positions = tet_mesh.get_vertex_positions()
        vertex_colors = tet_mesh.get_vertex_colors()
        return DifferentiableRenderingFunction.apply(
            self.renderer, self.tet_regularizer, self.use_abs_grad, vertex_positions, vertex_colors)
