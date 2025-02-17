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

import torch
import difftetvr as d


class DifferentiableRenderingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, renderer, tet_regularizer, use_abs_grad, vertex_positions, colors):
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
        # d_vertex_positions = ctx.renderer.get_tet_mesh().get_vertex_positions().grad
        # d_vertex_colors = ctx.renderer.get_tet_mesh().get_vertex_colors().grad
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
        if tet_mesh.get_color_storage() == d.ColorStorage.PER_VERTEX:
            colors = tet_mesh.get_vertex_colors()
        else:
            colors = tet_mesh.get_cell_colors()
        return DifferentiableRenderingFunction.apply(
            self.renderer, self.tet_regularizer, self.use_abs_grad, vertex_positions, colors)
