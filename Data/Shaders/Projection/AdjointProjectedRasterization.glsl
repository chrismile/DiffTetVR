/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

-- Vertex

#version 450

struct TriangleKeyValue {
    uint index;
    float depth;
};
layout(binding = 0, std430) readonly buffer TriangleKeyValueBuffer {
    TriangleKeyValue triangleKeyValues[];
};
layout(binding = 1, std430) readonly buffer TriangleVertexPositionBuffer {
    vec4 vertexPositions[];
};
layout(binding = 2, std430) readonly buffer TriangleVertexColorBuffer {
    vec4 vertexColors[];
};
layout(binding = 3, std430) readonly buffer TriangleVertexDepthBuffer {
    float vertexDepths[];
};
layout(binding = 4) uniform TriangleCounterBuffer {
    uint numTriangles;
};

layout(location = 0) flat out uint triangleIdx;
layout(location = 1) out vec4 p;
layout(location = 2) out vec4 fragmentColor;
layout(location = 3) out float fragmentDepth;

void main() {
    // The triangles are rendered in reverse order using index "numTriangles - i - 1".
    triangleIdx = triangleKeyValues[numTriangles - gl_VertexIndex / 3u - 1u].index;
    uint vertexIdx = triangleIdx * 3u + (gl_VertexIndex % 3u);
    p = vertexPositions[vertexIdx];
    fragmentColor = vertexColors[vertexIdx];
    fragmentDepth = vertexDepths[vertexIdx];
    gl_Position = p;
}


-- Fragment

#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_ARB_fragment_shader_interlock : require

layout(early_fragment_tests, pixel_interlock_ordered) in;

#define PROJECTED_RASTER
#include "BackwardCommon.glsl"
#include "BarycentricInterpolation.glsl"

layout(binding = 1, std430) readonly buffer TriangleVertexPositionBuffer {
    vec4 vertexPositions[];
};
layout(binding = 2, std430) readonly buffer TriangleVertexColorBuffer {
    vec4 vertexColors[];
};
layout(binding = 3, std430) readonly buffer TriangleVertexDepthBuffer {
    float vertexDepths[];
};

#ifndef PER_VERTEX_COLORS
layout(push_constant) uniform PushConstants {
    uint useAbsGrad;
};
layout(binding = 6, std430) readonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};
#endif

layout(binding = 7, scalar) coherent buffer VertexDepthGradientBuffer {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float vertexDepthGradients[]; // stride: float
#else
    uint vertexDepthGradients[]; // stride: float
#endif
};
void atomicAddGradDepth(uint idx, float value) {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    atomicAdd(vertexDepthGradients[idx], value);
#else
    uint oldValue = vertexDepthGradients[idx];
    uint expectedValue, newValue;
    do {
        expectedValue = oldValue;
        newValue = floatBitsToUint(uintBitsToFloat(oldValue) + value);
        oldValue = atomicCompSwap(vertexDepthGradients[idx], expectedValue, newValue);
    } while (oldValue != expectedValue);
#endif
}

in vec4 gl_FragCoord;
layout(location = 0) flat in uint triangleIdx;
layout(location = 1) in vec4 p;
layout(location = 2) in vec4 fragmentColor;
layout(location = 3) in float fragmentDepth;
layout(location = 0) out vec4 outputColor;

layout(binding = 10, rgba32f) uniform coherent image2D colorImageOpt;
layout(binding = 11, rgba32f) uniform coherent image2D adjointColors;

void main() {
    ivec2 workIdx = ivec2(gl_FragCoord.xy);

    float A = exp(-fragmentColor.a * fragmentDepth);
    float alphaAcc = 1.0 - A;
    vec4 colorAcc = vec4(fragmentColor.rgb * alphaAcc, alphaAcc);

    uint vertexIdx0 = triangleIdx * 3u;
    uint vertexIdx1 = vertexIdx0 + 1u;
    uint vertexIdx2 = vertexIdx0 + 2u;
    vec3 p0 = vertexPositions[vertexIdx0].xyz;
    vec3 p1 = vertexPositions[vertexIdx1].xyz;
    vec3 p2 = vertexPositions[vertexIdx2].xyz;
#ifdef PER_VERTEX_COLORS
    vec4 c0 = vertexColors[vertexIdx0];
    vec4 c1 = vertexColors[vertexIdx1];
    vec4 c2 = vertexColors[vertexIdx2];
#endif
    float d0 = vertexDepths[vertexIdx0];
    float d1 = vertexDepths[vertexIdx1];
    float d2 = vertexDepths[vertexIdx2];

    vec3 baryCoords = barycentricInterpolation(p0, p1, p2, p.xyz);
    float u = baryCoords.x;
    float v = baryCoords.y;

    // Begin critical section; other fragments mapping to same pixel need to wait for updated color/gradient.
    beginInvocationInterlockARB();
    vec4 colorRayOut = imageLoad(colorImageOpt, workIdx);
    vec4 dOut_dColorRayOut = imageLoad(adjointColors, workIdx);

    // Inversion trick from "Differentiable Direct Volume Rendering", Weiß et al. 2021.
    float alphaRayIn = (colorAcc.a - colorRayOut.a) / (colorAcc.a - 1.0);
    vec3 colorRayIn = colorRayOut.rgb - (1.0 - alphaRayIn) * colorAcc.rgb;

    // Compute adjoint for accumulated color/opacity.
    vec4 dOut_dColorAcc;
    dOut_dColorAcc.rgb = (1.0 - alphaRayIn) * dOut_dColorRayOut.rgb;
    dOut_dColorAcc.a = (1.0 - alphaRayIn) * dOut_dColorRayOut.a;

    // Backpropagation for the accumulated color.
    // colorCurrAdjoint.rgb stays the same (see paper cited above, Chat^(i) = Chat^(i+1)).
    float alphaNewAdjoint = dOut_dColorRayOut.a * (1.0 - colorAcc.a) - dot(dOut_dColorRayOut.rgb, colorAcc.rgb);
    dOut_dColorRayOut.a = alphaNewAdjoint;
    colorRayOut = vec4(colorRayIn, alphaRayIn);

    // Update intermediate color and gradients and release pixel lock.
    imageStore(colorImageOpt, workIdx, colorRayOut);
    imageStore(adjointColors, workIdx, dOut_dColorRayOut);
    endInvocationInterlockARB();

    float dOut_dAlphaAcc = dot(dOut_dColorAcc.rgb, fragmentColor.rgb) + dOut_dColorAcc.a;
    vec4 dOut_dc;
    dOut_dc.rgb = colorAcc.a * dOut_dColorAcc.rgb;
    dOut_dc.a = dOut_dAlphaAcc * fragmentDepth * A;
    float dOut_dd = dOut_dAlphaAcc * fragmentColor.a * A;

    float dOut_du = 0.0, dOut_dv = 0.0;
    vec3 dOut_dp0 = vec3(0.0), dOut_dp1 = vec3(0.0), dOut_dp2 = vec3(0.0);
#ifdef PER_VERTEX_COLORS
    vec4 dOut_dc0 = vec4(0.0), dOut_dc1 = vec4(0.0), dOut_dc2 = vec4(0.0);
#endif
    float dOut_dd0 = 0.0, dOut_dd1 = 0.0, dOut_dd2 = 0.0;
    baryAdjoint(
            p.xyz, p0, p1, p2,
#ifdef PER_VERTEX_COLORS
            c0, c1, c2,
#endif
            d0, d1, d2, baryCoords.x, baryCoords.y,
#ifdef PER_VERTEX_COLORS
            dOut_dc,
#endif
            dOut_dd, dOut_du, dOut_dv,
            dOut_dp0, dOut_dp1, dOut_dp2,
#ifdef PER_VERTEX_COLORS
            dOut_dc0, dOut_dc1, dOut_dc2,
#endif
            dOut_dd0, dOut_dd1, dOut_dd2);

#ifdef PER_VERTEX_COLORS
    atomicAddGradCol(vertexIdx0, dOut_dc0);
    atomicAddGradCol(vertexIdx1, dOut_dc1);
    atomicAddGradCol(vertexIdx2, dOut_dc2);
#else
    const uint tetIdx = triangleTetIndices[triangleIdx];
    /*
     * For testing an idea similar to the one from the following 3DGS paper:
     * "AbsGS: Recovering Fine Details for 3D Gaussian Splatting". 2024.
     * Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, Yong Dou.
     */
    if (useAbsGrad != 0u) {
        dOut_dc = abs(dOut_dc);
    }
    atomicAddGradCol(tetIdx, dOut_dc);
#endif

    atomicAddGradPos(vertexIdx0, dOut_dp0);
    atomicAddGradPos(vertexIdx1, dOut_dp1);
    atomicAddGradPos(vertexIdx2, dOut_dp2);

    atomicAddGradDepth(vertexIdx0, dOut_dd0);
    atomicAddGradDepth(vertexIdx1, dOut_dd1);
    atomicAddGradDepth(vertexIdx2, dOut_dd2);

    outputColor = vec4(0.0);
}
