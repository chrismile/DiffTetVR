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

#include "IntersectUniform.glsl"

struct TriangleKeyValue {
    uint index;
    float depth;
};
layout(binding = 1, std430) readonly buffer TriangleKeyValueBuffer {
    TriangleKeyValue triangleKeyValues[];
};
layout(binding = 2, std430) readonly buffer TriangleVertexPositionBuffer {
    vec4 vertexPositions[];
};
layout(binding = 3, std430) readonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};
layout(binding = 4) uniform TriangleCounterBuffer {
    uint numTriangles;
};

layout(location = 0) flat out uint tetIdx;

void main() {
    // The triangles are rendered in reverse order using index "numTriangles - i - 1".
    uint triangleIdx = triangleKeyValues[numTriangles - gl_VertexIndex / 3u - 1u].index;
    uint vertexIdx = triangleIdx * 3u + (gl_VertexIndex % 3u);
    tetIdx = triangleTetIndices[triangleIdx];
    gl_Position = vertexPositions[vertexIdx];
}


-- Fragment

#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_ARB_fragment_shader_interlock : require
//#extension GL_EXT_debug_printf : enable

layout(early_fragment_tests, pixel_interlock_ordered) in;

#include "TetFaceTable.glsl"
#include "RayCommon.glsl"
#include "IntersectUniform.glsl"

layout(push_constant) uniform PushConstants {
    uint useAbsGrad;
};

// Tet data.
layout(binding = 5, std430) readonly buffer TetIndexBuffer {
    uint tetsIndices[];
};
layout(binding = 6, scalar) readonly buffer TetVertexPositionBuffer {
    vec3 tetsVertexPositions[];
};
#ifdef PER_VERTEX_COLORS
layout(binding = 7, scalar) readonly buffer TetVertexColorBuffer {
    vec4 tetsVertexColors[];
};
#else
layout(binding = 7, scalar) readonly buffer TetCellColorBuffer {
    vec4 tetsCellColors[];
};
#endif

in vec4 gl_FragCoord;
layout(location = 0) flat in uint tetIdx;
layout(location = 0) out vec4 outputColor;

#include "BackwardCommon.glsl"
layout(binding = 10, rgba32f) uniform image2D colorImageOpt;
layout(binding = 11, rgba32f) uniform image2D adjointColors;

#include "RayIntersectionTests.glsl"
#include "BarycentricInterpolation.glsl"

void main() {
    // Compute camera ray direction.
    vec2 fragNdc = 2.0 * (gl_FragCoord.xy / vec2(viewportSize)) - 1.0;
    vec3 rayTarget = (invProjMat * vec4(fragNdc.xy, 1.0, 1.0)).xyz;
    vec3 normalizedTarget = normalize(rayTarget.xyz);
    vec3 rayDirection = (invViewMat * vec4(normalizedTarget, 0.0)).xyz;

    // Fetch the tet vertex positions and colors.
    vec3 tetVertexPositions[4];
#ifdef PER_VERTEX_COLORS
    vec4 tetVertexColors[4];
#endif
    const uint tetOffset = tetIdx * 4u;
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        uint tetGlobalVertIdx = tetsIndices[tetOffset + tetVertIdx];
        tetVertexPositions[tetVertIdx] = tetsVertexPositions[tetGlobalVertIdx];
#ifdef PER_VERTEX_COLORS
        tetVertexColors[tetVertIdx] = tetsVertexColors[tetGlobalVertIdx];
#endif
    }

    uint f0, f1; //< Face index hit 0, 1.
    float t0, t1; //< Ray distance hit 0, 1.
    if (!intersectRayTet(cameraPosition, rayDirection, tetVertexPositions, f0, f1, t0, t1)) {
        discard; // Should never happen...
    }

    vec3 pf0 = cameraPosition + t0 * rayDirection;
    vec3 pf1 = cameraPosition + t1 * rayDirection;

    // Barycentric interpolation (face 0 and 1).
    uint if00 = tetFaceTable[f0][0];
    uint if01 = tetFaceTable[f0][1];
    uint if02 = tetFaceTable[f0][2];
    vec3 pf00 = tetVertexPositions[if00];
    vec3 pf01 = tetVertexPositions[if01];
    vec3 pf02 = tetVertexPositions[if02];
#ifdef PER_VERTEX_COLORS
    vec4 cf00 = tetVertexColors[if00];
    vec4 cf01 = tetVertexColors[if01];
    vec4 cf02 = tetVertexColors[if02];
#endif
    vec3 bary0 = barycentricInterpolation(pf00, pf01, pf02, pf0);
    float uf0 = bary0.x;
    float vf0 = bary0.y;
#ifdef PER_VERTEX_COLORS
    vec4 fragment0Color = cf00 * bary0.x + cf01 * bary0.y + cf02 * bary0.z;
#endif
    if00 = tetsIndices[tetOffset + if00];
    if01 = tetsIndices[tetOffset + if01];
    if02 = tetsIndices[tetOffset + if02];

    uint if10 = tetFaceTable[f1][0];
    uint if11 = tetFaceTable[f1][1];
    uint if12 = tetFaceTable[f1][2];
    vec3 pf10 = tetVertexPositions[if10];
    vec3 pf11 = tetVertexPositions[if11];
    vec3 pf12 = tetVertexPositions[if12];
#ifdef PER_VERTEX_COLORS
    vec4 cf10 = tetVertexColors[if10];
    vec4 cf11 = tetVertexColors[if11];
    vec4 cf12 = tetVertexColors[if12];
#endif
    vec3 bary1 = barycentricInterpolation(pf10, pf11, pf12, pf1);
    float uf1 = bary1.x;
    float vf1 = bary1.y;
#ifdef PER_VERTEX_COLORS
    vec4 fragment1Color = cf10 * bary1.x + cf11 * bary1.y + cf12 * bary1.z;
#endif
    if10 = tetsIndices[tetOffset + if10];
    if11 = tetsIndices[tetOffset + if11];
    if12 = tetsIndices[tetOffset + if12];

    vec4 colorAcc; // temp from loop
    float A; // temp from loop
    vec4 dOut_dcf0 = vec4(0.0);
    vec4 dOut_dcf1 = vec4(0.0);
    float dOut_dt = 0.0;
    float t = t1 - t0;
    float tSeg = t / float(NUM_SUBDIVS);
    const float INV_N_SUB = 1.0 / float(NUM_SUBDIVS);
    ivec2 workIdx = ivec2(gl_FragCoord.xy);

    // Begin critical section; other fragments mapping to same pixel need to wait for updated color/gradient.
    beginInvocationInterlockARB();
    vec4 colorRayOut = imageLoad(colorImageOpt, workIdx);
    vec4 dOut_dColorRayOut = imageLoad(adjointColors, workIdx);

#ifdef PER_VERTEX_COLORS
    for (int s = NUM_SUBDIVS - 1; s > 0; s--) {
        float fbegin = (float(s)) * INV_N_SUB;
        float fmid = (float(s) + 0.5) * INV_N_SUB;
        float fend = (float(s) + 1.0) * INV_N_SUB;
        vec3 c0 = mix(fragment0Color.rgb, fragment1Color.rgb, fbegin);
        vec3 c1 = mix(fragment0Color.rgb, fragment1Color.rgb, fend);
        float alpha = mix(fragment0Color.a, fragment1Color.a, fmid);
        colorAcc = accumulateLinearConst(tSeg, c0, c1, alpha * attenuationCoefficient, A);

        // Inversion trick from "Differentiable Direct Volume Rendering", Weiß et al. 2021.
        float alphaRayIn = (colorAcc.a - colorRayOut.a) / (colorAcc.a - 1.0);
        vec3 colorRayIn = colorRayOut.rgb - (1.0 - alphaRayIn) * colorAcc.rgb;
        //colorRayOut.a = colorRayIn.a + (1.0 - colorRayIn.a) * colorAcc.a;
        //colorRayOut.rgb = colorRayIn.rgb + (1.0 - colorRayIn.a) * colorAcc.rgb;

        // Compute adjoint for accumulated color/opacity.
        vec4 dOut_dColorAcc;
        dOut_dColorAcc.rgb = (1.0 - alphaRayIn) * dOut_dColorRayOut.rgb;
        dOut_dColorAcc.a = (1.0 - alphaRayIn) * dOut_dColorRayOut.a;

        // Backpropagation for the accumulated color.
        // colorCurrAdjoint.rgb stays the same (see paper cited above, Chat^(i) = Chat^(i+1)).
        float alphaNewAdjoint = dOut_dColorRayOut.a * (1.0 - colorAcc.a) - dot(dOut_dColorRayOut.rgb, colorAcc.rgb);
        dOut_dColorRayOut.a = alphaNewAdjoint;
        colorRayOut = vec4(colorRayIn, alphaRayIn);

        // Compute adjoint for the pre-accumulation colors and opacity.
        vec3 dOut_dc0;
        vec3 dOut_dc1;
        float dOut_da;
        accumulateLinearConstAdjoint(
                tSeg, INV_N_SUB, c0, c1, alpha * attenuationCoefficient, A, dOut_dColorAcc,
                dOut_dt, dOut_dc0, dOut_dc1, dOut_da);
        dOut_da *= attenuationCoefficient;

        // Backpropagate gradients wrt. segmented color/opacity to intersection point color/opacity.
        // dc0_dcf0 = (1.0 - fbegin), dc1_dcf0 = (1.0 - fend), dc0_dcf1 = fbegin, dc1_dcf1 = fend.
        dOut_dcf0 += vec4((1.0 - fbegin) * dOut_dc0 + (1.0 - fend) * dOut_dc1, (1.0 - fmid) * dOut_da);
        dOut_dcf1 += vec4(fbegin * dOut_dc0 + fend * dOut_dc1, fmid * dOut_da);
    }
#else // !defined(PER_VERTEX_COLORS)
    // TODO
#endif // PER_VERTEX_COLORS

    vec3 dOut_dpf00, dOut_dpf01, dOut_dpf02, dOut_dpf10, dOut_dpf11, dOut_dpf12;
    vec4 dOut_dcf00, dOut_dcf01, dOut_dcf02, dOut_dcf10, dOut_dcf11, dOut_dcf12;
    float dOut_duf0, dOut_dvf0, dOut_duf1, dOut_dvf1;
    segmentLengthAdjoint(
            pf00, pf01, pf02, pf10, pf11, pf12, uf0, vf0, uf1, vf1, dOut_dt,
            dOut_duf0, dOut_dvf0, dOut_duf1, dOut_dvf1,
            dOut_dpf00, dOut_dpf01, dOut_dpf02, dOut_dpf10, dOut_dpf11, dOut_dpf12);
    //vec3 dOut_dpf00 = vec3(0.0), dOut_dpf01 = vec3(0.0), dOut_dpf02 = vec3(0.0), dOut_dpf10 = vec3(0.0), dOut_dpf11 = vec3(0.0), dOut_dpf12 = vec3(0.0);
    //vec4 dOut_dcf00, dOut_dcf01, dOut_dcf02, dOut_dcf10, dOut_dcf11, dOut_dcf12;
    //float dOut_duf0 = 0.0, dOut_dvf0 = 0.0, dOut_duf1 = 0.0, dOut_dvf1 = 0.0;
    baryAdjoint(
            pf0, pf00, pf01, pf02, cf00, cf01, cf02, uf0, vf0,
            dOut_dcf0, dOut_duf0, dOut_dvf0,
            dOut_dpf00, dOut_dpf01, dOut_dpf02, dOut_dcf00, dOut_dcf01, dOut_dcf02);
    baryAdjoint(
            pf1, pf10, pf11, pf12, cf10, cf11, cf12, uf1, vf1,
            dOut_dcf1, dOut_duf1, dOut_dvf1,
            dOut_dpf10, dOut_dpf11, dOut_dpf12, dOut_dcf10, dOut_dcf11, dOut_dcf12);

    // Update intermediate color and gradients and release pixel lock.
    imageStore(colorImageOpt, workIdx, colorRayOut);
    imageStore(adjointColors, workIdx, dOut_dColorRayOut);
    endInvocationInterlockARB();

    /*
     * For testing an idea similar to the one from the following 3DGS paper:
     * "AbsGS: Recovering Fine Details for 3D Gaussian Splatting". 2024.
     * Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, Yong Dou.
     */
    if (useAbsGrad != 0u) {
#ifdef PER_VERTEX_COLORS
        dOut_dcf00 = abs(dOut_dcf00);
        dOut_dcf01 = abs(dOut_dcf01);
        dOut_dcf02 = abs(dOut_dcf02);
        dOut_dcf10 = abs(dOut_dcf10);
        dOut_dcf11 = abs(dOut_dcf11);
        dOut_dcf12 = abs(dOut_dcf12);
        
#else
        dOut_dc = abs(dOut_dc);
#endif
        dOut_dpf00 = abs(dOut_dpf00);
        dOut_dpf01 = abs(dOut_dpf01);
        dOut_dpf02 = abs(dOut_dpf02);
        dOut_dpf10 = abs(dOut_dpf10);
        dOut_dpf11 = abs(dOut_dpf11);
        dOut_dpf12 = abs(dOut_dpf12);
    }

    // Accumulate gradients wrt. tet properties.
#ifdef PER_VERTEX_COLORS
    atomicAddGradCol(if00, dOut_dcf00);
    atomicAddGradCol(if01, dOut_dcf01);
    atomicAddGradCol(if02, dOut_dcf02);

    atomicAddGradCol(if10, dOut_dcf10);
    atomicAddGradCol(if11, dOut_dcf11);
    atomicAddGradCol(if12, dOut_dcf12);
#else
    atomicAddGradCol(tetIdx, dOut_dc);
#endif

    atomicAddGradPos(if00, dOut_dpf00);
    atomicAddGradPos(if01, dOut_dpf01);
    atomicAddGradPos(if02, dOut_dpf02);

    atomicAddGradPos(if10, dOut_dpf10);
    atomicAddGradPos(if11, dOut_dpf11);
    atomicAddGradPos(if12, dOut_dpf12);

    outputColor = vec4(0.0);
}
