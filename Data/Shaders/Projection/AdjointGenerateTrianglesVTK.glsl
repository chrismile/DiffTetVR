/*
 * This file contains adapted code from VTK from:
 * https://github.com/Kitware/VTK/blob/master/Rendering/VolumeOpenGL2/vtkOpenGLProjectedTetrahedraMapper.cxx
 *
 * The original license of VTK is:
 *
 * Copyright (c) 1993-2015 Ken Martin, Will Schroeder, Bill Lorensen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Ken Martin, Will Schroeder, or Bill Lorensen nor the names
 *    of any contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

-- Compute

#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_debug_printf : enable

layout(local_size_x = BLOCK_SIZE) in;

#include "ProjectionUniform.glsl"
#include "TetFaceTable.glsl"
#include "ForwardCommon.glsl"
#include "RayCommon.glsl"
#ifdef USE_CLIP_PLANE
#include "ClipPlane.glsl"
#endif

layout(push_constant) uniform PushConstants {
    uint useAbsGrad;
};

// Previously atomically increased linear append index.
layout(binding = 1) uniform TetCounterBuffer {
    uint globalTetCounter;
};
layout(binding = 2, std430) readonly buffer TetTriangleOffsetBuffer {
    uint tetOffsets[];
};

// Input tet data.
layout(binding = 3, std430) readonly buffer TetIndexBuffer {
    uint tetsIndices[];
};
layout(binding = 4, scalar) readonly buffer TetVertexPositionBuffer {
    vec3 tetsVertexPositions[];
};
layout(binding = 5, std430) readonly buffer TetVertexColorBuffer {
    vec4 tetsVertexColors[];
};

// Output triangle data.
layout(binding = 6, std430) readonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};

// Output triangle data gradients.
layout(binding = 10, scalar) readonly buffer TriangleVertexPositionGradientBuffer {
    vec4 triangleVertexPositionGradients[];
};
layout(binding = 11, std430) readonly buffer TriangleVertexColorGradientBuffer {
    vec4 triangleVertexColorGradients[];
};
layout(binding = 12, std430) readonly buffer TriangleVertexDepthGradientBuffer {
    float triangleVertexDepthGradients[];
};

#include "BackwardCommon.glsl"

float GetCorrectedDepth(float x, float y, float z1, float z2) {
    vec4 eye1 = invProjMat * vec4(x, y, z1, 1.0);
    vec4 eye2 = invProjMat * vec4(x, y, z2, 1.0);
    return distance(eye1.xyz / eye1.w, eye2.xyz / eye2.w);
}

void GetCorrectedDepthAdjoint(
        float x, float y, float z1, float z2,
        float dOut_dDepth, inout float dOut_dx, inout float dOut_dy, inout float dOut_dz1, inout float dOut_dz2) {
    vec4 phom = invProjMat * vec4(x, y, z1, 1.0);
    vec4 qhom = invProjMat * vec4(x, y, z2, 1.0);
    vec3 p = phom.xyz / phom.w;
    vec3 q = qhom.xyz / qhom.w;
    float depth = distance(p, q);

    vec3 dDepth_dp = vec3((p.x - q.x) / depth, (p.y - q.y) / depth, (p.z - q.z) / depth);
    //vec3 ddepth_dq = vec3((-p.x + q.x) / depth, (-p.y + q.y) / depth, (-p.z + q.z) / depth);
    vec3 dDepth_dq = -dDepth_dp; // dDepth_dq == -dDepth_dp

    float pwsqinv = 1.0 / (phom.w * phom.w);
    vec3 dp_dx = pwsqinv * (invProjMat[0].xyz * phom.w - phom.x * invProjMat[0][3]);
    vec3 dp_dy = pwsqinv * (invProjMat[1].xyz * phom.w - phom.y * invProjMat[1][3]);
    vec3 dp_dz1 = pwsqinv * (invProjMat[2].xyz * phom.w - phom.z * invProjMat[2][3]);

    float qwsqinv = 1.0 / (qhom.w * qhom.w);
    vec3 dq_dx = qwsqinv * (invProjMat[0].xyz * qhom.w - qhom.x * invProjMat[0][3]);
    vec3 dq_dy = qwsqinv * (invProjMat[1].xyz * qhom.w - qhom.y * invProjMat[1][3]);
    vec3 dq_dz2 = qwsqinv * (invProjMat[2].xyz * qhom.w - qhom.z * invProjMat[2][3]);

    dOut_dx += dOut_dDepth * dot(dDepth_dp, dp_dx);
    dOut_dy += dOut_dDepth * dot(dDepth_dp, dp_dy);
    dOut_dz1 += dOut_dDepth * dot(dDepth_dp, dp_dz1);
    dOut_dx += dOut_dDepth * dot(dDepth_dq, dq_dx);
    dOut_dy += dOut_dDepth * dot(dDepth_dq, dq_dy);
    dOut_dz2 += dOut_dDepth * dot(dDepth_dq, dq_dz2);
}

void addTetTetVertGrads(uint tetIdx, uint tetVertIdx, vec3 dOut_dPi, vec4 dOut_dCi) {
    uint tetGlobalVertIdx = tetsIndices[tetIdx * 4 + tetVertIdx];
    vec4 phom = viewProjMat * vec4(tetsVertexPositions[tetGlobalVertIdx], 1.0);
    float pwsqinv = 1.0 / (phom.w * phom.w);
    vec3 dp_dx = pwsqinv * (viewProjMat[0].xyz * phom.w - phom.x * viewProjMat[0][3]);
    vec3 dp_dy = pwsqinv * (viewProjMat[1].xyz * phom.w - phom.y * viewProjMat[1][3]);
    vec3 dp_dz = pwsqinv * (viewProjMat[2].xyz * phom.w - phom.z * viewProjMat[2][3]);
    vec3 dOut_dTetP = vec3(dot(dOut_dPi, dp_dx), dot(dOut_dPi, dp_dy), dot(dOut_dPi, dp_dz));

    /*
     * For testing an idea similar to the one from the following 3DGS paper:
     * "AbsGS: Recovering Fine Details for 3D Gaussian Splatting". 2024.
     * Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, Yong Dou.
     */
    if (useAbsGrad != 0u) {
        dOut_dCi = abs(dOut_dCi);
        dOut_dTetP = abs(dOut_dTetP);
    }
    atomicAddGradCol(tetVertIdx, dOut_dCi);
    atomicAddGradPos(tetVertIdx, dOut_dTetP);
}

void main() {
    const uint workIdx = gl_GlobalInvocationID.x;
    if (workIdx >= globalTetCounter) {
        return;
    }
    const uint triangleIdxOffset = tetOffsets[workIdx];
    const uint tetIdx = triangleTetIndices[triangleIdxOffset];

    // Read the tet vertex positions and colors from the global buffer.
    // tets have 4 points, 5th point here is used to insert a point in case of intersections
    vec3 tetVertexPosition[4];
    vec4 tetVertexColors[5];
    vec4 tetVertexPositionNdc[5];
    float tetDepths[5];
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        uint tetGlobalVertIdx = tetsIndices[tetIdx * 4 + tetVertIdx];
        tetVertexPosition[tetVertIdx] = tetsVertexPositions[tetGlobalVertIdx];
        tetVertexColors[tetVertIdx] = tetsVertexColors[tetGlobalVertIdx];
        vec4 vertexPosNdc = viewProjMat * vec4(tetsVertexPositions[tetGlobalVertIdx], 1.0);
        vertexPosNdc.xyz /= vertexPosNdc.w;
        vertexPosNdc.w = 1.0;
        tetVertexPositionNdc[tetVertIdx] = vertexPosNdc;
        tetDepths[tetVertIdx] = 0.0;
    }

    // The classic PT algorithm uses face normals to determine the projection class and then do calculations
    // individually. However, Wylie et al. (2002) shows how to use the intersection of two segments to calculate the
    // depth of the thick part for any case. Here, we use face normals to determine which segments to use. One segment
    // should be between two faces that are either both front facing or back facing. Obviously, we only need to test
    // three faces to find two such faces. We test the three faces connected to point 0.
    uint segment1[2];
    uint segment2[2];

    vec2 v1, v2, v3;
    v1.x = tetVertexPositionNdc[1].x - tetVertexPositionNdc[0].x;
    v1.y = tetVertexPositionNdc[1].y - tetVertexPositionNdc[0].y;
    v2.x = tetVertexPositionNdc[2].x - tetVertexPositionNdc[0].x;
    v2.y = tetVertexPositionNdc[2].y - tetVertexPositionNdc[0].y;
    v3.x = tetVertexPositionNdc[3].x - tetVertexPositionNdc[0].x;
    v3.y = tetVertexPositionNdc[3].y - tetVertexPositionNdc[0].y;

    float face_dir1 = v3.x * v2.y - v3.y * v2.x;
    float face_dir2 = v1.x * v3.y - v1.y * v3.x;
    float face_dir3 = v2.x * v1.y - v2.y * v1.x;

    if ((face_dir1 * face_dir2 >= 0.0) &&
        ((face_dir1 != 0.0)       // Handle a special case where 2 faces
          || (face_dir2 != 0.0))) // are perpendicular to the view plane.
    {
        segment1[0] = 0;
        segment1[1] = 3;
        segment2[0] = 1;
        segment2[1] = 2;
    }
    else if (face_dir1 * face_dir3 >= 0.0)
    {
        segment1[0] = 0;
        segment1[1] = 2;
        segment2[0] = 1;
        segment2[1] = 3;
    }
    else // Unless the tet is degenerate, face_dir2*face_dir3 >= 0
    {
        segment1[0] = 0;
        segment1[1] = 1;
        segment2[0] = 2;
        segment2[1] = 3;
    }

    vec3 P1 = tetVertexPositionNdc[segment1[0]].xyz;
    vec3 P2 = tetVertexPositionNdc[segment1[1]].xyz;
    vec3 P3 = tetVertexPositionNdc[segment2[0]].xyz;
    vec3 P4 = tetVertexPositionNdc[segment2[1]].xyz;
    vec4 C1 = tetVertexColors[segment1[0]];
    vec4 C2 = tetVertexColors[segment1[1]];
    vec4 C3 = tetVertexColors[segment2[0]];
    vec4 C4 = tetVertexColors[segment2[1]];

    // Find the intersection of the projection of the two segments in the XY plane.
    // This algorithm is based on that given in Graphics Gems III, pg. 199-202.
    vec3 A, B, C;
    // We can define the two lines parametrically as:
    //        P1 + alpha(A)
    //        P3 + beta(B)
    // where A = P2 - P1
    // and   B = P4 - P3.
    // alpha and beta are in the range [0,1] within the line segment.
    A = P2 - P1;
    B = P4 - P3;
    // The lines intersect when the values of the two parameteric equations are equal.
    // Setting them equal and moving everything to one side:
    //        0 = C + beta(B) - alpha(A)
    // where C = P3 - P1.
    C = P3 - P1;
    // When we project the lines to the xy plane (which we do by throwing away the z value), we have two equations and
    // two unknowns. The following are the solutions for alpha and beta.
    float denominator = (A.x * B.y - A.y * B.x);
    //if (denominator == 0)
    if (abs(denominator) < 1e-8)
        return; // Must be degenerated tetrahedra.
    float alpha = (B.y * C.x - B.x * C.y) / denominator;
    float beta = (A.y * C.x - A.x * C.y) / denominator;

    // Generate memory for the projected triangles in the linear append buffer.
    uint numGeneratedTris;
    if ((alpha >= 0.0) && (alpha <= 1.0)) {
        numGeneratedTris = 4;
    } else {
        numGeneratedTris = 3;
    }

    uint indices[6]; // Size 6 or 5
    if ((alpha >= 0.0) && (alpha <= 1.0))
    {
        // The two segments intersect.  This corresponds to class 2 in
        // Shirley and Tuchman (or one of the degenerate cases).

        // Make new point at intersection.
        tetVertexPositionNdc[4].xyz = P1 + alpha * A;
        tetVertexPositionNdc[4].w = 1.0;

        // Find depth at intersection.
        float depth = GetCorrectedDepth(
                tetVertexPositionNdc[4].x, tetVertexPositionNdc[4].y, tetVertexPositionNdc[4].z, P3.z + beta * B.z);

        // Find color and opacity at intersection.
        tetVertexColors[4] = (0.5 * (C1 + alpha * (C2 - C1) + C3 + beta * (C4 - C3)));

        // Record the depth at the intersection.
        tetDepths[4] = depth * attenuationCoefficient;

        // Establish the order in which the points should be rendered.
        indices[0] = 4;
        indices[1] = segment1[0];
        indices[2] = segment2[0];
        indices[3] = segment1[1];
        indices[4] = segment2[1];
        indices[5] = segment1[0];
    }
    else
    {
        // The two segments do not intersect.  This corresponds to class 1
        // in Shirley and Tuchman.
        if (alpha <= 0.0)
        {
            // Flip segment1 so that alpha is >= 1. P1 and P2 are also flipped as are C1-C2 and T1-T2.
            // Note that this will invalidate A. B and beta are unaffected.
            //std::swap(segment1[0], segment1[1]);
            uint tmp = segment1[0];
            segment1[0] = segment1[1];
            segment1[1] = tmp;
            alpha = 1.0 - alpha;
            vec3 tmpVec3;
            tmpVec3 = P1;
            P1 = P2;
            P2 = tmpVec3;
            vec4 tmpVec4;
            tmpVec4 = C1;
            C1 = C2;
            C2 = tmpVec4;
        }
        // From here on, we can assume P2 is the "thick" point.

        // Find the depth under the thick point.  Use the alpha and beta from intersection to determine location of face
        // under thick point.
        float edgez = P3.z + beta * B.z;
        float pointz = P1.z;
        float facez = (edgez + (alpha - 1.0) * pointz) / alpha;
        float depth = GetCorrectedDepth(P2.x, P2.y, P2.z, facez);

        // Fix color and opacity at thick point. Average color/opacity with color/opacity of opposite face.
        vec4 edgec = C3 + beta * (C4 - C3);
        vec4 pointc = C1;
        vec4 facec = (edgec + (alpha - 1.0) * pointc) / alpha;
        tetVertexColors[segment1[1]] = (0.5 * (facec + C2));

        // Record thickness at thick point.
        tetDepths[segment1[1]] = depth * attenuationCoefficient;

        // Establish the order in which the points should be rendered.
        indices[0] = segment1[1];
        indices[1] = segment1[0];
        indices[2] = segment2[0];
        indices[3] = segment2[1];
        indices[4] = segment1[0];
    }

    // Add the cells to the buffer.
    //uint i0 = indices[0];
    //for (int cellIdx = 0; cellIdx < numGeneratedTris; cellIdx++)
    //{
    //    uint i1 = indices[cellIdx + 1];
    //    uint i2 = indices[cellIdx + 2];
    //
    //    uint vertexOffset = triOffset * 3u;
    //    vertexPositions[vertexOffset] = vec4(tetVertexPositionNdc[i0].xyz, 1.0);
    //    vertexPositions[vertexOffset + 1] = vec4(tetVertexPositionNdc[i1].xyz, 1.0);
    //    vertexPositions[vertexOffset + 2] = vec4(tetVertexPositionNdc[i2].xyz, 1.0);
    //    vertexColors[vertexOffset] = tetVertexColors[i0];
    //    vertexColors[vertexOffset + 1] = tetVertexColors[i1];
    //    vertexColors[vertexOffset + 2] = tetVertexColors[i2];
    //    vertexDepths[vertexOffset] = tetDepths[i0];
    //    vertexDepths[vertexOffset + 1] = tetDepths[i1];
    //    vertexDepths[vertexOffset + 2] = tetDepths[i2];
    //    triOffset++;
    //}


    // Now do adjoint pass.
    uint vertexOffset = triangleIdxOffset;
    vec3 dOut_dTriP[5];
    vec4 dOut_dTriC[5];
    float dOut_dTriD[5];
    [[unroll]] for (uint i = 0; i < 5; i++) {
        dOut_dTriP[i] = vec3(0.0);
        dOut_dTriC[i] = vec4(0.0);
        dOut_dTriD[i] = 0.0;
    }
    uint i0 = indices[0];
    for (int cellIdx = 0; cellIdx < numGeneratedTris; cellIdx++)
    {
        uint i1 = indices[cellIdx + 1];
        uint i2 = indices[cellIdx + 2];

        //vertexPositions[vertexOffset] = vec4(tetVertexPositionNdc[i0].xyz, 1.0);
        //vertexPositions[vertexOffset + 1] = vec4(tetVertexPositionNdc[i1].xyz, 1.0);
        //vertexPositions[vertexOffset + 2] = vec4(tetVertexPositionNdc[i2].xyz, 1.0);
        //vertexColors[vertexOffset] = tetVertexColors[i0];
        //vertexColors[vertexOffset + 1] = tetVertexColors[i1];
        //vertexColors[vertexOffset + 2] = tetVertexColors[i2];
        //vertexDepths[vertexOffset] = tetDepths[i0];
        //vertexDepths[vertexOffset + 1] = tetDepths[i1];
        //vertexDepths[vertexOffset + 2] = tetDepths[i2];
        dOut_dTriP[i0] += triangleVertexPositionGradients[vertexOffset].xyz;
        dOut_dTriP[i1] += triangleVertexPositionGradients[vertexOffset + 1].xyz;
        dOut_dTriP[i2] += triangleVertexPositionGradients[vertexOffset + 2].xyz;
        dOut_dTriC[i0] += triangleVertexColorGradients[vertexOffset];
        dOut_dTriC[i1] += triangleVertexColorGradients[vertexOffset + 1];
        dOut_dTriC[i2] += triangleVertexColorGradients[vertexOffset] + 2;
        dOut_dTriD[i0] += triangleVertexDepthGradients[vertexOffset];
        dOut_dTriD[i1] += triangleVertexDepthGradients[vertexOffset + 1];
        dOut_dTriD[i2] += triangleVertexDepthGradients[vertexOffset] + 2;
        vertexOffset += 3;
    }

    //vec4 C1 = tetVertexColors[segment1[0]];
    //vec4 C2 = tetVertexColors[segment1[1]];
    //vec4 C3 = tetVertexColors[segment2[0]];
    //vec4 C4 = tetVertexColors[segment2[1]];
    vec3 dOut_dP1 = dOut_dTriP[segment1[0]];
    vec3 dOut_dP2 = dOut_dTriP[segment1[1]];
    vec3 dOut_dP3 = dOut_dTriP[segment2[0]];
    vec3 dOut_dP4 = dOut_dTriP[segment2[1]];
    vec4 dOut_dC1 = dOut_dTriC[segment1[0]];
    vec4 dOut_dC2 = dOut_dTriC[segment1[1]];
    vec4 dOut_dC3 = dOut_dTriC[segment2[0]];
    vec4 dOut_dC4 = dOut_dTriC[segment2[1]];
    vec3 dOut_dA = vec3(0.0);
    vec3 dOut_dB = vec3(0.0);
    vec3 dOut_dC = vec3(0.0);
    float dOut_dalpha = 0.0;
    float dOut_dbeta = 0.0;

    if ((alpha >= 0.0) && (alpha <= 1.0))
    {
        // Record the depth at the intersection.
        //tetDepths[4] = depth * attenuationCoefficient;
        float dOut_dDepth = dOut_dTriD[4] * attenuationCoefficient;

        // Find color and opacity at intersection.
        //tetVertexColors[4] = (0.5 * (C1 + alpha * (C2 - C1) + C3 + beta * (C4 - C3)));
        // tmp: vec4 dTriC4_dalpha = 0.5 * (C2 - C1);
        // tmp: vec4 dTriC4_dbeta = 0.5 * (C4 - C3);
        // tmp: float dTriC4_dC1 = 0.5 * (1.0 - alpha);
        // tmp: float dTriC4_dC2 = 0.5 * alpha;
        // tmp: float dTriC4_dC3 = 0.5 * (1.0 - beta);
        // tmp: float dTriC4_dC4 = 0.5 * beta;
        dOut_dalpha += dot(dOut_dTriC[4], 0.5 * (C2 - C1));
        dOut_dbeta += dot(dOut_dTriC[4], 0.5 * (C4 - C3));
        dOut_dC1 += dOut_dTriC[4] * (0.5 * (1.0 - alpha));
        dOut_dC2 += dOut_dTriC[4] * (0.5 * alpha);
        dOut_dC3 += dOut_dTriC[4] * (0.5 * (1.0 - beta));
        dOut_dC4 += dOut_dTriC[4] * (0.5 * beta);

        // Find depth at intersection.
        //float depth = GetCorrectedDepth(
        //tetVertexPositionNdc[4].x, tetVertexPositionNdc[4].y, tetVertexPositionNdc[4].z, P3.z + beta * B.z);
        float dOut_z2 = 0.0;
        GetCorrectedDepthAdjoint(
                tetVertexPositionNdc[4].x, tetVertexPositionNdc[4].y, tetVertexPositionNdc[4].z, P3.z + beta * B.z,
                dOut_dDepth, dOut_dP2.x, dOut_dP2.y, dOut_dP2.z, dOut_z2);
        dOut_dbeta += dOut_z2 * B.z;
        dOut_dP3.z += dOut_z2;
        dOut_dB.z += dOut_z2 * beta;

        // Make new point at intersection.
        //tetVertexPositionNdc[4].xyz = P1 + alpha * A;
        //tetVertexPositionNdc[4].w = 1.0;
        dOut_dP1 += dOut_dTriP[4];
        dOut_dalpha += dot(dOut_dTriP[4], A);
        dOut_dA += dOut_dTriP[4] * alpha;
    }
    else
    {
        // Record thickness at thick point.
        //tetDepths[segment1[1]] = depth * attenuationCoefficient;
        float dOut_dDepth = dOut_dTriD[segment1[1]] * attenuationCoefficient;

        // Fix color and opacity at thick point. Average color/opacity with color/opacity of opposite face.
        //tetVertexColors[segment1[1]] = (0.5 * (facec + C2));
        vec4 dOut_dfacec = dOut_dTriC[segment1[1]] * 0.5;
        vec4 dOut_dC2 = dOut_dTriC[segment1[1]] * 0.5;

        //vec4 pointc = C1;
        //vec4 facec = (edgec + (alpha - 1.0) * pointc) / alpha;
        // tmp: vec4 dfacec_dalpha = (-edgec - pointc) / (alpha * alpha);
        // tmp: float dfacec_dC1 = 1.0 - 1.0 / alpha; // pointc == C1
        vec4 edgec = C3 + beta * (C4 - C3);
        dOut_dalpha += dot(dOut_dfacec, (-edgec - C1) / (alpha * alpha));
        dOut_dC1 += dOut_dfacec * (1.0 - 1.0 / alpha); // pointc == C1
        float dfacec_dedgec = 1.0 / alpha;

        //vec4 edgec = C3 + beta * (C4 - C3);
        // tmp: float dedgec_dC3 = 1.0 - beta;
        // tmp: vec4 dedgec_dbeta = C4 - C3;
        // tmp: float dedgec_dC4 = beta;
        dOut_dC3 += dOut_dfacec * dfacec_dedgec * (1.0 - beta);
        dOut_dbeta += dfacec_dedgec * dot(dOut_dfacec, C4 - C3);
        dOut_dC4 += dOut_dfacec * (dfacec_dedgec * beta);

        //float depth = GetCorrectedDepth(P2.x, P2.y, P2.z, facez);
        float edgez = P3.z + beta * B.z;
        float facez = (edgez + (alpha - 1.0) * P1.z) / alpha;
        float dOut_dfacez = 0.0;
        GetCorrectedDepthAdjoint(P2.x, P2.y, P2.z, facez, dOut_dDepth, dOut_dP2.x, dOut_dP2.y, dOut_dP2.z, dOut_dfacez);

        // Find the depth under the thick point.  Use the alpha and beta from intersection to determine location of face
        // under thick point.
        //float pointz = P1.z;
        //float facez = (edgez + (alpha - 1.0) * pointz) / alpha;
        float dfacez_dalpha = (-edgez - P1.z) / (alpha * alpha);
        dOut_dalpha += dOut_dfacez * dfacez_dalpha;
        dOut_dP1.z += dOut_dfacez * (1.0 - 1.0 / alpha); // pointz == P1.z
        float dfacez_dedgez = 1.0 / alpha;
        float dOut_dedgez = dOut_dfacez * dfacez_dedgez;
        //float edgez = P3.z + beta * B.z;
        dOut_dP3.z += dOut_dedgez;
        dOut_dB.z += dOut_dedgez * beta;
        dOut_dbeta += dOut_dedgez * B.z;

        // The two segments do not intersect.  This corresponds to class 1
        // in Shirley and Tuchman.
        //if (alpha <= 0.0)
        //{
        //    // Flip segment1 so that alpha is >= 1. P1 and P2 are also flipped as are C1-C2 and T1-T2.
        //    // Note that this will invalidate A. B and beta are unaffected.
        //    //std::swap(segment1[0], segment1[1]);
        //    uint tmp = segment1[0];
        //    segment1[0] = segment1[1];
        //    segment1[1] = tmp;
        //    alpha = 1.0 - alpha;
        //    vec3 tmpVec3;
        //    tmpVec3 = P1;
        //    P1 = P2;
        //    P2 = tmpVec3;
        //    vec4 tmpVec4;
        //    tmpVec4 = C1;
        //    C1 = C2;
        //    C2 = tmpVec4;
        //}
        if (alpha >= 1.0)
        {
            uint tmp = segment1[0];
            segment1[0] = segment1[1];
            segment1[1] = tmp;
            alpha = 1.0 - alpha;
            dOut_dalpha = -dOut_dalpha;
            vec3 tmpVec3;
            tmpVec3 = P1;
            P1 = P2;
            P2 = tmpVec3;
            tmpVec3 = dOut_dP1;
            dOut_dP1 = dOut_dP2;
            dOut_dP2 = tmpVec3;
            vec4 tmpVec4;
            tmpVec4 = C1;
            C1 = C2;
            C2 = tmpVec4;
            tmpVec4 = dOut_dC1;
            dOut_dC1 = dOut_dC2;
            dOut_dC2 = tmpVec4;
        }
    }

    //float denominator = (A.x * B.y - A.y * B.x);
    //float alpha = (B.y * C.x - B.x * C.y) / denominator;
    //float beta = (A.y * C.x - A.x * C.y) / denominator;
    float T0 = A.x*B.y - A.y*B.x;
    float T0invsq = 1.0 / (T0 * T0);
    float T1 = B.x*C.y - B.y*C.x;
    float T2 = A.x*C.y - A.y*C.x;
    vec2 dalpha_dA, dalpha_dB, dalpha_dC, dbeta_dA, dbeta_dB, dbeta_dC;
    dalpha_dA.x = B.y * T0 / T0invsq;
    dalpha_dA.y = -B.x * T0 / T0invsq;
    dalpha_dB.x = (-A.y * T0 - C.y * T0) / T0invsq;
    dalpha_dB.y = (A.x * T0 + C.x * T0) / T0invsq;
    dalpha_dC.x = B.y / T0;
    dalpha_dC.y = -B.x / T0;
    dbeta_dA.x = (B.y * T2 - C.y * T0) / T0invsq;
    dbeta_dA.y = (-B.x * T2 + C.x * T0) / T0invsq;
    dbeta_dB.x = -A.y * T2 / T0invsq;
    dbeta_dB.y = A.x * T2 / T0invsq;
    dbeta_dC.x = A.y / T0;
    dbeta_dC.y = -A.x / T0;
    dOut_dA += dOut_dalpha * vec3(dalpha_dA, 0.0);
    dOut_dB += dOut_dalpha * vec3(dalpha_dB, 0.0);
    dOut_dC += dOut_dalpha * vec3(dalpha_dC, 0.0);
    dOut_dA += dOut_dbeta * vec3(dbeta_dA, 0.0);
    dOut_dB += dOut_dbeta * vec3(dbeta_dB, 0.0);
    dOut_dC += dOut_dbeta * vec3(dbeta_dC, 0.0);

    //A = P2 - P1;
    //B = P4 - P3;
    //C = P3 - P1;
    dOut_dP2 += dOut_dA;
    dOut_dP1 -= dOut_dA;
    dOut_dP4 += dOut_dB;
    dOut_dP3 -= dOut_dB;
    dOut_dP3 += dOut_dC;
    dOut_dP1 -= dOut_dC;

    debugPrintfEXT("t %f %f %f", dOut_dTriP[0].x, dOut_dTriC[0].x, dOut_dTriD[0].x);
    debugPrintfEXT("p %f %f %f %f", dOut_dP1.x, dOut_dP2.x, dOut_dP3.x, dOut_dP4.x);
    debugPrintfEXT("c %f %f %f %f", dOut_dC1.x, dOut_dC2.x, dOut_dC3.x, dOut_dC4.x);
    debugPrintfEXT("ABC %f %f %f", dOut_dA.x, dOut_dB.x, dOut_dC.x);
    debugPrintfEXT("ab %f %f", dOut_dalpha, dOut_dbeta);

    //vec3 P1 = tetVertexPositionNdc[segment1[0]].xyz;
    //vec3 P2 = tetVertexPositionNdc[segment1[1]].xyz;
    //vec3 P3 = tetVertexPositionNdc[segment2[0]].xyz;
    //vec3 P4 = tetVertexPositionNdc[segment2[1]].xyz;
    //vec4 C1 = tetVertexColors[segment1[0]];
    //vec4 C2 = tetVertexColors[segment1[1]];
    //vec4 C3 = tetVertexColors[segment2[0]];
    //vec4 C4 = tetVertexColors[segment2[1]];
    //[[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
    //    uint tetGlobalVertIdx = tetsIndices[tetIdx * 4 + tetVertIdx];
    //    tetVertexPosition[tetVertIdx] = tetsVertexPositions[tetGlobalVertIdx];
    //    tetVertexColors[tetVertIdx] = tetsVertexColors[tetGlobalVertIdx];
    //    vec4 vertexPosNdc = viewProjMat * vec4(tetsVertexPositions[tetGlobalVertIdx], 1.0);
    //    vertexPosNdc.xyz /= vertexPosNdc.w;
    //    vertexPosNdc.w = 1.0;
    //    tetVertexPositionNdc[tetVertIdx] = vertexPosNdc;
    //    tetDepths[tetVertIdx] = 0.0;
    //}
    uint tetVertexIdx;
    tetVertexIdx = segment1[0];
    addTetTetVertGrads(tetIdx, tetVertexIdx, dOut_dP1, dOut_dC1);
    tetVertexIdx = segment1[1];
    addTetTetVertGrads(tetIdx, tetVertexIdx, dOut_dP2, dOut_dC2);
    tetVertexIdx = segment2[0];
    addTetTetVertGrads(tetIdx, tetVertexIdx, dOut_dP3, dOut_dC3);
    tetVertexIdx = segment2[1];
    addTetTetVertGrads(tetIdx, tetVertexIdx, dOut_dP4, dOut_dC4);
}
