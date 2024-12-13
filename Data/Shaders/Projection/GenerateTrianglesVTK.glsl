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

layout(local_size_x = BLOCK_SIZE) in;

#include "ProjectionUniform.glsl"
#include "TetFaceTable.glsl"
#include "ForwardCommon.glsl"
#include "RayCommon.glsl"
#ifdef USE_CLIP_PLANE
#include "ClipPlane.glsl"
#endif

// Atomically increased linear append index.
layout(binding = 1, std430) buffer TriangleCounterBuffer {
    uint globalTriangleCounter;
};

// Input tet data.
layout(binding = 2, std430) readonly buffer TetIndexBuffer {
    uint tetsIndices[];
};
layout(binding = 3, scalar) readonly buffer TetVertexPositionBuffer {
    vec3 tetsVertexPositions[];
};
layout(binding = 4, std430) readonly buffer TetVertexColorBuffer {
    vec4 tetsVertexColors[];
};

// Output triangle data.
layout(binding = 5, scalar) writeonly buffer TriangleVertexPositionBuffer {
    vec4 vertexPositions[];
};
layout(binding = 6, std430) writeonly buffer TriangleVertexColorBuffer {
    vec4 vertexColors[];
};
layout(binding = 7, std430) writeonly buffer TriangleVertexDepthBuffer {
    float vertexDepths[];
};
#ifdef SUPPORT_ADJOINT
layout(binding = 8, std430) writeonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};
#endif

float GetCorrectedDepth(float x, float y, float z1, float z2) {
    if (useLinearDepthCorrection != 0u) {
        float depth = linearDepthCorrection * (z1 - z2);
        if (depth < 0.0) {
            depth = -depth;
        }
        return depth;
    } else {
        /*vec3 eye1, eye2;
        float invw;

        // This code does the same as the commented code above, but also collects
        // common arithmetic between the two matrix x vector operations.  An
        // optimizing compiler may or may not pick up on that.
        vec4 commonVec;

        commonVec[0] = (invProjMat[0] * x + invProjMat[4] * y + invProjMat[12]);
        commonVec[1] = (invProjMat[1] * x + invProjMat[5] * y + invProjMat[13]);
        commonVec[2] = (invProjMat[2] * x + invProjMat[6] * y + invProjMat[10] * z1 + invProjMat[14]);
        commonVec[3] = (invProjMat[3] * x + invProjMat[7] * y + invProjMat[15]);

        invw = 1.0 / (commonVec[3] + invProjMat[11] * z1);
        eye1[0] = invw * (commonVec[0] + invProjMat[8] * z1);
        eye1[1] = invw * (commonVec[1] + invProjMat[9] * z1);
        eye1[2] = invw * (commonVec[2] + invProjMat[10] * z1);

        invw = 1.0 / (commonVec[3] + invProjMat[11] * z2);
        eye2[0] = invw * (commonVec[0] + invProjMat[8] * z2);
        eye2[1] = invw * (commonVec[1] + invProjMat[9] * z2);
        eye2[2] = invw * (commonVec[2] + invProjMat[10] * z2);*/

        vec4 eye1 = invProjMat * vec4(x, y, z1, 1.0);
        vec4 eye2 = invProjMat * vec4(x, y, z2, 1.0);

        return distance(eye1.xyz / eye1.w, eye2.xyz / eye2.w);
    }
}

void main() {
    const uint tetIdx = gl_GlobalInvocationID.x;
    if (tetIdx >= numTets) {
        return;
    }

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

    // Do not render this cell if it is outside of the cutting planes. For most planes, cut if all points are outside.
    // For the near plane, cut if any points are outside because things can go very wrong if one of the points is behind
    // the view.
    if (((tetVertexPositionNdc[0].x > 1.0) && (tetVertexPositionNdc[1].x > 1.0) &&
            (tetVertexPositionNdc[2].x > 1.0) && (tetVertexPositionNdc[3].x > 1.0)) ||
        ((tetVertexPositionNdc[0].x < -1.0) && (tetVertexPositionNdc[1].x < -1.0) &&
            (tetVertexPositionNdc[2].x < -1.0) && (tetVertexPositionNdc[3].x < -1.0)) ||
        ((tetVertexPositionNdc[0].y > 1.0) && (tetVertexPositionNdc[1].y > 1.0) &&
            (tetVertexPositionNdc[2].y > 1.0) && (tetVertexPositionNdc[3].y > 1.0)) ||
        ((tetVertexPositionNdc[0].y < -1.0) && (tetVertexPositionNdc[1].y < -1.0) &&
            (tetVertexPositionNdc[2].y < -1.0) && (tetVertexPositionNdc[3].y < -1.0)) ||
        ((tetVertexPositionNdc[0].z > 1.0) && (tetVertexPositionNdc[1].z > 1.0) &&
            (tetVertexPositionNdc[2].z > 1.0) && (tetVertexPositionNdc[3].z > 1.0)) ||
        ((tetVertexPositionNdc[0].z < 0.0) || (tetVertexPositionNdc[1].z < 0.0) ||
            (tetVertexPositionNdc[2].z < 0.0) || (tetVertexPositionNdc[3].z < 0.0))) {
        return;
    }

#ifdef USE_CLIP_PLANE
    bool allOutside = true;
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        if (!checkIsPointOutsideClipPlane(tetVertexPosition[tetVertIdx])) {
            allOutside = false;
        }
    }
    if (allOutside) {
        return;
    }
#endif

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
    uint triOffset = atomicAdd(globalTriangleCounter, numGeneratedTris);

    /*if (numGeneratedTris != 4) {
        for (uint i = 0; i < 5; i++) {
            tetVertexColors[i].rgb = vec3(1.0, 0.0, 0.0);
        }
        C1 = tetVertexColors[segment1[0]];
        C2 = tetVertexColors[segment1[1]];
        C3 = tetVertexColors[segment2[0]];
        C4 = tetVertexColors[segment2[1]];
    }*/


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
    uint i0 = indices[0];
    for (int cellIdx = 0; cellIdx < numGeneratedTris; cellIdx++)
    {
        uint i1 = indices[cellIdx + 1];
        uint i2 = indices[cellIdx + 2];

        uint vertexOffset = triOffset * 3u;
        vertexPositions[vertexOffset] = vec4(tetVertexPositionNdc[i0].xyz, 1.0);
        vertexPositions[vertexOffset + 1] = vec4(tetVertexPositionNdc[i1].xyz, 1.0);
        vertexPositions[vertexOffset + 2] = vec4(tetVertexPositionNdc[i2].xyz, 1.0);
        vertexColors[vertexOffset] = tetVertexColors[i0];
        vertexColors[vertexOffset + 1] = tetVertexColors[i1];
        vertexColors[vertexOffset + 2] = tetVertexColors[i2];
        vertexDepths[vertexOffset] = tetDepths[i0];
        vertexDepths[vertexOffset + 1] = tetDepths[i1];
        vertexDepths[vertexOffset + 2] = tetDepths[i2];
#ifdef SUPPORT_ADJOINT
        triangleTetIndices[triOffset] = tetIdx;
#endif
        triOffset++;

        //indexArray.push_back(indices[0] + numPts);
        //indexArray.push_back(indices[cellIdx + 1] + numPts);
        //indexArray.push_back(indices[cellIdx + 2] + numPts);
    }
}
