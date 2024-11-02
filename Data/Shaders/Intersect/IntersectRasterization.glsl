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
layout(binding = 2, std430) readonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};

layout(location = 0) in vec3 vertexPositionWorld;
layout(location = 1) flat in uint tetIdx;

void main() {
    uint triangleIdx = triangleKeyValues[gl_VertexIndex / 3u].index;
    uint vertexIdx = triangleIdx * 3u + (gl_VertexIndex % 3u);
    vertexPositionWorld = invViewProjMatrix * vertexPositions[vertexIdx];
    tetIdx = triangleTetIndices[triangleIdx];
    gl_Position = vertexPositions[vertexIdx];
}


-- Fragment

#version 450

// Input tet data.
layout(binding = 2, std430) readonly buffer TetIndexBuffer {
    uint tetsIndices[];
};
layout(binding = 3, scalar) readonly buffer TetVertexPositionBuffer {
    vec3 tetsVertexPositions[];
};
layout(binding = 4, scalar) readonly buffer TetVertexColorBuffer {
    vec4 tetsVertexColors[];
};

#ifndef SHOW_TET_QUALITY
layout(binding = 6, scalar) readonly buffer VertexColorBuffer {
    vec4 vertexColors[];
};
#else
#define INVALID_TET 0xFFFFFFFFu
layout(binding = 6, scalar) readonly buffer FaceToTetMapBuffer {
    uvec2 faceToTetMap[];
};
layout(binding = 7, scalar) readonly buffer TetQualityBuffer {
    float tetQualityArray[];
};
layout (binding = 8) uniform MinMaxUniformBuffer {
    float minAttributeValue;
    float maxAttributeValue;
};
layout(binding = 9) uniform sampler1D transferFunctionTexture;
vec4 transferFunction(float attr) {
    // Transfer to range [0, 1].
    float posFloat = clamp((attr - minAttributeValue) / (maxAttributeValue - minAttributeValue), 0.0, 1.0);
    // Look up the color value.
    return texture(transferFunctionTexture, posFloat);
}
#endif

layout(location = 0) in vec3 vertexPositionWorld;
layout(location = 1) flat in uint tetIdx;
layout(location = 0) out vec4 outputColor;

const int tetFaceTable[4][4] = {
        { 0, 1, 2, 3 }, // Last index is point opposite to face.
        { 1, 0, 3, 2 }, // Last index is point opposite to face.
        { 0, 2, 3, 1 }, // Last index is point opposite to face.
        { 2, 1, 3, 0 }, // Last index is point opposite to face.
};

// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
const float RAY_INTERSECTION_EPSILON = 1e-6;
bool rayTriangleIntersect(vec3 ro, vec3 rd, vec3 p0, vec3 p1, vec3 p2, out float t) {
    // Compute plane normal.
    vec3 p0p1 = p1 - p0;
    vec3 p0p2 = p2 - p0;
    vec3 planeNormal = cross(p0p1, p0p2);

    // Check if the plane is parallel to the ray direction.
    float cosNormalRayDir = dot(planeNormal, rd);
    if (abs(cosNormalRayDir) < RAY_INTERSECTION_EPSILON) {
        return false;
    }

    float d = -dot(planeNormal, p0);
    t = -(dot(planeNormal, ro) + d) / cosNormalRayDir;

    // Check if the triangle intersection is behind the ray origin.
    if (t < 0.0) {
        return false;
    }

    vec3 intersectionPoint = ro + t * rd;

    // Test whether intersection point is inside of edge p0p1.
    vec3 p0p = intersectionPoint - p0;
    vec3 normalEdge = cross(p0p1, p0p);
    if (dot(planeNormal, normalEdge) < 0.0) {
        return false;
    }

    // Test whether intersection point is inside of edge p2p1.
    vec3 p2p1 = p2 - p1;
    vec3 p1p = intersectionPoint - p1;
    normalEdge = cross(p2p1, p1p);
    if (dot(planeNormal, normalEdge) < 0.0) {
        return false;
    }

    // Test whether intersection point is inside of edge p2p0.
    vec3 p2p0 = p0 - p2;
    vec3 p2p = intersectionPoint - p2;
    normalEdge = cross(p2p0, p2p);
    if (dot(planeNormal, normalEdge) < 0.0) {
        return false;
    }

    return true;
}

bool intersectRayTet(
        vec3 ro, vec3 rd, vec3 p[4],
        out uint f0, out uint f1, out float t0, out float t1) {
    t0 = 1e9;
    t1 = -1e9;
    float t;
    [[unroll]] for (uint tetFaceIdx = 0; tetFaceIdx < 4; tetFaceIdx++) {
        vec3 p0 = tetVertexPositions[tetFaceTable[tetFaceIdx][0]];
        vec3 p1 = tetVertexPositions[tetFaceTable[tetFaceIdx][1]];
        vec3 p2 = tetVertexPositions[tetFaceTable[tetFaceIdx][2]];
        t = 1e9;
        if (rayTriangleIntersect(ro, rd, p0, p1, p2, t)) {
            if (t < t0) {
                f0 = tetFaceIdx;
                t0 = t;
            }
            if (t > t1) {
                f1 = tetFaceIdx;
                t1 = t;
            }
        } else if (t < 0.0) {
            f0 = tetFaceIdx;
            t0 = 0.0;
        }
    }
    return t1 >= t0;
}

#include "ForwardCommon.glsl"

void main() {
    vec3 tetVertexPositions[4];
    vec4 tetVertexColors[4];
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        uint tetGlobalVertIdx = tetsIndices[tetOffset + tetVertIdx];
        tetVertexPositions[tetVertIdx] = tetsVertexPositions[tetGlobalVertIdx];
        tetVertexColors[tetVertIdx] = tetsVertexColors[tetGlobalVertIdx];
    }

    uint f0, f1; //< Face index hit 0, 1.
    float t0, t1; //< Ray distance hit 0, 1.
    // needs to return: face idx, bary coords (back and front)
    if (!intersectRayTet(ro, rd, tetVertexPositions, f0, f1, t0, t1)) {
        discard; // Should never happen...
    }

    vec3 intersectPos0 = ro + t0 * rd;
    vec3 intersectPos1 = ro + t1 * rd;

    // Barycentric interpolation (face 0 and 1).
    i0 = tetFaceTable[f0][0];
    i1 = tetFaceTable[f0][1];
    i2 = tetFaceTable[f0][2];
    p0 = tetVertexPositions[i0];
    p1 = tetVertexPositions[i1];
    p2 = tetVertexPositions[i2];
    c0 = tetVertexColors[i0];
    c1 = tetVertexColors[i1];
    c2 = tetVertexColors[i2];
    vec3 barycentricCoordinates = barycentricInterpolation(p0, p1, p2, intersectPos0);
    vec4 fragment1Color = c0 * barycentricCoordinates.x + c1 * barycentricCoordinates.y + c2 * barycentricCoordinates.z;

    i0 = tetFaceTable[f1][0];
    i1 = tetFaceTable[f1][1];
    i2 = tetFaceTable[f1][2];
    p0 = tetVertexPositions[i0];
    p1 = tetVertexPositions[i1];
    p2 = tetVertexPositions[i2];
    c0 = tetVertexColors[i0];
    c1 = tetVertexColors[i1];
    c2 = tetVertexColors[i2];
    barycentricCoordinates = barycentricInterpolation(p0, p1, p2, intersectPos1);
    vec4 fragment2Color = c0 * barycentricCoordinates.x + c1 * barycentricCoordinates.y + c2 * barycentricCoordinates.z;

#ifdef USE_SUBDIVS
    vec4 rayColor = vec4(0.0);
    float t = t1 - t0;
    tSeg = t / float(NUM_SUBDIVS);
    const float INV_N_SUB = 1.0 / float(NUM_SUBDIVS);
    for (int s = 0; s < NUM_SUBDIVS; s++) {
        float fbegin = (float(s)) * INV_N_SUB;
        float fmid = (float(s) + 0.5) * INV_N_SUB;
        float fend = (float(s) + 1.0) * INV_N_SUB;
        vec3 c0 = mix(fragment1Color.rgb, fragment2Color.rgb, fbegin);
        vec3 c1 = mix(fragment1Color.rgb, fragment2Color.rgb, fend);
        float alpha = mix(fragment1Color.a, fragment2Color.a, fmid);
        currentColor = accumulateLinearConst(tSeg, c0, c1, alpha * attenuationCoefficient);
        rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
        rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
    }
#else
    vec4 rayColor;
    currentColor = accumulateLinear(
            t, fragment1Color.rgb, fragment2Color.rgb,
            fragment1Color.a * attenuationCoefficient, fragment2Color.a * attenuationCoefficient);
    rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
    rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
#endif

    outputColor = rayColor;
}
