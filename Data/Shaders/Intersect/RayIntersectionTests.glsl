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

// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
const float RAY_INTERSECTION_EPSILON = 1e-9;
bool rayTriangleIntersect(vec3 ro, vec3 rd, vec3 p0, vec3 p1, vec3 p2, out float t, inout bool isInside) {
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

    isInside = true;

    // Check if the triangle intersection is in front of the ray origin.
    return t >= 0.0;
}

bool intersectRayTet(
        vec3 ro, vec3 rd, vec3 tetVertexPositions[4],
        out uint f0, out uint f1, out float t0, out float t1) {
    t0 = 1e9;
    t1 = -1e9;
    float t;
    bool isInside;
    [[unroll]] for (uint tetFaceIdx = 0; tetFaceIdx < 4; tetFaceIdx++) {
        vec3 p0 = tetVertexPositions[tetFaceTable[tetFaceIdx][0]];
        vec3 p1 = tetVertexPositions[tetFaceTable[tetFaceIdx][1]];
        vec3 p2 = tetVertexPositions[tetFaceTable[tetFaceIdx][2]];
        t = 1e9;
        isInside = false;
        bool intersectsTri = rayTriangleIntersect(ro, rd, p0, p1, p2, t, isInside);
        if (intersectsTri) {
            if (t < t0) {
                f0 = tetFaceIdx;
                t0 = t;
            }
            if (t > t1) {
                f1 = tetFaceIdx;
                t1 = t;
            }
        } else if (isInside && t < 0.0) {
            f0 = tetFaceIdx;
            t0 = 0.0;
        }
    }
    return t1 >= t0;
}
