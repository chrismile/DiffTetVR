/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2020 - 2021, Christoph Neuhauser
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

// Use early z-test to cull transparent fragments occluded by opaque fragments.
// Additionaly, use fragment interlock.
layout(early_fragment_tests) in;

in vec4 gl_FragCoord;

// A fragment node stores rendering information about one specific fragment
struct LinkedListFragmentNode {
    // Bit 0: isFrontFacing, Bit 1: isBoundary, Bit 2-31: faceIndex.
    uint color;
    // Depth value of the fragment
    float depth;
    // The index of the next node in "nodes" array
    uint next;
};

uint packDepth(float normalizedDepth, uint bitArray) {
    //return (min(2147483647u, uint(normalizedDepth * 2147483648.0)) << 1u) | isBoundaryBit;
    return (min(1073741823, uint(normalizedDepth * 1073741824.0)) << 2u) | bitArray;
}

float unpackDepth(uint packedDepth) {
    //return float((packedDepth >> 1u) & 2147483647u) / 2147483647.0;
    return float((packedDepth >> 2u) & 1073741823u) / 1073741823.0;
}

layout(binding = 0) uniform UniformDataBuffer {
    // Inverse of (projectionMatrix * viewMatrix).
    mat4 inverseViewProjectionMatrix;

    // Number of fragments we can store in total.
    uint linkedListSize;
    // Size of the viewport in x direction (in pixels).
    int viewportW;
    // Camera near/far plane distance.
    float zNear, zFar;

    // Camera front vector.
    vec3 cameraFront;
    // Volume attenuation.
    float attenuationCoefficient;

    // Viewport size in x/y direction.
    uvec2 viewportSize;

    // Size of the viewport in x direction (in pixels) without padding.
    int viewportLinearW;
    int paddingUniform;
};

// Fragment-and-link buffer (linked list). Stores "nodesPerPixel" number of fragments.
layout(std430, binding = 1) buffer FragmentBuffer {
    LinkedListFragmentNode fragmentBuffer[];
};

// Start-offset buffer (mapping pixels to first pixel in the buffer) of size viewportSize.x * viewportSize.y.
layout(std430, binding = 2) coherent buffer StartOffsetBuffer {
    uint startOffset[];
};

// Position of the first free fragment node in the linked list.
layout(std430, binding = 3) buffer FragCounterBuffer {
    uint fragCounter;
};

#include "TiledAddress.glsl"

#ifdef SHOW_DEPTH_COMPLEXITY
// Stores the number of fragments using atomic operations.
layout(binding = 11) coherent buffer DepthComplexityCounterBuffer {
    uint depthComplexityCounterBuffer[];
};

uint addrGenLinear(uvec2 addr2D) {
    return addr2D.x + viewportLinearW * addr2D.y;
}
#endif
