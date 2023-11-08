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

-- Vertex

#version 450 core

#extension GL_EXT_scalar_block_layout : require

layout(binding = 4, scalar) readonly buffer TriangleIndicesBuffer {
    uint triangleIndices[];
};
layout(binding = 5, scalar) readonly buffer VertexPositionBuffer {
    vec3 vertexPositions[];
};
layout(binding = 6, scalar) readonly buffer VertexColorBuffer {
    vec4 vertexColors[];
};
layout(binding = 7, scalar) readonly buffer FaceBoundaryBitBuffer {
    uint faceBoundaryBitArray[];
};

layout(location = 0) out vec4 fragmentColor;
layout(location = 1) flat out uint isBoundaryBit;

void main() {
    const uint faceIndex = gl_VertexIndex / 3u;
    const uint vertexIndex = triangleIndices[gl_VertexIndex];
    vec3 vertexPosition = vertexPositions[vertexIndex];
    fragmentColor = vertexColors[vertexIndex];
    isBoundaryBit = faceBoundaryBitArray[faceIndex];
    gl_Position = mvpMatrix * vec4(vertexPosition, 1.0);
}


-- Fragment

#version 450 core

#include "LinkedListHeader.glsl"

layout(location = 0) in vec4 fragmentColor;
layout(location = 1) flat in uint isBoundaryBit;

layout(location = 0) out vec4 fragColor;

void main() {
    if (fragmentColor.a < 0.001) {
#ifndef GATHER_NO_DISCARD
        discard;
#else
        return;
#endif
    }

    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);
    uint pixelIndex = addrGen(uvec2(x,y));

    LinkedListFragmentNode frag;
    frag.color = packUnorm4x8(fragmentColor);
    frag.depth = packDepth(gl_FragCoord.z, isBoundaryBit);
    frag.next = -1;

    uint insertIndex = atomicAdd(fragCounter, 1u);

    if (insertIndex < linkedListSize) {
        // Insert the fragment into the linked list
        frag.next = atomicExchange(startOffset[pixelIndex], insertIndex);
        fragmentBuffer[insertIndex] = frag;
    }
}
