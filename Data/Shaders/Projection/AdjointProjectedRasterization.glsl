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

layout(binding = 0, std430) readonly buffer SortedTetIndexBuffer {
    uint sortedTriangleIndices[];
};
layout(binding = 1, std430) readonly buffer VertexPositionBuffer {
    uint vertexPositions[];
};
layout(binding = 2, std430) readonly buffer VertexColorBuffer {
    uint vertexColors[];
};
layout(binding = 3, std430) readonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[]; //< maps triangle -> tet index.
};

//flat out

void main() {

}


-- Fragment

#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_ARB_fragment_shader_interlock : require

layout(early_fragment_tests, pixel_interlock_ordered) in;

#include "TiledAddress.glsl"

void main() {
    uint x = uint(gl_FragCoord.x);
    uint y = uint(gl_FragCoord.y);
    uint pixelIndex = addrGen(uvec2(x,y));

    beginInvocationInterlockARB();
    // TODO
    endInvocationInterlockARB();
}
