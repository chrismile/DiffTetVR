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

layout(location = 0) in vec3 vertexPosition;

void main() {
    gl_Position = vec4(vertexPosition, 1.0);
}


-- Fragment

#version 450 core

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_terminate_invocation : require
//#extension GL_EXT_debug_printf : enable

#include "LinkedListHeader.glsl"
#include "DepthHelper.glsl"

uint colorList[MAX_NUM_FRAGS];
float depthList[MAX_NUM_FRAGS];

layout(binding = 4, scalar) readonly buffer TriangleIndicesBuffer {
    uint triangleIndices[];
};
layout(binding = 5, scalar) readonly buffer VertexPositionBuffer {
    vec3 vertexPositions[];
};

#ifndef SHOW_TET_QUALITY
#ifdef PER_VERTEX_COLORS
layout(binding = 6, scalar) readonly buffer VertexColorBuffer {
    vec4 vertexColors[];
};
#else
layout(binding = 6, scalar) readonly buffer CellColorBuffer {
    vec4 cellColors[];
};
#endif
#endif

#if defined(SHOW_TET_QUALITY) || !defined(PER_VERTEX_COLORS)
#define INVALID_TET 0xFFFFFFFFu
layout(binding = 7, scalar) readonly buffer FaceToTetMapBuffer {
    uvec2 faceToTetMap[];
};
#endif

#ifndef PER_VERTEX_COLORS
/*
 * Input: The indices of the (up to two) tets incident to two tet mesh faces.
 * Output: The tet index shared by the two faces, or INVALID_TET if no face is shared.
 */
uint tetIdUnion(uvec2 ids0, uvec2 ids1) {
    uvec4 unionVals;
    if ((ids0.x == ids1.x || ids0.x == ids1.y) && ids0.x != INVALID_TET) {
        return ids0.x;
    } else if ((ids0.y == ids1.x || ids0.y == ids1.y) && ids0.y != INVALID_TET) {
        return ids0.y;
    } else {
        return INVALID_TET;
    }
}
#endif

#ifdef SHOW_TET_QUALITY
layout(binding = 8, scalar) readonly buffer TetQualityBuffer {
    float tetQualityArray[];
};
layout (binding = 9) uniform MinMaxUniformBuffer {
    float minAttributeValue;
    float maxAttributeValue;
};
layout(binding = 10) uniform sampler1D transferFunctionTexture;
vec4 transferFunction(float attr) {
    // Transfer to range [0, 1].
    float posFloat = clamp((attr - minAttributeValue) / (maxAttributeValue - minAttributeValue), 0.0, 1.0);
    // Look up the color value.
    return texture(transferFunctionTexture, posFloat);
}
#endif

#ifdef USE_TERMINATION_INDEX
#ifdef BACKWARD_PASS
layout(binding = 12, r32ui) uniform readonly uimage2D terminationIndexImage;
#else
layout(binding = 12, r32ui) uniform writeonly uimage2D terminationIndexImage;
#endif
#endif


#include "LinkedListSort.glsl"

#ifdef USE_QUICKSORT
#include "LinkedListQuicksort.glsl"
#endif

layout(location = 0) out vec4 fragColor;

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);
    uint pixelIndex = addrGen(uvec2(x,y));

    // Get start offset from array
    uint fragOffset = startOffset[pixelIndex];

#ifdef INITIALIZE_ARRAY_POW2
    for (int i = 0; i < MAX_NUM_FRAGS; i++) {
        colorList[i] = 0;
        depthList[i] = 0;
    }
#endif

    // Collect all fragments for this pixel
    int numFrags = 0;
    LinkedListFragmentNode fragment;
    for (int i = 0; i < MAX_NUM_FRAGS; i++) {
        if (fragOffset == -1) {
            // End of list reached
            break;
        }

#if defined(FRAGMENT_BUFFER_REFERENCE_ARRAY)
        FragmentBufferEntry fbe = FragmentBufferEntry(
                fagmentBuffers[fragOffset / NUM_FRAGS_PER_BUFFER] + 12u * uint64_t(fragOffset % NUM_FRAGS_PER_BUFFER));
        fragment.color = fbe.color;
        fragment.depth = fbe.depth;
        fragment.next = fbe.next;
#elif defined(FRAGMENT_BUFFER_ARRAY)
        fragment = fragmentBuffers[nonuniformEXT(fragOffset / NUM_FRAGS_PER_BUFFER)].fragmentBuffer[fragOffset % NUM_FRAGS_PER_BUFFER];
#else
        fragment = fragmentBuffer[fragOffset];
#endif
        fragOffset = fragment.next;

        colorList[i] = fragment.color;
        depthList[i] = fragment.depth;

        numFrags++;
    }

    if (numFrags == 0) {
        // glslang changed its behavior for discard in version 15.4: https://github.com/KhronosGroup/glslang/pull/3954
        // getNextFragment is called for fragment 0, so we explicitly need a terminate, not a demote.
        terminateInvocation;
    }

    fragColor = sortingAlgorithm(numFrags);
}
