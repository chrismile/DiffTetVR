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

// Previously atomically increased linear append index.
layout(binding = 0, std430) uniform TriangleCounterBuffer {
    uint globalTriangleCounter;
};

// Previously generated triangle data.
layout(binding = 1, std430) readonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};

// Atomically increased linear append index.
layout(binding = 2, std430) coherent buffer TetCounterBuffer {
    uint globalTetCounter;
};
layout(binding = 3, std430) writeonly buffer TetTriangleOffsetBuffer {
    uint tetOffsets[];
};

void main() {
    const uint triIdx = gl_GlobalInvocationID.x;
    if (triIdx >= globalTriangleCounter) {
        return;
    }

    uint lastIdx = 0xFFFFFFFFu;
    if (triIdx != 0) {
        lastIdx = triangleTetIndices[triIdx - 1u];
    }
    uint currIdx = triangleTetIndices[triIdx];
    if (currIdx != lastIdx) {
        uint offset = atomicAdd(globalTetCounter, 1u);
        tetOffsets[offset] = triIdx;
    }
}
