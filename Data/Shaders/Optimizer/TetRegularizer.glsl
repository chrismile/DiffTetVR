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

-- Compute

#version 450 core

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require

/*
 * Regularizer penalizing strongly deformed tetrahedral elements (esp. those with negative volume).
 */

layout(local_size_x = BLOCK_SIZE) in;

layout(push_constant) uniform PushConstants {
    uint numTets;
};

layout(binding = 0) uniform RegularizerSettingsBuffer {
    float lambda; ///< Regularizer strength.
    float beta; ///< Softplus parameter.
};

layout(binding = 1, std430) readonly buffer CellIndicesBuffer {
    uvec4 cellIndices[];
};

layout(binding = 2, scalar) readonly buffer VertexPositionBuffer {
    vec3 vertexPositions[];
};

layout(binding = 3, scalar) coherent buffer VertexPositionGradientBuffer {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float vertexPositionGradients[]; // stride: vec3
#else
    uint vertexPositionGradients[]; // stride: vec3
#endif
};

void atomicAddGradPos(uint idx, vec3 value) {
    const uint stride = idx * 3;
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    atomicAdd(vertexPositionGradients[stride], value.x);
    atomicAdd(vertexPositionGradients[stride + 1], value.y);
    atomicAdd(vertexPositionGradients[stride + 2], value.z);
#else
    [[unroll]] for (uint i = 0; i < 3; i++) {
        uint oldValue = vertexPositionGradients[stride + i];
        uint expectedValue, newValue;
        do {
            expectedValue = oldValue;
            newValue = floatBitsToUint(uintBitsToFloat(oldValue) + value[i]);
            oldValue = atomicCompSwap(vertexPositionGradients[stride + i], expectedValue, newValue);
        } while (oldValue != expectedValue);
    }
#endif
}

float computeVolume(vec3 p0, vec3 p1, vec3 p2, vec3 p3) {
    return -dot(cross(p1 - p0, p2 - p0), p3 - p0) / 6.0;
}

float computeEdgeLengthRms(vec3 p0, vec3 p1, vec3 p2, vec3 p3) {
    float l01 = length(p0 - p1);
    float l02 = length(p0 - p2);
    float l03 = length(p0 - p3);
    float l12 = length(p1 - p2);
    float l13 = length(p1 - p3);
    float l23 = length(p2 - p3);
    float l_rms = 0.0f;
    l_rms += l01 * l01;
    l_rms += l02 * l02;
    l_rms += l03 * l03;
    l_rms += l12 * l12;
    l_rms += l13 * l13;
    l_rms += l23 * l23;
    l_rms = sqrt(l_rms / 6.0);
    return l_rms;
}

void main() {
    uint tetIdx = gl_GlobalInvocationID.x;
    if (tetIdx >= numTets) {
        return;
    }

    const uvec4 vertexIndices = cellIndices[tetIdx];
    const vec3 p0 = vertexPositions[vertexIndices.x];
    const vec3 p1 = vertexPositions[vertexIndices.y];
    const vec3 p2 = vertexPositions[vertexIndices.z];
    const vec3 p3 = vertexPositions[vertexIndices.w];

    /**
     * Use quality metric by Parthasarathy et al. (1991) described in:
     * https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf
     */
    const float V = computeVolume(p0, p1, p2, p3);
    const float l_rms = computeEdgeLengthRms(p0, p1, p2, p3);
    const float quality = 6.0 * sqrt(2.0) * V / (l_rms * l_rms * l_rms);
    /**
     * Softplus(q) = 1/beta * log(1 + exp(beta * q)
     * d Softplus(q) / dq = 1 / (1 + exp(-beta * q))
     * Normally "-beta" would be used below, but we want a mirrored softplus (negative/low quality gets penalized).
     */
    const float dSoftplus_dQuality = -lambda / (1.0 + exp(beta * quality));

    // Backpropagate the loss to the individual vertex positions.
    float T0 = 62.353829072479584*((p0.x - p3.x)*((p0.y - p1.y)*(p0.z - p2.z) - (p0.y - p2.y)*(p0.z - p1.z)) + (p0.y - p3.y)*(-(p0.x - p1.x)*(p0.z - p2.z) + (p0.x - p2.x)*(p0.z - p1.z)) + (p0.z - p3.z)*((p0.x - p1.x)*(p0.y - p2.y) - (p0.x - p2.x)*(p0.y - p1.y)));
    float T1 = pow(p0.x - p1.x, 2) + pow(p0.x - p2.x, 2) + pow(p0.x - p3.x, 2) + pow(p0.y - p1.y, 2) + pow(p0.y - p2.y, 2) + pow(p0.y - p3.y, 2) + pow(p0.z - p1.z, 2) + pow(p0.z - p2.z, 2) + pow(p0.z - p3.z, 2) + pow(p1.x - p2.x, 2) + pow(p1.x - p3.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.y - p3.y, 2) + pow(p1.z - p2.z, 2) + pow(p1.z - p3.z, 2) + pow(p2.x - p3.x, 2) + pow(p2.y - p3.y, 2) + pow(p2.z - p3.z, 2);
    float T2 = pow(pow(p0.x - p1.x, 2) + pow(p0.x - p2.x, 2) + pow(p0.x - p3.x, 2) + pow(p0.y - p1.y, 2) + pow(p0.y - p2.y, 2) + pow(p0.y - p3.y, 2) + pow(p0.z - p1.z, 2) + pow(p0.z - p2.z, 2) + pow(p0.z - p3.z, 2) + pow(p1.x - p2.x, 2) + pow(p1.x - p3.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.y - p3.y, 2) + pow(p1.z - p2.z, 2) + pow(p1.z - p3.z, 2) + pow(p2.x - p3.x, 2) + pow(p2.y - p3.y, 2) + pow(p2.z - p3.z, 2), 5.0/2.0);
    float T3 = pow(pow(p0.x - p1.x, 2) + pow(p0.x - p2.x, 2) + pow(p0.x - p3.x, 2) + pow(p0.y - p1.y, 2) + pow(p0.y - p2.y, 2) + pow(p0.y - p3.y, 2) + pow(p0.z - p1.z, 2) + pow(p0.z - p2.z, 2) + pow(p0.z - p3.z, 2) + pow(p1.x - p2.x, 2) + pow(p1.x - p3.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.y - p3.y, 2) + pow(p1.z - p2.z, 2) + pow(p1.z - p3.z, 2) + pow(p2.x - p3.x, 2) + pow(p2.y - p3.y, 2) + pow(p2.z - p3.z, 2), 5.0/2.0);
    float T4 = pow(pow(p0.x - p1.x, 2) + pow(p0.x - p2.x, 2) + pow(p0.x - p3.x, 2) + pow(p0.y - p1.y, 2) + pow(p0.y - p2.y, 2) + pow(p0.y - p3.y, 2) + pow(p0.z - p1.z, 2) + pow(p0.z - p2.z, 2) + pow(p0.z - p3.z, 2) + pow(p1.x - p2.x, 2) + pow(p1.x - p3.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.y - p3.y, 2) + pow(p1.z - p2.z, 2) + pow(p1.z - p3.z, 2) + pow(p2.x - p3.x, 2) + pow(p2.y - p3.y, 2) + pow(p2.z - p3.z, 2), 5.0/2.0);
    T2 = max(T2, 1e-5);
    T3 = max(T3, 1e-5);
    T4 = max(T4, 1e-5);
    //if (T2 > -1e-5) T2 = -1e-5;
    //if (T2 < +1e-5) T2 = +1e-5;
    //if (T3 > -1e-5) T3 = -1e-5;
    //if (T3 < +1e-5) T3 = +1e-5;
    //if (T4 > -1e-5) T4 = -1e-5;
    //if (T4 < +1e-5) T4 = +1e-5;
    float dq_dp0x = (T0*(-3*p0.x + p1.x + p2.x + p3.x) + 20.784609690826528*((p0.y - p1.y)*(p0.z - p2.z) - (p0.y - p2.y)*(p0.z - p1.z) - (p0.y - p3.y)*(p1.z - p2.z) + (p0.z - p3.z)*(p1.y - p2.y))*(T1))/T2;
    float dq_dp0y = (T0*(-3*p0.y + p1.y + p2.y + p3.y) + 20.784609690826528*(-(p0.x - p1.x)*(p0.z - p2.z) + (p0.x - p2.x)*(p0.z - p1.z) + (p0.x - p3.x)*(p1.z - p2.z) - (p0.z - p3.z)*(p1.x - p2.x))*(T1))/T3;
    float dq_dp0z = (T0*(-3*p0.z + p1.z + p2.z + p3.z) + 20.784609690826528*((p0.x - p1.x)*(p0.y - p2.y) - (p0.x - p2.x)*(p0.y - p1.y) - (p0.x - p3.x)*(p1.y - p2.y) + (p0.y - p3.y)*(p1.x - p2.x))*(T1))/T2;
    float dq_dp1x = (20.784609690826528*(-(p0.y - p2.y)*(p0.z - p3.z) + (p0.y - p3.y)*(p0.z - p2.z))*(T1) + T0*(p0.x - 3*p1.x + p2.x + p3.x))/T4;
    float dq_dp1y = (20.784609690826528*((p0.x - p2.x)*(p0.z - p3.z) - (p0.x - p3.x)*(p0.z - p2.z))*(T1) + T0*(p0.y - 3*p1.y + p2.y + p3.y))/T2;
    float dq_dp1z = (20.784609690826528*(-(p0.x - p2.x)*(p0.y - p3.y) + (p0.x - p3.x)*(p0.y - p2.y))*(T1) + T0*(p0.z - 3*p1.z + p2.z + p3.z))/T3;
    float dq_dp2x = (20.784609690826528*((p0.y - p1.y)*(p0.z - p3.z) - (p0.y - p3.y)*(p0.z - p1.z))*(T1) + T0*(p0.x + p1.x - 3*p2.x + p3.x))/T2;
    float dq_dp2y = (20.784609690826528*(-(p0.x - p1.x)*(p0.z - p3.z) + (p0.x - p3.x)*(p0.z - p1.z))*(T1) + T0*(p0.y + p1.y - 3*p2.y + p3.y))/T4;
    float dq_dp2z = (20.784609690826528*((p0.x - p1.x)*(p0.y - p3.y) - (p0.x - p3.x)*(p0.y - p1.y))*(T1) + T0*(p0.z + p1.z - 3*p2.z + p3.z))/T2;
    float dq_dp3x = (20.784609690826528*(-(p0.y - p1.y)*(p0.z - p2.z) + (p0.y - p2.y)*(p0.z - p1.z))*(T1) + T0*(p0.x + p1.x + p2.x - 3*p3.x))/T3;
    float dq_dp3y = (20.784609690826528*((p0.x - p1.x)*(p0.z - p2.z) - (p0.x - p2.x)*(p0.z - p1.z))*(T1) + T0*(p0.y + p1.y + p2.y - 3*p3.y))/T2;
    float dq_dp3z = (20.784609690826528*(-(p0.x - p1.x)*(p0.y - p2.y) + (p0.x - p2.x)*(p0.y - p1.y))*(T1) + T0*(p0.z + p1.z + p2.z - 3*p3.z))/T4;

    atomicAddGradPos(vertexIndices.x, dSoftplus_dQuality * vec3(dq_dp0x, dq_dp0y, dq_dp0z));
    atomicAddGradPos(vertexIndices.y, dSoftplus_dQuality * vec3(dq_dp1x, dq_dp1y, dq_dp1z));
    atomicAddGradPos(vertexIndices.z, dSoftplus_dQuality * vec3(dq_dp2x, dq_dp2y, dq_dp2z));
    atomicAddGradPos(vertexIndices.w, dSoftplus_dQuality * vec3(dq_dp3x, dq_dp3y, dq_dp3z));
}
