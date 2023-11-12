/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
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

-- Compute.Image

#version 450 core

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y) in;

layout(binding = 0) uniform UniformBuffer {
    uint imageWidth, imageHeight, batchSize;
};

layout(binding = 1, rgba32f) uniform readonly image2D colorImageGT[BATCH_SIZE];
layout(binding = 2, rgba32f) uniform readonly image2D colorImageOpt[BATCH_SIZE];
layout(binding = 3, rgba32f) uniform writeonly image2D adjointColors[BATCH_SIZE];

void main() {
    const uvec2 imageIdx = gl_GlobalInvocationID.xy;
    if (imageIdx.x >= imageWidth || imageIdx.y >= imageHeight) {
        return;
    }
    const float invN = 1.0 / float(imageWidth * imageHeight * batchSize);
    for (uint batchIdx = 0; batchIdx < BATCH_SIZE; batchIdx++) {
        vec4 colorDiff = imageLoad(colorImageOpt[batchIdx], imageIdx) - imageLoad(colorImageOpt[batchIdx], imageIdx);

        vec4 adjointColor;
#if defined(L1_LOSS)
        adjointColor = invN * (2.0 * vec4(greaterThanEqual(colorDiff, vec4(0.0))) - vec4(1.0));
#elif defined(L2_LOSS)
        adjointColor = (invN * 2.0) * colorDiff;
#endif

        imageStore(adjointColors[batchIdx], imageIdx, adjointColor);
    }
}


-- Compute.Buffer

#version 450 core

layout(local_size_x = BLOCK_SIZE) in;

layout(binding = 0) uniform UniformBuffer {
    uint imageWidth, imageHeight, batchSize;
};

layout(binding = 1, std430) readonly buffer ColorImageGTBuffer {
    vec4 colorImageGT[];
};
layout(binding = 2, std430) readonly buffer FinalImageOptBuffer {
    vec4 colorImageOpt[];
};
layout(binding = 3, std430) writeonly buffer AdjointColorsBuffer {
    vec4 adjointColors[];
};

void main() {
    const uint workSizeLinear = imageWidth * imageHeight * batchSize;
    const float invN = 1.0 / float(workSizeLinear);
    const uint workStep = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
    for (uint workIdx = gl_GlobalInvocationID.x; workIdx < workSizeLinear; workIdx += workStep) {
        vec4 colorDiff = colorImageOpt[workIdx] - colorImageGT[workIdx];
        vec4 adjointColor;
#if defined(L1_LOSS)
        adjointColor = invN * (2.0 * vec4(greaterThanEqual(colorDiff, vec4(0.0))) - vec4(1.0));
#elif defined(L2_LOSS)
        adjointColor = (invN * 2.0) * colorDiff;
#endif
        adjointColors[workIdx] = adjointColor;
    }
}
