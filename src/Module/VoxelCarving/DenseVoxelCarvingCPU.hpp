/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2025, Christoph Neuhauser
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

#ifndef DIFFTETVR_DENSEVOXELCARVINGCPU_HPP
#define DIFFTETVR_DENSEVOXELCARVINGCPU_HPP

#include "VoxelCarving.hpp"

class DenseVoxelCarvingCPU : public VoxelCarving {
public:
    DenseVoxelCarvingCPU(const sgl::AABB3& gridBoundingBox, const glm::uvec3& gridResolution);
    ~DenseVoxelCarvingCPU() override;
    void processNextFrame(const torch::Tensor& inputImage, const CameraSettings& cameraSettings) override;
    sgl::AABB3 computeNonEmptyBoundingBox() override;

private:
    /// Alpha channel threshold used for deciding what is foreground and background.
    const float BG_THRESH = 0.1f;

    /// Projects the voxel location into the input image and retrieves the alpha channel value.
    uint8_t sampleImage(
            const at::TensorAccessor<float, 3>& inputImageAccessor, const glm::mat4& gridSpaceToClipSpaceMatrix,
            const glm::vec4& voxelPosition);

    /// Export the volume for debugging purposes (data can be opened with https://github.com/chrismile/Correrender).
    bool exportVolume(const std::string& filePath);

    uint8_t* voxelGrid = nullptr;
};

#endif //DIFFTETVR_DENSEVOXELCARVINGCPU_HPP
