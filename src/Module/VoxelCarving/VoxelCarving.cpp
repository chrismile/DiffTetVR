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

#include <glm/ext/matrix_clip_space.hpp>

#include <Math/Geometry/MatrixUtil.hpp>

#include "VoxelCarving.hpp"

void CameraSettings::setIntrinsics(uint32_t _imgWidth, uint32_t _imgHeight, float _fovy, float _near, float _far) {
    if (imgWidth == _imgWidth && imgHeight == _imgHeight && fovy == _fovy && near == _near && far == _far) {
        return;
    }
    imgWidth = _imgWidth;
    imgHeight = _imgHeight;
    fovy = _fovy;
    near = _near;
    far = _far;
    auto aspect = float(imgWidth) / float(imgWidth);
    projectionMatrix = glm::perspectiveRH_ZO(fovy, aspect, near, far);
}

void CameraSettings::setViewMatrix(const glm::mat4& _viewMatrix) {
    viewMatrix = _viewMatrix;
}

VoxelCarving::VoxelCarving(const sgl::AABB3& gridBoundingBox, const glm::uvec3& gridResolution)
        : gridBoundingBox(gridBoundingBox), gridResolution(gridResolution) {
    worldSpaceToVoxelGridSpaceMatrix =
            sgl::matrixScaling(glm::vec3(gridResolution) / (gridBoundingBox.max - gridBoundingBox.min))
            * sgl::matrixTranslation(-gridBoundingBox.min);
    voxelGridSpaceToWorldSpaceMatrix = glm::inverse(worldSpaceToVoxelGridSpaceMatrix);
}
