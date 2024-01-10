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

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>

#include "TetRegularizerPass.hpp"

TetRegularizerPass::TetRegularizerPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void TetRegularizerPass::setBuffers(
        const sgl::vk::BufferPtr& _cellIndicesBuffer,
        const sgl::vk::BufferPtr& _vertexPositionBuffer,
        const sgl::vk::BufferPtr& _vertexPositionGradientBuffer) {
    if (cellIndicesBuffer != _cellIndicesBuffer) {
        cellIndicesBuffer = _cellIndicesBuffer;
        if (computeData) {
            computeData->setStaticBuffer(cellIndicesBuffer, "CellIndicesBuffer");
        }
    }
    if (vertexPositionBuffer != _vertexPositionBuffer) {
        vertexPositionBuffer = _vertexPositionBuffer;
        if (computeData) {
            computeData->setStaticBuffer(vertexPositionBuffer, "VertexPositionBuffer");
        }
    }
    if (vertexPositionGradientBuffer != _vertexPositionGradientBuffer) {
        vertexPositionGradientBuffer = _vertexPositionGradientBuffer;
        if (computeData) {
            computeData->setStaticBuffer(vertexPositionGradientBuffer, "VertexPositionGradientBuffer");
        }
    }
}

void TetRegularizerPass::setSettings(float lambda, float beta) {
    if (uniformData.lambda != lambda || uniformData.beta != beta) {
        uniformData.lambda = lambda;
        uniformData.beta = beta;
        uniformBuffer->updateData(
                sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    }
}

void TetRegularizerPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "TetRegularizer.Compute" }, preprocessorDefines);
}

void TetRegularizerPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "RegularizerSettingsBuffer");
    computeData->setStaticBuffer(cellIndicesBuffer, "CellIndicesBuffer");
    computeData->setStaticBuffer(vertexPositionBuffer, "VertexPositionBuffer");
    computeData->setStaticBuffer(vertexPositionGradientBuffer, "VertexPositionGradientBuffer");
}

void TetRegularizerPass::_render() {
    auto numTets = uint32_t(cellIndicesBuffer->getSizeInBytes() / sizeof(glm::uvec4));
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, numTets);
    renderer->dispatch(computeData, sgl::uiceil(numTets, computeBlockSize), 1, 1);
}
