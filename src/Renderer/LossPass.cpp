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

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>

#include "LossPass.hpp"

LossPass::LossPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void LossPass::setImageViews(
        const sgl::vk::ImageViewPtr& _colorImageGT,
        const sgl::vk::ImageViewPtr& _colorImageOpt,
        const sgl::vk::ImageViewPtr& _adjointColorsImageView,
        const sgl::vk::BufferPtr& _startOffsetBufferGT,
        const sgl::vk::BufferPtr& _startOffsetBufferOpt) {
    if (colorImageGT != _colorImageGT) {
        colorImageGT = _colorImageGT;
        if (computeData) {
            computeData->setStaticImageView(colorImageGT, "colorImageGT");
        }
    }
    if (colorImageOpt != _colorImageOpt) {
        colorImageOpt = _colorImageOpt;
        if (computeData) {
            computeData->setStaticImageView(colorImageOpt, "colorImageOpt");
        }
    }
    if (adjointColorsImageView != _adjointColorsImageView) {
        adjointColorsImageView = _adjointColorsImageView;
        if (computeData) {
            computeData->setStaticImageView(adjointColorsImageView, "AdjointColorsImageView");
        }
    }
    if (startOffsetBufferGT != _startOffsetBufferGT) {
        startOffsetBufferGT = _startOffsetBufferGT;
        if (computeData) {
            computeData->setStaticBuffer(startOffsetBufferGT, "StartOffsetBufferGT");
        }
    }
    if (startOffsetBufferOpt != _startOffsetBufferOpt) {
        startOffsetBufferOpt = _startOffsetBufferOpt;
        if (computeData) {
            computeData->setStaticBuffer(startOffsetBufferOpt, "StartOffsetBufferOpt");
        }
    }
}

void LossPass::setSettings(
        LossType _lossType, uint32_t imageWidth, uint32_t imageHeight,
        uint32_t paddedViewportWidth, uint32_t paddedViewportHeight,
        std::map<std::string, std::string> _preprocessorDefinesRenderer) {
    if (lossType != _lossType) {
        lossType = _lossType;
        setShaderDirty();
    }
    if (imageWidth != uniformData.imageWidth || imageHeight != uniformData.imageHeight) {
        uniformData.imageWidth = imageWidth;
        uniformData.imageHeight = imageHeight;
        isUniformBufferDirty = true;
    }
    if (paddedViewportWidth != uniformData.paddedViewportWidth
            || paddedViewportHeight != uniformData.paddedViewportHeight) {
        uniformData.paddedViewportWidth = paddedViewportWidth;
        uniformData.paddedViewportHeight = paddedViewportHeight;
        isUniformBufferDirty = true;
    }
    if (preprocessorDefinesRenderer != _preprocessorDefinesRenderer) {
        preprocessorDefinesRenderer = _preprocessorDefinesRenderer;
        setShaderDirty();
    }
}

void LossPass::updateUniformBuffer() {
    if (isUniformBufferDirty) {
        isUniformBufferDirty = false;
        uniformBuffer->updateData(
                sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void LossPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines = preprocessorDefinesRenderer;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(blockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(blockSizeY)));
    if (lossType == LossType::L1) {
        preprocessorDefines.insert(std::make_pair("L1_LOSS", ""));
    } else if (lossType == LossType::L2) {
        preprocessorDefines.insert(std::make_pair("L2_LOSS", ""));
    }
    if (renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat32AtomicAdd) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_BUFFER_FLOAT_ATOMIC_ADD", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "Loss.Compute.Image" }, preprocessorDefines);
}

void LossPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticImageView(colorImageGT, "colorImageGT");
    computeData->setStaticImageView(colorImageOpt, "colorImageOpt");
    computeData->setStaticImageView(adjointColorsImageView, "adjointColors");
    computeData->setStaticBuffer(startOffsetBufferGT, "StartOffsetBufferGT");
    computeData->setStaticBuffer(startOffsetBufferOpt, "StartOffsetBufferOpt");
}

void LossPass::_render() {
    renderer->dispatch(
            computeData,
            sgl::uiceil(uniformData.imageWidth, blockSizeX),
            sgl::uiceil(uniformData.imageHeight, blockSizeY),
            1);
}
