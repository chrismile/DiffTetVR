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

#ifndef DIFFTETVR_RASTERCOMMON_HPP
#define DIFFTETVR_RASTERCOMMON_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "TetMeshVolumeRenderer.hpp"

#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
typedef struct radix_sort_vk_target radix_sort_vk_target_t;
typedef struct radix_sort_vk radix_sort_vk_t;
#endif

/*class GenerateTrianglesPass;
class InitializeIndirectCommandBufferPass;
class ComputeTrianglesDepthPass;
class ProjectedRasterPass;
class AdjointProjectedRasterPass;
class IntersectRasterPass;
class AdjointIntersectRasterPass;*/

enum class SortingAlgorithm {
    FUCHSIA_RADIX_SORT, CPU_STD_SORT
};
const char* const SORTING_ALGORITHM_NAMES[] = {
        "Radix Sort (Fuchsia)", "CPU std::sort"
};

/*class GenerateTrianglesPass : public sgl::vk::ComputePass {
public:
    explicit GenerateTrianglesPass(TetMeshRendererIntersection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "GenerateTriangles.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        const auto& tetMesh = volumeRenderer->getTetMesh();
        computeData->setStaticBuffer(tetMesh->getTriangleIndexBuffer(), "TriangleCounterBuffer");
        computeData->setStaticBuffer(volumeRenderer->getUniformDataBuffer(), "UniformDataBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleCounterBuffer(), "TriangleCounterBuffer");
        computeData->setStaticBuffer(tetMesh->getCellIndicesBuffer(), "TetIndexBuffer");
        computeData->setStaticBuffer(tetMesh->getVertexPositionBuffer(), "TetVertexPositionBuffer");
        computeData->setStaticBuffer(tetMesh->getVertexColorBuffer(), "TetVertexColorBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleVertexPositionBuffer(), "TriangleVertexPositionBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleVertexColorBuffer(), "TriangleVertexColorBuffer");
        volumeRenderer->setRenderDataBindings(computeData);
    }
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {
    }
    void _render() override {
        const auto& tetMesh = volumeRenderer->getTetMesh();
        const sgl::vk::BufferPtr& triangleCounterBuffer = volumeRenderer->getTriangleCounterBuffer();
        triangleCounterBuffer->fill(0, renderer->getVkCommandBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                triangleCounterBuffer);
        renderer->dispatch(computeData, sgl::uiceil(uint32_t(tetMesh->getNumCells()), BLOCK_SIZE), 1, 1);
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                volumeRenderer->getTriangleVertexPositionBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                volumeRenderer->getTriangleVertexColorBuffer());
    }

private:
    TetMeshRendererIntersection* volumeRenderer;
    const uint32_t BLOCK_SIZE = 256;
};

class InitializeIndirectCommandBufferPass : public sgl::vk::ComputePass {
public:
    explicit InitializeIndirectCommandBufferPass(TetMeshRendererIntersection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "InitializeIndirectCommandBuffer.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        computeData->setStaticBuffer(volumeRenderer->getTriangleCounterBuffer(), "TriangleCounterBuffer");
        computeData->setStaticBuffer(volumeRenderer->getDrawIndirectBuffer(), "DrawIndirectCommandBuffer");
        computeData->setStaticBuffer(volumeRenderer->getDispatchIndirectBuffer(), "DispatchIndirectCommandBuffer");
        volumeRenderer->setRenderDataBindings(computeData);
    }
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {
    }
    void _render() override {
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                volumeRenderer->getTriangleCounterBuffer());
        renderer->dispatch(computeData, 1, 1, 1);
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                volumeRenderer->getDrawIndirectBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                volumeRenderer->getDispatchIndirectBuffer());
    }

private:
    TetMeshRendererIntersection* volumeRenderer;
    const uint32_t BLOCK_SIZE = 256; // Value of ComputeTrianglesDepthPass; only used for computing #workgroups.
};

class ComputeTrianglesDepthPass : public sgl::vk::ComputePass {
public:
    explicit ComputeTrianglesDepthPass(TetMeshRendererIntersection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "ComputeTriangleDepths.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        computeData->setStaticBuffer(volumeRenderer->getTriangleCounterBuffer(), "TriangleCounterBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleVertexPositionBuffer(), "TriangleVertexPositionBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleKeyValueBuffer(), "TriangleKeyValueBuffer");
        volumeRenderer->setRenderDataBindings(computeData);
    }
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {
    }
    void _render() override {
        renderer->dispatchIndirect(computeData, volumeRenderer->getDispatchIndirectBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                volumeRenderer->getTriangleKeyValueBuffer());
    }

private:
    TetMeshRendererIntersection* volumeRenderer;
    const uint32_t BLOCK_SIZE = 256;
};

class ProjectedRasterPass : public sgl::vk::RasterPass {
public:
    explicit ProjectedRasterPass(TetMeshRendererProjection* volumeRenderer)
            : RasterPass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }
    void recreateSwapchain(uint32_t width, uint32_t height) override {
        framebuffer = std::make_shared<sgl::vk::Framebuffer>(device, width, height);
        volumeRenderer->setFramebufferAttachments(framebuffer, VK_ATTACHMENT_LOAD_OP_CLEAR);
        framebufferDirty = true;
        dataDirty = true;
    }
    void setAttachmentClearColor(const glm::vec4& color) {
        if (framebuffer) {
            bool dataDirtyOld = dataDirty;
            recreateSwapchain(framebuffer->getWidth(), framebuffer->getHeight());
            dataDirty = dataDirtyOld;
        } else {
            setDataDirty();
        }
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "ProjectedRasterization.Vertex", "ProjectedRasterization.Fragment" }, preprocessorDefines);
    }
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
        rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
        rasterData->setStaticBuffer(volumeRenderer->getSortedTriangleKeyValueBuffer(), "TriangleKeyValueBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexPositionBuffer(), "TriangleVertexPositionBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexColorBuffer(), "TriangleVertexColorBuffer");
        rasterData->setIndirectDrawBuffer(volumeRenderer->getDrawIndirectBuffer(), sizeof(VkDrawIndirectCommand));
        rasterData->setIndirectDrawCount(1);
        volumeRenderer->setRenderDataBindings(rasterData);
    }
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
        pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
        pipelineInfo.setIsFrontFaceCcw(true);
        pipelineInfo.setColorWriteEnabled(true);
        pipelineInfo.setDepthWriteEnabled(false);
        pipelineInfo.setDepthTestEnabled(false);
        pipelineInfo.setBlendMode(sgl::vk::BlendMode::BACK_TO_FRONT_PREMUL_ALPHA);
    }

private:
    TetMeshRendererProjection* volumeRenderer;
};

class IntersectRasterPass : public sgl::vk::RasterPass {
public:
    explicit IntersectRasterPass(TetMeshRendererIntersection* volumeRenderer)
            : RasterPass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }
    void recreateSwapchain(uint32_t width, uint32_t height) override {
        framebuffer = std::make_shared<sgl::vk::Framebuffer>(device, width, height);
        volumeRenderer->setFramebufferAttachments(framebuffer, VK_ATTACHMENT_LOAD_OP_CLEAR);
        framebufferDirty = true;
        dataDirty = true;
    }
    void setAttachmentClearColor(const glm::vec4& color) {
        if (framebuffer) {
            bool dataDirtyOld = dataDirty;
            recreateSwapchain(framebuffer->getWidth(), framebuffer->getHeight());
            dataDirty = dataDirtyOld;
        } else {
            setDataDirty();
        }
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "IntersectRasterization.Vertex", "IntersectRasterization.Fragment" }, preprocessorDefines);
    }
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
        rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
        rasterData->setStaticBuffer(volumeRenderer->getSortedTriangleKeyValueBuffer(), "TriangleKeyValueBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexPositionBuffer(), "TriangleVertexPositionBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexColorBuffer(), "TriangleVertexColorBuffer");
        rasterData->setIndirectDrawBuffer(volumeRenderer->getDrawIndirectBuffer(), sizeof(VkDrawIndirectCommand));
        rasterData->setIndirectDrawCount(1);
        volumeRenderer->setRenderDataBindings(rasterData);
    }
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
        pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
        pipelineInfo.setIsFrontFaceCcw(true);
        pipelineInfo.setColorWriteEnabled(true);
        pipelineInfo.setDepthWriteEnabled(false);
        pipelineInfo.setDepthTestEnabled(false);
        pipelineInfo.setBlendMode(sgl::vk::BlendMode::BACK_TO_FRONT_PREMUL_ALPHA);
    }

private:
    TetMeshRendererIntersection* volumeRenderer;
};*/

#endif //DIFFTETVR_RASTERCOMMON_HPP
