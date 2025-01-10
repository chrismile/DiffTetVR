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

#include <iostream>

#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/Passes/BlitRenderPass.hpp>
#ifndef DISABLE_IMGUI
#include <ImGui/Widgets/PropertyEditor.hpp>
#endif

#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
#include <radix_sort/radix_sort_vk.h>
#endif

#include "Tet/TetMesh.hpp"
#include "TetMeshRendererProjection.hpp"

class GenerateTrianglesProjPass : public sgl::vk::ComputePass {
public:
    explicit GenerateTrianglesProjPass(TetMeshRendererProjection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
        preprocessorDefines.insert(std::make_pair("PI_SQRT", std::to_string(std::sqrt(sgl::PI))));
        preprocessorDefines.insert(std::make_pair("INV_PI_SQRT", std::to_string(1.0f / std::sqrt(sgl::PI))));
        const auto& tetMesh = volumeRenderer->getTetMesh();
        if (tetMesh->getUseGradients()) {
            preprocessorDefines.insert(std::make_pair("SUPPORT_ADJOINT", ""));
        }
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "GenerateTrianglesVTK.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        const auto& tetMesh = volumeRenderer->getTetMesh();
        computeData->setStaticBuffer(volumeRenderer->getUniformDataBuffer(), "UniformDataBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleCounterBuffer(), "TriangleCounterBuffer");
        computeData->setStaticBuffer(tetMesh->getCellIndicesBuffer(), "TetIndexBuffer");
        computeData->setStaticBuffer(tetMesh->getVertexPositionBuffer(), "TetVertexPositionBuffer");
        computeData->setStaticBuffer(tetMesh->getVertexColorBuffer(), "TetVertexColorBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleVertexPositionBuffer(), "TriangleVertexPositionBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleVertexColorBuffer(), "TriangleVertexColorBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleVertexDepthBuffer(), "TriangleVertexDepthBuffer");
        computeData->setStaticBufferOptional(volumeRenderer->getTriangleTetIndexBuffer(), "TriangleTetIndexBuffer");
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
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                volumeRenderer->getTriangleVertexDepthBuffer());
        if (tetMesh->getUseGradients()) {
            renderer->insertBufferMemoryBarrier(
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    volumeRenderer->getTriangleTetIndexBuffer());
        }
    }

private:
    TetMeshRendererProjection* volumeRenderer;
    const uint32_t BLOCK_SIZE = 256;
};

class InitializeIndirectCommandBufferProjPass : public sgl::vk::ComputePass {
public:
    explicit InitializeIndirectCommandBufferProjPass(TetMeshRendererProjection* volumeRenderer)
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
    TetMeshRendererProjection* volumeRenderer;
    const uint32_t BLOCK_SIZE = 256; // Value of ComputeTrianglesDepthPass; only used for computing #workgroups.
};

class ComputeTrianglesDepthProjPass : public sgl::vk::ComputePass {
public:
    explicit ComputeTrianglesDepthProjPass(TetMeshRendererProjection* volumeRenderer)
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
    TetMeshRendererProjection* volumeRenderer;
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
        preprocessorDefines.insert(std::make_pair("BACK_TO_FRONT_BLENDING", ""));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "ProjectedRasterization.Vertex", "ProjectedRasterization.Fragment" }, preprocessorDefines);
    }
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override {
        rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
        rasterData->setStaticBuffer(volumeRenderer->getSortedTriangleKeyValueBuffer(), "TriangleKeyValueBuffer");
        rasterData->setStaticBufferOptional(volumeRenderer->getTriangleCounterBuffer(), "TriangleCounterBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexPositionBuffer(), "TriangleVertexPositionBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexColorBuffer(), "TriangleVertexColorBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexDepthBuffer(), "TriangleVertexDepthBuffer");
        rasterData->setIndirectDrawBuffer(volumeRenderer->getDrawIndirectBuffer(), sizeof(VkDrawIndirectCommand));
        rasterData->setIndirectDrawCount(1);
        volumeRenderer->setRenderDataBindings(rasterData);
    }
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) override {
        pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
        pipelineInfo.setIsFrontFaceCcw(true);
        pipelineInfo.setColorWriteEnabled(true);
        pipelineInfo.setDepthWriteEnabled(false);
        pipelineInfo.setDepthTestEnabled(false);
        pipelineInfo.setBlendMode(sgl::vk::BlendMode::BACK_TO_FRONT_PREMUL_ALPHA);
        //pipelineInfo.setBlendMode(sgl::vk::BlendMode::BACK_TO_FRONT_STRAIGHT_ALPHA);
    }
    void _render() override {
        renderer->render(rasterData, framebuffer);
    }

private:
    TetMeshRendererProjection* volumeRenderer;
};

class AdjointProjectedRasterPass : public sgl::vk::RasterPass {
public:
    explicit AdjointProjectedRasterPass(TetMeshRendererProjection* volumeRenderer)
            : RasterPass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
        auto* device = volumeRenderer->getRenderer()->getDevice();
        if (!device->isDeviceExtensionSupported(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME)) {
            sgl::Logfile::get()->throwError(
                    std::string() + "Error in AdjointIntersectRasterPass::AdjointIntersectRasterPass: "
                    + "The extension \"" + VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME
                    + "\" is not supported by the used device.");
        }
    }
    void recreateSwapchain(uint32_t width, uint32_t height) override {
        framebuffer = std::make_shared<sgl::vk::Framebuffer>(device, width, height);

        sgl::vk::AttachmentState attachmentState;
        attachmentState.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentState.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentState.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachmentState.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        framebuffer->setColorAttachment(outputImageView, 0, attachmentState, glm::vec4(0.0f));

        framebufferDirty = true;
        dataDirty = true;
    }
    void setOutputImage(sgl::vk::ImageViewPtr& imageView) {
        outputImageView = imageView;
        setDataDirty();
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BACKWARD_PASS", ""));
        preprocessorDefines.insert(std::make_pair("PI_SQRT", std::to_string(std::sqrt(sgl::PI))));
        preprocessorDefines.insert(std::make_pair("INV_PI_SQRT", std::to_string(1.0f / std::sqrt(sgl::PI))));
        if (renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat32AtomicAdd) {
            preprocessorDefines.insert(std::make_pair("SUPPORT_BUFFER_FLOAT_ATOMIC_ADD", ""));
            preprocessorDefines.insert(std::make_pair(
                    "__extensions", "GL_EXT_shader_atomic_float;GL_EXT_control_flow_attributes"));
        }
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "AdjointProjectedRasterization.Vertex", "AdjointProjectedRasterization.Fragment" },
                preprocessorDefines);
    }
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override {
        rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
        rasterData->setStaticBuffer(volumeRenderer->getSortedTriangleKeyValueBuffer(), "TriangleKeyValueBuffer");
        rasterData->setStaticBufferOptional(volumeRenderer->getTriangleCounterBuffer(), "TriangleCounterBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexPositionBuffer(), "TriangleVertexPositionBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexColorBuffer(), "TriangleVertexColorBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexDepthBuffer(), "TriangleVertexDepthBuffer");
        rasterData->setStaticBuffer(
                volumeRenderer->getTriangleVertexPositionGradientBuffer(), "VertexPositionGradientBuffer");
        rasterData->setStaticBuffer(
                volumeRenderer->getTriangleVertexColorGradientBuffer(), "VertexColorGradientBuffer");
        rasterData->setStaticBuffer(
                volumeRenderer->getTriangleVertexDepthGradientBuffer(), "VertexDepthGradientBuffer");
        rasterData->setIndirectDrawBuffer(volumeRenderer->getDrawIndirectBuffer(), sizeof(VkDrawIndirectCommand));
        rasterData->setIndirectDrawCount(1);
        volumeRenderer->setRenderDataBindings(rasterData);

    }
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) override {
        pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
        pipelineInfo.setIsFrontFaceCcw(true);
        pipelineInfo.setColorWriteEnabled(false);
        pipelineInfo.setDepthWriteEnabled(false);
        pipelineInfo.setDepthTestEnabled(false);
        pipelineInfo.setBlendMode(sgl::vk::BlendMode::OVERWRITE);
    }

private:
    TetMeshRendererProjection* volumeRenderer;
    sgl::vk::ImageViewPtr outputImageView;
};

class CompactTriangleTetListPass : public sgl::vk::ComputePass {
public:
    explicit CompactTriangleTetListPass(TetMeshRendererProjection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "CompactTriangleTetList.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        computeData->setStaticBuffer(volumeRenderer->getTriangleCounterBuffer(), "TriangleCounterBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTriangleTetIndexBuffer(), "TriangleTetIndexBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTetCounterBuffer(), "TetCounterBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTetTriangleOffsetBuffer(), "TetTriangleOffsetBuffer");
        volumeRenderer->setRenderDataBindings(computeData);
    }
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {
    }
    void _render() override {
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                volumeRenderer->getTetCounterBuffer());
        const sgl::vk::BufferPtr& tetCounterBuffer = volumeRenderer->getTetCounterBuffer();
        tetCounterBuffer->fill(0, renderer->getVkCommandBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                tetCounterBuffer);
        renderer->dispatchIndirect(computeData, volumeRenderer->getDispatchIndirectBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                volumeRenderer->getTetTriangleOffsetBuffer());
    }

private:
    TetMeshRendererProjection* volumeRenderer;
    const uint32_t BLOCK_SIZE = 256;
};

class InitializeIndirectCommandBufferAdjointPass : public sgl::vk::ComputePass {
public:
    explicit InitializeIndirectCommandBufferAdjointPass(TetMeshRendererProjection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "InitializeIndirectCommandBufferAdjoint.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        computeData->setStaticBuffer(volumeRenderer->getTetCounterBuffer(), "TetCounterBuffer");
        computeData->setStaticBuffer(
                volumeRenderer->getDispatchIndirectAdjointBuffer(), "DispatchIndirectCommandBuffer");
        volumeRenderer->setRenderDataBindings(computeData);
    }
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {
    }
    void _render() override {
        renderer->dispatch(computeData, 1, 1, 1);
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                volumeRenderer->getDispatchIndirectAdjointBuffer());
    }

private:
    TetMeshRendererProjection* volumeRenderer;
    const uint32_t BLOCK_SIZE = 256; // Value of AdjointGenerateTrianglesPass; only used for computing #workgroups.
};

class AdjointGenerateTrianglesPass : public sgl::vk::ComputePass {
public:
    explicit AdjointGenerateTrianglesPass(TetMeshRendererProjection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BACKWARD_PASS", ""));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
        preprocessorDefines.insert(std::make_pair("PI_SQRT", std::to_string(std::sqrt(sgl::PI))));
        preprocessorDefines.insert(std::make_pair("INV_PI_SQRT", std::to_string(1.0f / std::sqrt(sgl::PI))));
        if (renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat32AtomicAdd) {
            preprocessorDefines.insert(std::make_pair("SUPPORT_BUFFER_FLOAT_ATOMIC_ADD", ""));
            preprocessorDefines.insert(std::make_pair(
                    "__extensions", "GL_EXT_shader_atomic_float;GL_EXT_control_flow_attributes"));
        }
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "AdjointGenerateTrianglesVTK.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        const auto& tetMesh = volumeRenderer->getTetMesh();
        computeData->setStaticBuffer(volumeRenderer->getUniformDataBuffer(), "UniformDataBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTetCounterBuffer(), "TetCounterBuffer");
        computeData->setStaticBuffer(volumeRenderer->getTetTriangleOffsetBuffer(), "TetTriangleOffsetBuffer");

        computeData->setStaticBuffer(tetMesh->getCellIndicesBuffer(), "TetIndexBuffer");
        computeData->setStaticBuffer(tetMesh->getVertexPositionBuffer(), "TetVertexPositionBuffer");
        computeData->setStaticBuffer(tetMesh->getVertexColorBuffer(), "TetVertexColorBuffer");

        computeData->setStaticBuffer(volumeRenderer->getTriangleTetIndexBuffer(), "TriangleTetIndexBuffer");

        computeData->setStaticBuffer(
                volumeRenderer->getTriangleVertexPositionGradientBuffer(), "TriangleVertexPositionGradientBuffer");
        computeData->setStaticBuffer(
                volumeRenderer->getTriangleVertexColorGradientBuffer(), "TriangleVertexColorGradientBuffer");
        computeData->setStaticBuffer(
                volumeRenderer->getTriangleVertexDepthGradientBuffer(), "TriangleVertexDepthGradientBuffer");

        volumeRenderer->setRenderDataBindings(computeData);
    }
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {
    }
    void _render() override {
        renderer->dispatchIndirect(computeData, volumeRenderer->getDispatchIndirectAdjointBuffer());
    }

private:
    TetMeshRendererProjection* volumeRenderer;
    const uint32_t BLOCK_SIZE = 256;
};

TetMeshRendererProjection::TetMeshRendererProjection(
        sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow)
        : TetMeshVolumeRenderer(renderer, camera, transferFunctionWindow) {
    sgl::vk::Device* device = renderer->getDevice();
    uniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    createTriangleCounterBuffer();
    drawIndirectBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    dispatchIndirectBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(VkDispatchIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    generateTrianglesPass = std::make_shared<GenerateTrianglesProjPass>(this);
    initializeIndirectCommandBufferPass = std::make_shared<InitializeIndirectCommandBufferProjPass>(this);
    computeTrianglesDepthPass = std::make_shared<ComputeTrianglesDepthProjPass>(this);
    projectedRasterPass = std::make_shared<ProjectedRasterPass>(this);

#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
    const auto& physicalDeviceProperties = device->getPhysicalDeviceProperties();
    const auto& subgroupProperties = device->getPhysicalDeviceSubgroupProperties();
    radixSortVkTarget = radix_sort_vk_target_auto_detect(&physicalDeviceProperties, &subgroupProperties, 2u);
    if (!radixSortVkTarget) {
        sgl::Logfile::get()->throwError(
                "Error in TetMeshRendererProjection::TetMeshRendererProjection: Could not detect radix sort target.");
    }
    radixSortVk = radix_sort_vk_create(device->getVkDevice(), nullptr, VK_NULL_HANDLE, radixSortVkTarget);
    if (!radixSortVk) {
        sgl::Logfile::get()->throwError(
                "Error in TetMeshRendererProjection::TetMeshRendererProjection: Could not fetch radix sort implementation.");
    }
#endif

    TetMeshRendererProjection::onClearColorChanged();
}

TetMeshRendererProjection::~TetMeshRendererProjection() {
    renderer->getDevice()->waitIdle();
#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
    sgl::vk::Device* device = renderer->getDevice();
    if (radixSortVk) {
        radix_sort_vk_destroy(radixSortVk, device->getVkDevice(), nullptr);
    }
#endif
}

void TetMeshRendererProjection::setTetMeshData(const TetMeshPtr& _tetMesh) {
    TetMeshVolumeRenderer::setTetMeshData(_tetMesh);

    bool useGradientsChanged = false;
    if (useGradientsCached != tetMesh->getUseGradients()) {
        useGradientsCached = tetMesh->getUseGradients();
        setShadersDirty(VolumeRendererPassType::GATHER);
        useGradientsChanged = true;
    }

    triangleVertexPositionBuffer = {};
    triangleVertexColorBuffer = {};
    triangleVertexDepthBuffer = {};
    triangleTetIndexBuffer = {};
    tetTriangleOffsetBuffer = {};
    triangleVertexPositionGradientBuffer = {};
    triangleVertexColorGradientBuffer = {};
    triangleVertexDepthGradientBuffer = {};

    sgl::vk::Device* device = renderer->getDevice();
    size_t maxNumProjectedTriangles = tetMesh->getNumCells() * 4;
    size_t maxNumProjectedVertices = maxNumProjectedTriangles * 3;
    triangleVertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, maxNumProjectedVertices * sizeof(glm::vec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    triangleVertexColorBuffer = std::make_shared<sgl::vk::Buffer>(
            device, maxNumProjectedVertices * sizeof(glm::vec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    triangleVertexDepthBuffer = std::make_shared<sgl::vk::Buffer>(
            device, maxNumProjectedVertices * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    if (tetMesh->getUseGradients()) {
        triangleTetIndexBuffer = std::make_shared<sgl::vk::Buffer>(
                device, maxNumProjectedTriangles * sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        tetTriangleOffsetBuffer = std::make_shared<sgl::vk::Buffer>(
                device, tetMesh->getNumCells() * sizeof(uint32_t),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        // Add VK_BUFFER_USAGE_TRANSFER_SRC_BIT for debugging.
        triangleVertexPositionGradientBuffer = std::make_shared<sgl::vk::Buffer>(
                device, maxNumProjectedVertices * sizeof(glm::vec3),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        triangleVertexColorGradientBuffer = std::make_shared<sgl::vk::Buffer>(
                device, maxNumProjectedVertices * sizeof(glm::vec4),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        triangleVertexDepthGradientBuffer = std::make_shared<sgl::vk::Buffer>(
                device, maxNumProjectedVertices * sizeof(float),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    }

    if (useGradientsChanged) {
        tetCounterBuffer = {};
        dispatchIndirectAdjointBuffer = {};
    }
    if (useGradientsChanged && tetMesh->getUseGradients()) {
        tetCounterBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(uint32_t),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        dispatchIndirectAdjointBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(VkDispatchIndirectCommand),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
    }

    recreateSortingBuffers();
}

void TetMeshRendererProjection::createTriangleCounterBuffer() {
    sgl::vk::Device* device = renderer->getDevice();
    VkBufferUsageFlags usage =
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (sortingAlgorithm == SortingAlgorithm::CPU_STD_SORT) {
        usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    triangleCounterBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t), usage, VMA_MEMORY_USAGE_GPU_ONLY);
    if (initializeIndirectCommandBufferPass) {
        initializeIndirectCommandBufferPass->setDataDirty();
    }
}

void TetMeshRendererProjection::recreateSortingBuffers() {
    sgl::vk::Device* device = renderer->getDevice();
    size_t maxNumProjectedTriangles = tetMesh->getNumCells() * 4;

    sortingBufferEven = {};
    sortingBufferOdd = {};
    sortedTriangleKeyValueBuffer = {};
    sortingInternalBuffer = {};
    sortingIndirectBuffer = {};
    triangleKeyValueBufferCpu = {};
    triangleCounterBufferCpu = {};
    if (sortingAlgorithm == SortingAlgorithm::FUCHSIA_RADIX_SORT) {
#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
        radix_sort_vk_memory_requirements memoryRequirements{};
        radix_sort_vk_get_memory_requirements(radixSortVk, uint32_t(maxNumProjectedTriangles), &memoryRequirements);

        assert(memoryRequirements.keyval_size == sizeof(uint64_t));
        sgl::vk::BufferSettings bufferSettings{};
        bufferSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;

        bufferSettings.sizeInBytes = memoryRequirements.keyvals_size;
        bufferSettings.alignment = uint32_t(memoryRequirements.keyvals_alignment);
        bufferSettings.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        sortingBufferEven = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);
        sortingBufferOdd = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);
        triangleKeyValueBuffer = sortingBufferEven;
        sortedTriangleKeyValueBuffer = {};

        bufferSettings.sizeInBytes = memoryRequirements.internal_size;
        bufferSettings.alignment = uint32_t(memoryRequirements.internal_alignment);
        bufferSettings.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sortingInternalBuffer = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);

        bufferSettings.sizeInBytes = memoryRequirements.indirect_size;
        bufferSettings.alignment = uint32_t(memoryRequirements.indirect_alignment);
        bufferSettings.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        sortingIndirectBuffer = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);
#endif
    } else if (sortingAlgorithm == SortingAlgorithm::CPU_STD_SORT) {
        triangleKeyValueBuffer = std::make_shared<sgl::vk::Buffer>(
                device, maxNumProjectedTriangles * sizeof(uint64_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        sortedTriangleKeyValueBuffer = triangleKeyValueBuffer;

        triangleKeyValueBufferCpu = std::make_shared<sgl::vk::Buffer>(
                device, maxNumProjectedTriangles * sizeof(uint64_t),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
        triangleCounterBufferCpu = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(uint32_t),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    }

    generateTrianglesPass->setDataDirty();
    computeTrianglesDepthPass->setDataDirty();
    projectedRasterPass->setDataDirty();
    if (adjointProjectedRasterPass) {
        adjointProjectedRasterPass->setDataDirty();
        compactTriangleTetListPass->setDataDirty();
        initializeIndirectCommandBufferAdjointPass->setDataDirty();
        adjointGenerateTrianglesPass->setDataDirty();
    }
}

void TetMeshRendererProjection::setAdjointPassData(
        sgl::vk::ImageViewPtr _colorAdjointImage, sgl::vk::ImageViewPtr _adjointPassBackbuffer) {
    TetMeshVolumeRenderer::setAdjointPassData(
            std::move(_colorAdjointImage), std::move(_adjointPassBackbuffer));
    if (!adjointProjectedRasterPass) {
        adjointProjectedRasterPass = std::make_shared<AdjointProjectedRasterPass>(this);
        compactTriangleTetListPass = std::make_shared<CompactTriangleTetListPass>(this);
        initializeIndirectCommandBufferAdjointPass = std::make_shared<InitializeIndirectCommandBufferAdjointPass>(this);
        adjointGenerateTrianglesPass = std::make_shared<AdjointGenerateTrianglesPass>(this);
    }
    adjointProjectedRasterPass->setDataDirty();
    compactTriangleTetListPass->setDataDirty();
    initializeIndirectCommandBufferAdjointPass->setDataDirty();
    adjointGenerateTrianglesPass->setDataDirty();
}

void TetMeshRendererProjection::recreateSwapchain(uint32_t width, uint32_t height) {
    useExternalFragmentBuffer = false;
    TetMeshVolumeRenderer::recreateSwapchain(width, height);

    startOffsetBuffer = {};
    fragmentCounterBuffer = {};

    generateTrianglesPass->recreateSwapchain(width, height);
    computeTrianglesDepthPass->recreateSwapchain(width, height);
    //projectedRasterPass->setOutputImage(outputImageView);
    projectedRasterPass->recreateSwapchain(width, height);

    if (adjointProjectedRasterPass) {
        adjointProjectedRasterPass->setOutputImage(adjointPassBackbuffer);
        adjointProjectedRasterPass->recreateSwapchain(width, height);
        compactTriangleTetListPass->recreateSwapchain(width, height);
        initializeIndirectCommandBufferAdjointPass->recreateSwapchain(width, height);
        adjointGenerateTrianglesPass->recreateSwapchain(width, height);
    }
}

void TetMeshRendererProjection::recreateSwapchainExternal(
        uint32_t width, uint32_t height, size_t _fragmentBufferSize, const sgl::vk::BufferPtr& _fragmentBuffer,
        const sgl::vk::BufferPtr& _startOffsetBuffer, const sgl::vk::BufferPtr& _fragmentCounterBuffer) {
    useExternalFragmentBuffer = true;
    TetMeshVolumeRenderer::recreateSwapchainExternal(
            width, height, _fragmentBufferSize, _fragmentBuffer, _startOffsetBuffer, _fragmentCounterBuffer);

    generateTrianglesPass->recreateSwapchain(width, height);
    computeTrianglesDepthPass->recreateSwapchain(width, height);
    //projectedRasterPass->setOutputImage(outputImageView);
    projectedRasterPass->recreateSwapchain(width, height);

    if (adjointProjectedRasterPass) {
        adjointProjectedRasterPass->setOutputImage(adjointPassBackbuffer);
        adjointProjectedRasterPass->recreateSwapchain(width, height);
        compactTriangleTetListPass->recreateSwapchain(width, height);
        initializeIndirectCommandBufferAdjointPass->recreateSwapchain(width, height);
        adjointGenerateTrianglesPass->recreateSwapchain(width, height);
    }
}

void TetMeshRendererProjection::getVulkanShaderPreprocessorDefines(
        std::map<std::string, std::string>& preprocessorDefines) {
    TetMeshVolumeRenderer::getVulkanShaderPreprocessorDefines(preprocessorDefines);
}

void TetMeshRendererProjection::setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData) {
    TetMeshVolumeRenderer::setRenderDataBindings(renderData);

    renderData->setStaticBufferOptional(uniformDataBuffer, "UniformDataBuffer");
}

void TetMeshRendererProjection::setFramebufferAttachments(sgl::vk::FramebufferPtr& framebuffer, VkAttachmentLoadOp loadOp) {
    TetMeshVolumeRenderer::setFramebufferAttachments(framebuffer, loadOp);
}

void TetMeshRendererProjection::onClearColorChanged() {
    projectedRasterPass->setAttachmentClearColor(clearColor.getFloatColorRGBA());
}

/*template<class T>
void printBuffer(const sgl::vk::BufferPtr& bufferGpu) {
    auto bufferCpu = std::make_shared<sgl::vk::Buffer>(
            bufferGpu->getDevice(), bufferGpu->getSizeInBytes(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    auto commandBuffer = bufferGpu->getDevice()->beginSingleTimeCommands();
    bufferGpu->copyDataTo(bufferCpu, commandBuffer);
    bufferGpu->getDevice()->endSingleTimeCommands(commandBuffer);
    size_t numEntries = bufferGpu->getSizeInBytes() / sizeof(T);
    auto* dataPtr = reinterpret_cast<T*>(bufferCpu->mapMemory());
    for (size_t i = 0; i < numEntries; i++) {
        std::cout << dataPtr[i];
        if (i != numEntries - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
    bufferCpu->unmapMemory();
}*/

void TetMeshRendererProjection::computeLinearDepthCorrection() {
    /*
     * TODO: We could adapt the code from
     * https://github.com/Kitware/VTK/blob/master/Rendering/VolumeOpenGL2/vtkOpenGLProjectedTetrahedraMapper.cxx
     * to support linear depth correction.
     */
    uniformData.useLinearDepthCorrection = false;
    uniformData.linearDepthCorrection = 1.0f;
}

void TetMeshRendererProjection::render() {
    uniformData.viewProjMat = (*camera)->getViewProjMatrix();
    uniformData.invProjMat = glm::inverse((*camera)->getProjectionMatrix());
    uniformData.cameraPosition = (*camera)->getPosition();
    uniformData.attenuationCoefficient = attenuationCoefficient;
    uniformData.numTets = uint32_t(tetMesh->getNumCells());
    uniformData.earlyRayTerminationAlpha = 1.0f - earlyRayOutThresh;
    uniformDataBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    generateTrianglesPass->render();
    initializeIndirectCommandBufferPass->render();
    computeTrianglesDepthPass->render();

    if (sortingAlgorithm == SortingAlgorithm::FUCHSIA_RADIX_SORT) {
#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
        radix_sort_vk_sort_indirect_info sortInfo{};
        sortInfo.ext = nullptr;
        sortInfo.key_bits = 32u;

        VkDescriptorBufferInfo descriptorBufferInfo{};
        descriptorBufferInfo.buffer = triangleCounterBuffer->getVkBuffer();
        descriptorBufferInfo.range = triangleCounterBuffer->getSizeInBytes();
        sortInfo.count = descriptorBufferInfo;

        descriptorBufferInfo.buffer = sortingBufferEven->getVkBuffer();
        descriptorBufferInfo.range = sortingBufferEven->getSizeInBytes();
        sortInfo.keyvals_even = descriptorBufferInfo;

        descriptorBufferInfo.buffer = sortingBufferOdd->getVkBuffer();
        descriptorBufferInfo.range = sortingBufferOdd->getSizeInBytes();
        sortInfo.keyvals_odd = descriptorBufferInfo;

        descriptorBufferInfo.buffer = sortingInternalBuffer->getVkBuffer();
        descriptorBufferInfo.range = sortingInternalBuffer->getSizeInBytes();
        sortInfo.internal = descriptorBufferInfo;

        descriptorBufferInfo.buffer = sortingIndirectBuffer->getVkBuffer();
        descriptorBufferInfo.range = sortingIndirectBuffer->getSizeInBytes();
        sortInfo.indirect = descriptorBufferInfo;

        radix_sort_vk_sort_indirect(
                radixSortVk, &sortInfo, renderer->getDevice()->getVkDevice(), renderer->getVkCommandBuffer(),
                &descriptorBufferInfo);
        if (!sortedTriangleKeyValueBuffer || descriptorBufferInfo.buffer != sortedTriangleKeyValueBuffer->getVkBuffer()) {
            if (sortedTriangleKeyValueBuffer) {
                renderer->getDevice()->waitIdle();
                projectedRasterPass->setDataDirty();
                if (adjointProjectedRasterPass) {
                    adjointProjectedRasterPass->setDataDirty();
                }
            }
            if (descriptorBufferInfo.buffer == sortingBufferEven->getVkBuffer()) {
                sortedTriangleKeyValueBuffer = sortingBufferEven;
            } else {
                sortedTriangleKeyValueBuffer = sortingBufferOdd;
            }
        }
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                sortedTriangleKeyValueBuffer);

        // Debug code.
        /*renderer->syncWithCpu();
        printBuffer<uint32_t>(triangleCounterBuffer);
        printBuffer<uint32_t>(sortingBufferEven);
        printBuffer<uint32_t>(sortingBufferOdd);
        printBuffer<uint32_t>(sortedTriangleKeyValueBuffer);
        printBuffer<uint32_t>(sortingIndirectBuffer);
        exit(1);*/
#endif
    } else if (sortingAlgorithm == SortingAlgorithm::CPU_STD_SORT) {
        triangleCounterBuffer->copyDataTo(triangleCounterBufferCpu, renderer->getVkCommandBuffer());
        triangleKeyValueBuffer->copyDataTo(triangleKeyValueBufferCpu, renderer->getVkCommandBuffer());
        renderer->syncWithCpu();
        auto triangleCount = *reinterpret_cast<uint32_t*>(triangleCounterBufferCpu->mapMemory());
        triangleCounterBufferCpu->unmapMemory();
        auto* triangleKeyValueData = reinterpret_cast<uint64_t*>(triangleKeyValueBufferCpu->mapMemory());
        std::sort(triangleKeyValueData, triangleKeyValueData + triangleCount);
        triangleKeyValueBufferCpu->unmapMemory();
        triangleKeyValueBufferCpu->copyDataTo(sortedTriangleKeyValueBuffer, renderer->getVkCommandBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                sortedTriangleKeyValueBuffer);
    }

    if (showDepthComplexity) {
        depthComplexityCounterBuffer->fill(0, renderer->getVkCommandBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                depthComplexityCounterBuffer);
        auto viewportLinearW = uint32_t(windowWidth);
        projectedRasterPass->buildIfNecessary();
        renderer->pushConstants(
                projectedRasterPass->getGraphicsPipeline(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, viewportLinearW);
    }
    projectedRasterPass->render();
}

void TetMeshRendererProjection::renderAdjoint() {
    triangleVertexPositionGradientBuffer->fill(0, renderer->getVkCommandBuffer());
    triangleVertexColorGradientBuffer->fill(0, renderer->getVkCommandBuffer());
    triangleVertexDepthGradientBuffer->fill(0, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    adjointProjectedRasterPass->render();

    // TODO: Debug code.
    /*auto* device = renderer->getDevice();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    auto triangleVertexPositionGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
            device, triangleVertexPositionGradientBuffer->getSizeInBytes(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    auto triangleVertexColorGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
            device, triangleVertexColorGradientBuffer->getSizeInBytes(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    auto triangleVertexDepthGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
            device, triangleVertexDepthGradientBuffer->getSizeInBytes(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    triangleVertexPositionGradientBuffer->copyDataTo(triangleVertexPositionGradientStagingBuffer, renderer->getVkCommandBuffer());
    triangleVertexColorGradientBuffer->copyDataTo(triangleVertexColorGradientStagingBuffer, renderer->getVkCommandBuffer());
    triangleVertexDepthGradientBuffer->copyDataTo(triangleVertexDepthGradientStagingBuffer, renderer->getVkCommandBuffer());
    renderer->syncWithCpu();
    auto* vertexPositionGrads = reinterpret_cast<glm::vec3*>(triangleVertexPositionGradientStagingBuffer->mapMemory());
    auto* vertexColorGrads = reinterpret_cast<glm::vec4*>(triangleVertexColorGradientStagingBuffer->mapMemory());
    auto* vertexDepthGrads = reinterpret_cast<float*>(triangleVertexDepthGradientStagingBuffer->mapMemory());
    auto numVerts = int(triangleVertexDepthGradientStagingBuffer->getSizeInBytes() / sizeof(float));
    for (int i = 0; i < numVerts; i++) {
        std::cout << "v_idx: " << vertexPositionGrads[i].x << std::endl;
        std::cout << "p: " << vertexPositionGrads[i].x << " " << vertexPositionGrads[i].y << " " << vertexPositionGrads[i].z << std::endl;
        std::cout << "c: " << vertexColorGrads[i].x << " " << vertexColorGrads[i].y << " " << vertexColorGrads[i].z << " " << vertexColorGrads[i].w << std::endl;
        std::cout << "d: " << vertexDepthGrads[i] << std::endl;
    }
    triangleVertexPositionGradientStagingBuffer->unmapMemory();
    triangleVertexColorGradientStagingBuffer->unmapMemory();
    triangleVertexDepthGradientStagingBuffer->unmapMemory();*/

    compactTriangleTetListPass->render();

    initializeIndirectCommandBufferAdjointPass->render();

    adjointGenerateTrianglesPass->buildIfNecessary();
    uint32_t useAbsGradUint = useAbsGrad ? 1 : 0;
    renderer->pushConstants(
            adjointGenerateTrianglesPass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, useAbsGradUint);
    adjointGenerateTrianglesPass->render();
}

void TetMeshRendererProjection::setShadersDirty(VolumeRendererPassType passType) {
    if ((int(passType) & int(VolumeRendererPassType::GATHER)) != 0) {
        generateTrianglesPass->setShaderDirty();
        computeTrianglesDepthPass->setShaderDirty();
    }
    if ((int(passType) & int(VolumeRendererPassType::RESOLVE)) != 0) {
        projectedRasterPass->setShaderDirty();
    }
    if ((int(passType) & int(VolumeRendererPassType::OTHER)) != 0) {
        //sortPass->setShaderDirty();
        if (adjointProjectedRasterPass) {
            adjointProjectedRasterPass->setShaderDirty();
            compactTriangleTetListPass->setShaderDirty();
            initializeIndirectCommandBufferAdjointPass->setShaderDirty();
            adjointGenerateTrianglesPass->setShaderDirty();
        }
    }
}

#ifndef DISABLE_IMGUI
void TetMeshRendererProjection::renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.addCombo(
            "Sorting Algorithm", (int*)&sortingAlgorithm,
            SORTING_ALGORITHM_NAMES, IM_ARRAYSIZE(SORTING_ALGORITHM_NAMES))) {
        renderer->syncWithCpu();
        renderer->getDevice()->waitIdle();
        createTriangleCounterBuffer();
        recreateSortingBuffers();
        reRender = true;
    }
    renderGuiShared(propertyEditor);
}
#endif
