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

#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/Passes/BlitRenderPass.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Tet/TetMesh.hpp"
#include "TetMeshRendererProjection.hpp"

class GenerateTrianglesPass : public sgl::vk::ComputePass {
public:
    explicit GenerateTrianglesPass(TetMeshRendererProjection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", ""));
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
    TetMeshRendererProjection* volumeRenderer;
    const uint32_t BLOCK_SIZE = 256;
};

class InitializeIndirectCommandBufferPass : public sgl::vk::ComputePass {
public:
    explicit InitializeIndirectCommandBufferPass(TetMeshRendererProjection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", ""));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "ComputeTriangleDepths.Compute" }, preprocessorDefines);
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

class ComputeTrianglesDepthPass : public sgl::vk::ComputePass {
public:
    explicit ComputeTrianglesDepthPass(TetMeshRendererProjection* volumeRenderer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", ""));
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
    void clearFragmentBuffer() {
        if (rasterData) {
            rasterData->setStaticBufferOptional({}, "FragmentBuffer");
            rasterData->setStaticBufferArrayOptional({}, "FragmentBuffer");
        }
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("RESOLVE_PASS", ""));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "LinkedListGather.Vertex", "LinkedListGather.Fragment" }, preprocessorDefines);
    }
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
        rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
        rasterData->setStaticBuffer(volumeRenderer->getTriangleKeyValueBuffer(), "TriangleKeyValueBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexPositionBuffer(), "TriangleVertexPositionBuffer");
        rasterData->setStaticBuffer(volumeRenderer->getTriangleVertexColorBuffer(), "TriangleVertexColorBuffer");
        rasterData->setIndirectDrawBuffer(volumeRenderer->getDrawIndirectBuffer(), sizeof(VkDrawIndirectCommand));
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

TetMeshRendererProjection::TetMeshRendererProjection(
        sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow)
        : TetMeshVolumeRenderer(renderer, camera, transferFunctionWindow) {
    sgl::vk::Device* device = renderer->getDevice();
    uniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    triangleCounterBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    drawIndirectBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    dispatchIndirectBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(VkDispatchIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    generateTrianglesPass = std::make_shared<GenerateTrianglesPass>(this);
    initializeIndirectCommandBufferPass = std::make_shared<InitializeIndirectCommandBufferPass>(this);
    computeTrianglesDepthPass = std::make_shared<ComputeTrianglesDepthPass>(this);
    projectedRasterPass = std::make_shared<ProjectedRasterPass>(this);

    TetMeshRendererProjection::onClearColorChanged();
}

TetMeshRendererProjection::~TetMeshRendererProjection() {
}

void TetMeshRendererProjection::setTetMeshData(const TetMeshPtr& _tetMesh) {
    TetMeshVolumeRenderer::setTetMeshData(_tetMesh);

    sgl::vk::Device* device = renderer->getDevice();
    size_t maxNumProjectedTriangles = tetMesh->getNumCells() * 4;
    size_t maxNumProjectedVertices = maxNumProjectedTriangles * 3;
    triangleVertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, maxNumProjectedVertices * sizeof(glm::vec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    triangleVertexColorBuffer = std::make_shared<sgl::vk::Buffer>(
            device, maxNumProjectedVertices * sizeof(glm::vec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    triangleKeyValueBuffer = std::make_shared<sgl::vk::Buffer>(
            device, maxNumProjectedTriangles * sizeof(uint64_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    generateTrianglesPass->setDataDirty();
    computeTrianglesDepthPass->setDataDirty();
    projectedRasterPass->setDataDirty();
}

void TetMeshRendererProjection::setAdjointPassData(
        sgl::vk::ImageViewPtr _colorAdjointImage, sgl::vk::ImageViewPtr _adjointPassBackbuffer,
        sgl::vk::BufferPtr _vertexPositionGradientBuffer, sgl::vk::BufferPtr _vertexColorGradientBuffer) {
    TetMeshVolumeRenderer::setAdjointPassData(
            std::move(_colorAdjointImage),
            std::move(_adjointPassBackbuffer),
            std::move(_vertexPositionGradientBuffer),
            std::move(_vertexColorGradientBuffer));
    /*if (!adjointRasterPass) {
        adjointRasterPass = std::make_shared<AdjointRasterPass>(this);
        adjointRasterPass->setColorWriteEnabled(false);
        adjointRasterPass->setAttachmentLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);
        adjointRasterPass->setAttachmentStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE);
        adjointRasterPass->setOutputImageInitialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        adjointRasterPass->setOutputImageFinalLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        adjointRasterPass->setBlendMode(sgl::vk::BlendMode::OVERWRITE);
    }
    adjointRasterPass->setDataDirty();*/
}

void TetMeshRendererProjection::recreateSwapchain(uint32_t width, uint32_t height) {
    useExternalFragmentBuffer = false;
    TetMeshVolumeRenderer::recreateSwapchain(width, height);

    size_t startOffsetBufferSizeBytes = sizeof(uint32_t) * paddedWindowWidth * paddedWindowHeight;
    startOffsetBuffer = {}; // Delete old data first (-> refcount 0)
    startOffsetBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), startOffsetBufferSizeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    fragmentCounterBuffer = {}; // Delete old data first (-> refcount 0)
    fragmentCounterBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    generateTrianglesPass->recreateSwapchain(width, height);
    computeTrianglesDepthPass->recreateSwapchain(width, height);
    //projectedRasterPass->setOutputImage(outputImageView);
    projectedRasterPass->recreateSwapchain(width, height);

    /*if (adjointRasterPass) {
        //adjointRasterPass->setOutputImage(adjointPassBackbuffer);
        adjointRasterPass->recreateSwapchain(width, height);
    }*/
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

    /*if (adjointRasterPass) {
        adjointRasterPass->setOutputImage(adjointPassBackbuffer);
        adjointRasterPass->recreateSwapchain(width, height);
    }*/
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

void TetMeshRendererProjection::render() {
    uniformData.viewProjMat = (*camera)->getViewProjMatrix();
    uniformData.invProjMat = glm::inverse((*camera)->getProjectionMatrix());
    uniformData.cameraPosition = (*camera)->getPosition();
    uniformData.attenuationCoefficient = attenuationCoefficient;
    uniformData.numTets = uint32_t(tetMesh->getNumCells());
    uniformDataBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    generateTrianglesPass->render();
    initializeIndirectCommandBufferPass->render();
    // TODO: Sort.
    computeTrianglesDepthPass->render();
    projectedRasterPass->render();
}

void TetMeshRendererProjection::renderAdjoint() {
    //adjointRasterPass->render();
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
        /*if (adjointProjectedRasterPass) {
            adjointProjectedRasterPass->setShaderDirty();
        }*/
    }
}

void TetMeshRendererProjection::renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) {
    renderGuiShared(propertyEditor);
}
