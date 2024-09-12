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
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "GenerateTriangles.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        const auto& tetMesh = volumeRenderer->getTetMesh();
        computeData->setStaticBuffer(tetMesh->getTriangleIndexBuffer(), "TriangleCounterBuffer");

        // TODO

        // Input tet data.
        //computeData->setStaticBuffer(tetMesh->getTetIndexBuffer(), "TetIndexBuffer");
        computeData->setStaticBuffer(tetMesh->getVertexPositionBuffer(), "TetVertexPositionBuffer");
        computeData->setStaticBuffer(tetMesh->getVertexPositionBuffer(), "TetVertexColorBuffer");

        // Output triangle data.
        computeData->setStaticBuffer(tetMesh->getVertexPositionBuffer(), "VertexPositionBuffer");
        computeData->setStaticBuffer(tetMesh->getFaceBoundaryBitBuffer(), "VertexColorBuffer");

        volumeRenderer->setRenderDataBindings(computeData);
    }
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {
    }

private:
    TetMeshRendererProjection* volumeRenderer;
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
        const auto& tetMesh = volumeRenderer->getTetMesh();
        //rasterData->setIndexBuffer(tetMesh->getTriangleIndexBuffer());
        //rasterData->setVertexBuffer(tetMesh->getVertexPositionBuffer(), "vertexPosition");
        //rasterData->setVertexBuffer(tetMesh->getVertexColorBuffer(), "vertexColor");
        auto numIndexedVertices = tetMesh->getTriangleIndexBuffer()->getSizeInBytes() / sizeof(uint32_t);
        rasterData->setStaticBuffer(tetMesh->getTriangleIndexBuffer(), "TriangleIndicesBuffer");
        rasterData->setStaticBuffer(tetMesh->getVertexPositionBuffer(), "VertexPositionBuffer");
        //rasterData->setStaticBuffer(tetMesh->getVertexColorBuffer(), "VertexColorBuffer");
        rasterData->setStaticBuffer(tetMesh->getFaceBoundaryBitBuffer(), "FaceBoundaryBitBuffer");
        rasterData->setNumVertices(numIndexedVertices);
        volumeRenderer->setRenderDataBindings(rasterData);
    }
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
        //pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_BACK);
        pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
        pipelineInfo.setIsFrontFaceCcw(true);
        pipelineInfo.setColorWriteEnabled(false);
        pipelineInfo.setDepthWriteEnabled(false);
        pipelineInfo.setDepthTestEnabled(false);
        //pipelineInfo.setVertexBufferBindingByLocationIndex("vertexPosition", sizeof(glm::vec3));
        //pipelineInfo.setVertexBufferBindingByLocationIndex("vertexColor", sizeof(glm::vec4));
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

    /*gatherRasterPass = std::make_shared<GatherRasterPass>(this);

    resolveRasterPass = std::make_shared<ResolveRasterPass>(this);
    resolveRasterPass->setOutputImageFinalLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    resolveRasterPass->setBlendMode(sgl::vk::BlendMode::BACK_TO_FRONT_PREMUL_ALPHA);

    clearRasterPass = std::make_shared<ClearRasterPass>(this);
    clearRasterPass->setColorWriteEnabled(false);
    clearRasterPass->setAttachmentLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    clearRasterPass->setAttachmentStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE);
    clearRasterPass->setOutputImageInitialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
    clearRasterPass->setOutputImageFinalLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);*/

    TetMeshRendererProjection::onClearColorChanged();
}

TetMeshRendererProjection::~TetMeshRendererProjection() {
}

void TetMeshRendererProjection::setTetMeshData(const TetMeshPtr& _tetMesh) {
    TetMeshVolumeRenderer::setTetMeshData(_tetMesh);
    //gatherRasterPass->setDataDirty();
    //resolveRasterPass->setDataDirty();
    //clearRasterPass->setDataDirty();
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

    /*gatherRasterPass->recreateSwapchain(width, height);

    resolveRasterPass->setOutputImage(outputImageView);
    resolveRasterPass->recreateSwapchain(width, height);

    clearRasterPass->setOutputImage(outputImageView);
    clearRasterPass->recreateSwapchain(width, height);

    if (adjointRasterPass) {
        adjointRasterPass->setOutputImage(adjointPassBackbuffer);
        adjointRasterPass->recreateSwapchain(width, height);
    }*/
}

void TetMeshRendererProjection::recreateSwapchainExternal(
        uint32_t width, uint32_t height, size_t _fragmentBufferSize, sgl::vk::BufferPtr _fragmentBuffer,
        sgl::vk::BufferPtr _startOffsetBuffer, sgl::vk::BufferPtr _fragmentCounterBuffer) {
    useExternalFragmentBuffer = true;
    TetMeshVolumeRenderer::recreateSwapchainExternal(
            width, height, _fragmentBufferSize,
            std::move(_fragmentBuffer),
            std::move(_startOffsetBuffer),
            std::move(_fragmentCounterBuffer));

    /*gatherRasterPass->recreateSwapchain(width, height);

    resolveRasterPass->setOutputImage(outputImageView);
    resolveRasterPass->recreateSwapchain(width, height);

    clearRasterPass->setOutputImage(outputImageView);
    clearRasterPass->recreateSwapchain(width, height);

    if (adjointRasterPass) {
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
    /*resolveRasterPass->setAttachmentClearColor(clearColor.getFloatColorRGBA());
    if ((clearColor.getA() == 0) != (alphaMode == AlphaMode::STRAIGHT)) {
        resolveRasterPass->setShaderDirty();
        alphaMode = clearColor.getA() == 0 ? AlphaMode::STRAIGHT : AlphaMode::PREMUL;
        resolveRasterPass->setBlendMode(
                alphaMode == AlphaMode::PREMUL
                ? sgl::vk::BlendMode::BACK_TO_FRONT_PREMUL_ALPHA : sgl::vk::BlendMode::OVERWRITE);
    }*/
}

void TetMeshRendererProjection::render() {
    auto imageSettings = outputImageView->getImage()->getImageSettings();
    uniformData.inverseViewProjectionMatrix = (*camera)->getInverseViewProjMatrix();

    uniformData.linkedListSize = static_cast<uint32_t>(fragmentBufferSize);
    uniformData.viewportW = paddedWindowWidth;
    uniformData.viewportSize = glm::uvec2(imageSettings.width, imageSettings.height);
    uniformData.viewportLinearW = int(imageSettings.width);
    uniformData.zNear = (*camera)->getNearClipDistance();
    uniformData.zFar = (*camera)->getFarClipDistance();
    uniformData.cameraFront = (*camera)->getCameraFront();
    uniformData.cameraPosition = (*camera)->getPosition();
    uniformData.attenuationCoefficient = attenuationCoefficient;
    uniformDataBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    /*clear();
    gather();
    resolve();*/
}

void TetMeshRendererProjection::renderAdjoint() {
    //adjointRasterPass->render();
}

void TetMeshRendererProjection::setShadersDirty(VolumeRendererPassType passType) {
    /*if ((int(passType) & int(VolumeRendererPassType::GATHER)) != 0) {
        generateTrianglesPass->setShaderDirty();
        projectedRasterPass->setShaderDirty();
    }
    if ((int(passType) & int(VolumeRendererPassType::RESOLVE)) != 0) {
        generateTrianglesPass->setShaderDirty();
        projectedRasterPass->setShaderDirty();
    }
    if ((int(passType) & int(VolumeRendererPassType::OTHER)) != 0) {
        sortPass->setShaderDirty();
        if (adjointProjectedRasterPass) {
            adjointProjectedRasterPass->setShaderDirty();
        }
    }*/
}

void TetMeshRendererProjection::renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) {
    renderGuiShared(propertyEditor);
}
