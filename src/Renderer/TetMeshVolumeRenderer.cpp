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

#include <Utils/Convert.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/Passes/BlitRenderPass.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Tet/TetMesh.hpp"
#include "TetMeshVolumeRenderer.hpp"

class GatherRasterPass : public sgl::vk::RasterPass {
public:
    explicit GatherRasterPass(TetMeshVolumeRenderer* volumeRenderer)
            : RasterPass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer) {
    }
    void recreateSwapchain(uint32_t width, uint32_t height) override {
        framebuffer = std::make_shared<sgl::vk::Framebuffer>(device, width, height);
        volumeRenderer->setFramebufferAttachments(framebuffer, VK_ATTACHMENT_LOAD_OP_CLEAR);
        framebufferDirty = true;
        dataDirty = true;
    }

protected:
    void loadShader() override {
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
        rasterData->setStaticBuffer(tetMesh->getVertexColorBuffer(), "VertexColorBuffer");
        rasterData->setStaticBuffer(tetMesh->getFaceBoundaryBitBuffer(), "FaceBoundaryBitBuffer");
        rasterData->setNumVertices(numIndexedVertices);
        volumeRenderer->setRenderDataBindings(rasterData);
    }
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
        //pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_BACK);
        pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
        pipelineInfo.setColorWriteEnabled(false);
        pipelineInfo.setDepthWriteEnabled(false);
        pipelineInfo.setDepthTestEnabled(false);
        //pipelineInfo.setVertexBufferBindingByLocationIndex("vertexPosition", sizeof(glm::vec3));
        //pipelineInfo.setVertexBufferBindingByLocationIndex("vertexColor", sizeof(glm::vec4));
    }

private:
    TetMeshVolumeRenderer* volumeRenderer;
};

class ResolveRasterPass : public sgl::vk::BlitRenderPass {
public:
    explicit ResolveRasterPass(TetMeshVolumeRenderer* volumeRenderer)
            : BlitRenderPass(volumeRenderer->getRenderer(), { "LinkedListResolve.Vertex", "LinkedListResolve.Fragment" }),
              volumeRenderer(volumeRenderer) {
        this->setAttachmentLoadOp(VK_ATTACHMENT_LOAD_OP_CLEAR);
    }

protected:
    void loadShader() override {
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("RESOLVE_PASS", ""));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(shaderIds, preprocessorDefines);
    }
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
        rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
        rasterData->setIndexBuffer(indexBuffer);
        rasterData->setVertexBuffer(vertexBuffer, 0);
        volumeRenderer->setRenderDataBindings(rasterData);
    }

private:
    TetMeshVolumeRenderer* volumeRenderer;
};

class ClearRasterPass : public sgl::vk::BlitRenderPass {
public:
    explicit ClearRasterPass(TetMeshVolumeRenderer* volumeRenderer)
            : BlitRenderPass(volumeRenderer->getRenderer(), { "LinkedListClear.Vertex", "LinkedListClear.Fragment" }),
              volumeRenderer(volumeRenderer) {
        this->setAttachmentLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    }

protected:
    void loadShader() override {
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("RESOLVE_PASS", ""));
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(shaderIds, preprocessorDefines);
    }
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override {
        rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
        rasterData->setIndexBuffer(indexBuffer);
        rasterData->setVertexBuffer(vertexBuffer, 0);
        volumeRenderer->setRenderDataBindings(rasterData);
    }

private:
    TetMeshVolumeRenderer* volumeRenderer;
};

TetMeshVolumeRenderer::TetMeshVolumeRenderer(sgl::vk::Renderer* renderer, sgl::CameraPtr* camera)
        : renderer(renderer), camera(camera) {
    sgl::vk::Device* device = renderer->getDevice();
    uniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    maxStorageBufferSize = std::min(
            size_t(device->getMaxMemoryAllocationSize()), size_t(device->getMaxStorageBufferRange()));

    gatherRasterPass = std::make_shared<GatherRasterPass>(this);

    resolveRasterPass = std::make_shared<ResolveRasterPass>(this);
    resolveRasterPass->setOutputImageFinalLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    resolveRasterPass->setBlendMode(sgl::vk::BlendMode::BACK_TO_FRONT_STRAIGHT_ALPHA);

    clearRasterPass = std::make_shared<ClearRasterPass>(this);
    clearRasterPass->setColorWriteEnabled(false);
    clearRasterPass->setAttachmentLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    clearRasterPass->setAttachmentStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE);
    clearRasterPass->setOutputImageInitialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
    clearRasterPass->setOutputImageFinalLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    onClearColorChanged();
}

TetMeshVolumeRenderer::~TetMeshVolumeRenderer() {
}

void TetMeshVolumeRenderer::updateLargeMeshMode() {
    // More than one million cells?
    LargeMeshMode newMeshLargeMeshMode = MESH_SIZE_MEDIUM;
    if (tetMesh->getNumCells() > size_t(1e6)) { // > 1m line cells
        newMeshLargeMeshMode = MESH_SIZE_LARGE;
    }
    if (newMeshLargeMeshMode != largeMeshMode) {
        renderer->getDevice()->waitIdle();
        largeMeshMode = newMeshLargeMeshMode;
        expectedAvgDepthComplexity = MESH_MODE_DEPTH_COMPLEXITIES_PPLL[int(largeMeshMode)][0];
        expectedMaxDepthComplexity = MESH_MODE_DEPTH_COMPLEXITIES_PPLL[int(largeMeshMode)][1];
        if (outputImageView) {
            reallocateFragmentBuffer();
        }
        resolveRasterPass->setShaderDirty();
    }
}

void TetMeshVolumeRenderer::setTetMeshData(const TetMeshPtr& _tetMesh) {
    tetMesh = _tetMesh;
    gatherRasterPass->setDataDirty();
    resolveRasterPass->setDataDirty();
    clearRasterPass->setDataDirty();
    updateLargeMeshMode();
    reRender = true;
}

void TetMeshVolumeRenderer::setOutputImage(sgl::vk::ImageViewPtr& colorImage) {
    outputImageView = colorImage;
}

void TetMeshVolumeRenderer::recreateSwapchain(uint32_t width, uint32_t height) {
    paddedWindowWidth = int(width), paddedWindowHeight = int(height);
    getScreenSizeWithTiling(paddedWindowWidth, paddedWindowHeight);

    reallocateFragmentBuffer();

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

    gatherRasterPass->recreateSwapchain(width, height);

    resolveRasterPass->setOutputImage(outputImageView);
    resolveRasterPass->recreateSwapchain(width, height);

    clearRasterPass->setOutputImage(outputImageView);
    clearRasterPass->recreateSwapchain(width, height);
}

void TetMeshVolumeRenderer::setUseLinearRGB(bool _useLinearRGB) {
}

void TetMeshVolumeRenderer::setClearColor(const sgl::Color& _clearColor) {
    clearColor = _clearColor;
    onClearColorChanged();
}

void TetMeshVolumeRenderer::getVulkanShaderPreprocessorDefines(
        std::map<std::string, std::string>& preprocessorDefines) {
    preprocessorDefines.insert(std::make_pair("MAX_NUM_FRAGS", sgl::toString(expectedMaxDepthComplexity)));
    if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_QUICKSORT
        || sortingAlgorithmMode == SORTING_ALGORITHM_MODE_QUICKSORT_HYBRID) {
        int stackSize = int(std::ceil(std::log2(expectedMaxDepthComplexity)) * 2 + 4);
        preprocessorDefines.insert(std::make_pair("STACK_SIZE", sgl::toString(stackSize)));
    }

    if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_PRIORITY_QUEUE) {
        preprocessorDefines.insert(std::make_pair("sortingAlgorithm", "frontToBackPQ"));
        if (renderer->getDevice()->getDeviceDriverId() == VK_DRIVER_ID_AMD_PROPRIETARY
            || renderer->getDevice()->getDeviceDriverId() == VK_DRIVER_ID_AMD_OPEN_SOURCE) {
            preprocessorDefines.insert(std::make_pair("INITIALIZE_ARRAY_POW2", ""));
        }
    } else if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_BUBBLE_SORT) {
        preprocessorDefines.insert(std::make_pair("sortingAlgorithm", "bubbleSort"));
    } else if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_INSERTION_SORT) {
        preprocessorDefines.insert(std::make_pair("sortingAlgorithm", "insertionSort"));
    } else if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_SHELL_SORT) {
        preprocessorDefines.insert(std::make_pair("sortingAlgorithm", "shellSort"));
    } else if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_MAX_HEAP) {
        preprocessorDefines.insert(std::make_pair("sortingAlgorithm", "heapSort"));
    } else if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_BITONIC_SORT) {
        preprocessorDefines.insert(std::make_pair("sortingAlgorithm", "bitonicSort"));
    } else if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_QUICKSORT) {
        preprocessorDefines.insert(std::make_pair("sortingAlgorithm", "quicksort"));
    } else if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_QUICKSORT_HYBRID) {
        preprocessorDefines.insert(std::make_pair("sortingAlgorithm", "quicksortHybrid"));
    }

    if (sortingAlgorithmMode == SORTING_ALGORITHM_MODE_QUICKSORT
            || sortingAlgorithmMode == SORTING_ALGORITHM_MODE_QUICKSORT_HYBRID) {
        preprocessorDefines.insert(std::make_pair("USE_QUICKSORT", ""));
    }
}

void TetMeshVolumeRenderer::setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData) {
    renderData->setStaticBufferOptional(fragmentBuffer, "FragmentBuffer");
    renderData->setStaticBuffer(startOffsetBuffer, "StartOffsetBuffer");
    renderData->setStaticBufferOptional(fragmentCounterBuffer, "FragCounterBuffer");
    renderData->setStaticBufferOptional(uniformDataBuffer, "UniformDataBuffer");
}

void TetMeshVolumeRenderer::setFramebufferAttachments(sgl::vk::FramebufferPtr& framebuffer, VkAttachmentLoadOp loadOp) {
    sgl::vk::AttachmentState attachmentState;
    attachmentState.loadOp = loadOp;
    attachmentState.initialLayout =
            loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ?
            VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    framebuffer->setColorAttachment(outputImageView, 0, attachmentState, clearColor.getFloatColorRGBA());
}

void TetMeshVolumeRenderer::reallocateFragmentBuffer() {
    int width = int(outputImageView->getImage()->getImageSettings().width);
    int height = int(outputImageView->getImage()->getImageSettings().height);
    int paddedWidth = width, paddedHeight = height;
    getScreenSizeWithTiling(paddedWidth, paddedHeight);

    fragmentBufferSize = size_t(expectedAvgDepthComplexity) * size_t(paddedWidth) * size_t(paddedHeight);
    size_t fragmentBufferSizeBytes = 12ull * fragmentBufferSize;
    if (fragmentBufferSizeBytes > maxStorageBufferSize) {
        sgl::Logfile::get()->writeError(
                std::string() + "Fragment buffer size was larger than maximum allocation size ("
                + std::to_string(maxStorageBufferSize) + "). Clamping to maximum allocation size.",
                false);
        fragmentBufferSize = maxStorageBufferSize / 12ull;
        fragmentBufferSizeBytes = fragmentBufferSize * 12ull;
    } else {
        sgl::Logfile::get()->writeInfo(
                std::string() + "Fragment buffer size GiB: "
                + std::to_string(double(fragmentBufferSizeBytes) / 1024.0 / 1024.0 / 1024.0));
    }

    fragmentBuffer = {}; // Delete old data first (-> refcount 0)
    fragmentBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), fragmentBufferSizeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void TetMeshVolumeRenderer::onClearColorChanged() {
    resolveRasterPass->setAttachmentClearColor(clearColor.getFloatColorRGBA());
}

void TetMeshVolumeRenderer::render() {
    uniformData.linkedListSize = static_cast<uint32_t>(fragmentBufferSize);
    uniformData.viewportW = paddedWindowWidth;
    uniformData.zNear = (*camera)->getNearClipDistance();
    uniformData.zFar = (*camera)->getFarClipDistance();
    uniformData.attenuationCoefficient = attenuationCoefficient;
    uniformDataBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    clear();
    gather();
    resolve();
}

void TetMeshVolumeRenderer::clear() {
    clearRasterPass->render();
    fragmentCounterBuffer->fill(0, renderer->getVkCommandBuffer());
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            startOffsetBuffer);
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            fragmentCounterBuffer);
}

void TetMeshVolumeRenderer::gather() {
    renderer->setProjectionMatrix((*camera)->getProjectionMatrix());
    renderer->setViewMatrix((*camera)->getViewMatrix());
    renderer->setModelMatrix(sgl::matrixIdentity());

    gatherRasterPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void TetMeshVolumeRenderer::resolve() {
    resolveRasterPass->render();
}

void TetMeshVolumeRenderer::renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.addCombo(
            "Sorting Mode", (int*)&sortingAlgorithmMode,
            SORTING_MODE_NAMES, NUM_SORTING_MODES)) {
        resolveRasterPass->setShaderDirty();
        reRender = true;
    }
    if (propertyEditor.addSliderFloat("attenuationCoefficient", &attenuationCoefficient, 1.0f, 1000.0f)) {
        reRender = true;
    }
}

bool TetMeshVolumeRenderer::selectTilingModeUI(sgl::PropertyEditor& propertyEditor) {
    const char* indexingModeNames[] = { "1x1", "2x2", "2x8", "8x2", "4x4", "8x8", "8x8 Morton Code" };
    if (propertyEditor.addCombo(
            "Tiling Mode", (int*)&tilingModeIndex,
            indexingModeNames, IM_ARRAYSIZE(indexingModeNames))) {
        // Select new mode
        if (tilingModeIndex == 0) {
            // No tiling
            tileWidth = 1;
            tileHeight = 1;
        } else if (tilingModeIndex == 1) {
            tileWidth = 2;
            tileHeight = 2;
        } else if (tilingModeIndex == 2) {
            tileWidth = 2;
            tileHeight = 8;
        } else if (tilingModeIndex == 3) {
            tileWidth = 8;
            tileHeight = 2;
        } else if (tilingModeIndex == 4) {
            tileWidth = 4;
            tileHeight = 4;
        } else if (tilingModeIndex == 5) {
            tileWidth = 8;
            tileHeight = 8;
        } else if (tilingModeIndex == 6) {
            tileWidth = 8;
            tileHeight = 8;
        }

        return true;
    }
    return false;
}

void TetMeshVolumeRenderer::setNewTilingMode(int newTileWidth, int newTileHeight, bool useMortonCode /* = false */) {
    tileWidth = newTileWidth;
    tileHeight = newTileHeight;
    tilingUseMortonCode = useMortonCode;

    // Select new mode.
    if (tileWidth == 1 && tileHeight == 1) {
        // No tiling.
        tilingModeIndex = 0;
    } else if (tileWidth == 2 && tileHeight == 2) {
        tilingModeIndex = 1;
    } else if (tileWidth == 2 && tileHeight == 8) {
        tilingModeIndex = 2;
    } else if (tileWidth == 8 && tileHeight == 2) {
        tilingModeIndex = 3;
    } else if (tileWidth == 4 && tileHeight == 4) {
        tilingModeIndex = 4;
    } else if (tileWidth == 8 && tileHeight == 8 && !useMortonCode) {
        tilingModeIndex = 5;
    } else if (tileWidth == 8 && tileHeight == 8 && useMortonCode) {
        tilingModeIndex = 6;
    } else {
        // Invalid mode, just set to mode 5, too.
        tilingModeIndex = 5;
    }
}

void TetMeshVolumeRenderer::getScreenSizeWithTiling(int& screenWidth, int& screenHeight) {
    if (screenWidth % tileWidth != 0) {
        screenWidth = (screenWidth / tileWidth + 1) * tileWidth;
    }
    if (screenHeight % tileHeight != 0) {
        screenHeight = (screenHeight / tileHeight + 1) * tileHeight;
    }
}
