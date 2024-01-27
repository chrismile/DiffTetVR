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
#include <ImGui/Widgets/NumberFormatting.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>
#include <utility>

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
    TetMeshVolumeRenderer* volumeRenderer;
};

class ResolveRasterPass : public sgl::vk::BlitRenderPass {
public:
    explicit ResolveRasterPass(TetMeshVolumeRenderer* volumeRenderer)
            : BlitRenderPass(volumeRenderer->getRenderer(), { "LinkedListResolve.Vertex", "LinkedListResolve.Fragment" }),
              volumeRenderer(volumeRenderer) {
        this->setAttachmentLoadOp(VK_ATTACHMENT_LOAD_OP_CLEAR);
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
        preprocessorDefines.insert(std::make_pair("PI_SQRT", std::to_string(std::sqrt(sgl::PI))));
        preprocessorDefines.insert(std::make_pair("INV_PI_SQRT", std::to_string(1.0f / std::sqrt(sgl::PI))));
        if (volumeRenderer->getShowTetQuality()) {
            preprocessorDefines.insert(std::make_pair("SHOW_TET_QUALITY", ""));
            if (volumeRenderer->getUseShading()) {
                preprocessorDefines.insert(std::make_pair("USE_SHADING", ""));
            }
        }
        if (volumeRenderer->getAlphaMode() == AlphaMode::STRAIGHT) {
            preprocessorDefines.insert(std::make_pair("ALPHA_MODE_STRAIGHT", ""));
        }
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(shaderIds, preprocessorDefines);
    }
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
        rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
        rasterData->setIndexBuffer(indexBuffer);
        rasterData->setVertexBuffer(vertexBuffer, 0);
        const auto& tetMesh = volumeRenderer->getTetMesh();
        rasterData->setStaticBuffer(tetMesh->getTriangleIndexBuffer(), "TriangleIndicesBuffer");
        rasterData->setStaticBuffer(tetMesh->getVertexPositionBuffer(), "VertexPositionBuffer");
        if (volumeRenderer->getShowTetQuality()) {
            rasterData->setStaticBuffer(tetMesh->getFaceToTetMapBuffer(), "FaceToTetMapBuffer");
            rasterData->setStaticBuffer(tetMesh->getTetQualityBuffer(), "TetQualityBuffer");
            rasterData->setStaticTexture(
                    volumeRenderer->getTransferFunctionWindow()->getTransferFunctionMapTextureVulkan(),
                    "transferFunctionTexture");
            rasterData->setStaticBuffer(
                    volumeRenderer->getTransferFunctionWindow()->getMinMaxUboVulkan(),
                    "MinMaxUniformBuffer");
        } else {
            rasterData->setStaticBuffer(tetMesh->getVertexColorBuffer(), "VertexColorBuffer");
        }
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

class AdjointRasterPass : public sgl::vk::BlitRenderPass {
public:
    explicit AdjointRasterPass(TetMeshVolumeRenderer* volumeRenderer)
            : BlitRenderPass(volumeRenderer->getRenderer(), { "LinkedListResolve.Vertex", "LinkedListResolve.Fragment" }),
              volumeRenderer(volumeRenderer) {
        this->setAttachmentLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);
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
        preprocessorDefines.insert(std::make_pair("BACKWARD_PASS", ""));
        preprocessorDefines.insert(std::make_pair("RESOLVE_PASS", ""));
        preprocessorDefines.insert(std::make_pair("PI_SQRT", std::to_string(std::sqrt(sgl::PI))));
        preprocessorDefines.insert(std::make_pair("INV_PI_SQRT", std::to_string(1.0f / std::sqrt(sgl::PI))));
        if (renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat32AtomicAdd) {
            preprocessorDefines.insert(std::make_pair("SUPPORT_BUFFER_FLOAT_ATOMIC_ADD", ""));
            preprocessorDefines.insert(std::make_pair(
                    "__extensions", "GL_EXT_shader_atomic_float;GL_EXT_control_flow_attributes"));
        } else {
            preprocessorDefines.insert(std::make_pair("__extensions", "GL_EXT_control_flow_attributes"));
        }
        volumeRenderer->getVulkanShaderPreprocessorDefines(preprocessorDefines);
        shaderStages = sgl::vk::ShaderManager->getShaderStages(shaderIds, preprocessorDefines);
    }
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
        rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
        rasterData->setIndexBuffer(indexBuffer);
        rasterData->setVertexBuffer(vertexBuffer, 0);
        const auto& tetMesh = volumeRenderer->getTetMesh();
        rasterData->setStaticBuffer(tetMesh->getTriangleIndexBuffer(), "TriangleIndicesBuffer");
        rasterData->setStaticBuffer(tetMesh->getVertexPositionBuffer(), "VertexPositionBuffer");
        rasterData->setStaticBuffer(tetMesh->getVertexColorBuffer(), "VertexColorBuffer");
        volumeRenderer->setRenderDataBindings(rasterData);
    }

private:
    TetMeshVolumeRenderer* volumeRenderer;
};

TetMeshVolumeRenderer::TetMeshVolumeRenderer(
        sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow)
        : renderer(renderer), camera(camera), transferFunctionWindow(transferFunctionWindow) {
    sgl::vk::Device* device = renderer->getDevice();
    uniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    maxStorageBufferSize = std::min(
            size_t(device->getMaxMemoryAllocationSize()), size_t(device->getMaxStorageBufferRange()));

    auto memoryHeapIndex = uint32_t(device->findMemoryHeapIndex(VK_MEMORY_HEAP_DEVICE_LOCAL_BIT));
    size_t availableVram = device->getMemoryHeapBudgetVma(memoryHeapIndex);
    double availableMemoryFactor = 28.0 / 32.0;
    maxDeviceMemoryBudget = size_t(double(availableVram) * availableMemoryFactor);

    gatherRasterPass = std::make_shared<GatherRasterPass>(this);

    resolveRasterPass = std::make_shared<ResolveRasterPass>(this);
    resolveRasterPass->setOutputImageFinalLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    resolveRasterPass->setBlendMode(sgl::vk::BlendMode::BACK_TO_FRONT_PREMUL_ALPHA);

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

void TetMeshVolumeRenderer::onTransferFunctionMapRebuilt() {
    ;
}

void TetMeshVolumeRenderer::updateLargeMeshMode() {
    // More than one million cells?
    LargeMeshMode newMeshLargeMeshMode = MESH_SIZE_SMALL;
    if (tetMesh->getNumCells() > size_t(1e6)) { // > 1m cells
        newMeshLargeMeshMode = MESH_SIZE_LARGE;
    } else if (tetMesh->getNumCells() > size_t(1e5)) { // > 1m cells
        newMeshLargeMeshMode = MESH_SIZE_MEDIUM;
    }
    if (newMeshLargeMeshMode != largeMeshMode) {
        renderer->getDevice()->waitIdle();
        largeMeshMode = newMeshLargeMeshMode;
        expectedAvgDepthComplexity = MESH_MODE_DEPTH_COMPLEXITIES_PPLL[int(largeMeshMode)][0];
        expectedMaxDepthComplexity = MESH_MODE_DEPTH_COMPLEXITIES_PPLL[int(largeMeshMode)][1];
        if (outputImageView && !useExternalFragmentBuffer) {
            reallocateFragmentBuffer();
        }
        resolveRasterPass->setShaderDirty();
        gatherRasterPass->setDataDirty();
        if (adjointRasterPass) {
            adjointRasterPass->setShaderDirty();
        }
    }
}

void TetMeshVolumeRenderer::setTetMeshData(const TetMeshPtr& _tetMesh) {
    tetMesh = _tetMesh;
    gatherRasterPass->setDataDirty();
    resolveRasterPass->setDataDirty();
    clearRasterPass->setDataDirty();
    updateLargeMeshMode();

    if (showTetQuality) {
        tetMesh->setTetQualityMetric(tetQualityMetric);
    }

    statisticsUpToDate = false;
    counterPrintFrags = 0.0f;
    firstFrame = true;
    totalNumFragments = 0;
    usedLocations = 1;
    maxComplexity = 0;
    bufferSize = 1;

    reRender = true;
}

void TetMeshVolumeRenderer::setOutputImage(sgl::vk::ImageViewPtr& colorImage) {
    outputImageView = colorImage;
}

void TetMeshVolumeRenderer::setAdjointPassData(
        sgl::vk::ImageViewPtr _colorAdjointImage, sgl::vk::ImageViewPtr _adjointPassBackbuffer,
        sgl::vk::BufferPtr _vertexPositionGradientBuffer, sgl::vk::BufferPtr _vertexColorGradientBuffer) {
    if (!adjointRasterPass) {
        adjointRasterPass = std::make_shared<AdjointRasterPass>(this);
        adjointRasterPass->setColorWriteEnabled(false);
        adjointRasterPass->setAttachmentLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);
        adjointRasterPass->setAttachmentStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE);
        adjointRasterPass->setOutputImageInitialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        adjointRasterPass->setOutputImageFinalLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        adjointRasterPass->setBlendMode(sgl::vk::BlendMode::OVERWRITE);
    }
    colorAdjointImage = std::move(_colorAdjointImage);
    adjointPassBackbuffer = std::move(_adjointPassBackbuffer);
    vertexPositionGradientBuffer = std::move(_vertexPositionGradientBuffer);
    vertexColorGradientBuffer = std::move(_vertexColorGradientBuffer);
}

void TetMeshVolumeRenderer::recreateSwapchain(uint32_t width, uint32_t height) {
    windowWidth = int(width);
    windowHeight = int(height);
    paddedWindowWidth = windowWidth, paddedWindowHeight = windowHeight;
    getScreenSizeWithTiling(paddedWindowWidth, paddedWindowHeight);
    useExternalFragmentBuffer = false;

    createDepthComplexityBuffers();
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

    if (adjointRasterPass) {
        adjointRasterPass->setOutputImage(adjointPassBackbuffer);
        adjointRasterPass->recreateSwapchain(width, height);
    }
}

void TetMeshVolumeRenderer::recreateSwapchainExternal(
        uint32_t width, uint32_t height, size_t _fragmentBufferSize, sgl::vk::BufferPtr _fragmentBuffer,
        sgl::vk::BufferPtr _startOffsetBuffer, sgl::vk::BufferPtr _fragmentCounterBuffer) {
    windowWidth = int(width);
    windowHeight = int(height);
    paddedWindowWidth = windowWidth, paddedWindowHeight = windowHeight;
    getScreenSizeWithTiling(paddedWindowWidth, paddedWindowHeight);
    useExternalFragmentBuffer = true;

    fragmentBufferMode = FragmentBufferMode::BUFFER;
    numFragmentBuffers = 1;
    cachedNumFragmentBuffers = 1;
    fragmentBuffers = {};
    fragmentBufferReferenceBuffer = {};
    fragmentBufferSize = _fragmentBufferSize;
    fragmentBuffer = std::move(_fragmentBuffer);
    startOffsetBuffer = std::move(_startOffsetBuffer);
    fragmentCounterBuffer = std::move(_fragmentCounterBuffer);

    gatherRasterPass->recreateSwapchain(width, height);

    resolveRasterPass->setOutputImage(outputImageView);
    resolveRasterPass->recreateSwapchain(width, height);

    clearRasterPass->setOutputImage(outputImageView);
    clearRasterPass->recreateSwapchain(width, height);

    if (adjointRasterPass) {
        adjointRasterPass->setOutputImage(adjointPassBackbuffer);
        adjointRasterPass->recreateSwapchain(width, height);
    }
}

void TetMeshVolumeRenderer::setUseLinearRGB(bool _useLinearRGB) {
}

void TetMeshVolumeRenderer::setClearColor(const sgl::Color& _clearColor) {
    clearColor = _clearColor;
    onClearColorChanged();
}

void TetMeshVolumeRenderer::getVulkanShaderPreprocessorDefines(
        std::map<std::string, std::string>& preprocessorDefines) {
    if (showDepthComplexity) {
        preprocessorDefines.insert(std::make_pair("SHOW_DEPTH_COMPLEXITY", ""));
    }

    if (fragmentBufferMode == FragmentBufferMode::BUFFER_REFERENCE_ARRAY) {
        preprocessorDefines.insert(std::make_pair("FRAGMENT_BUFFER_REFERENCE_ARRAY", ""));
    } else if (fragmentBufferMode == FragmentBufferMode::BUFFER_ARRAY) {
        preprocessorDefines.insert(std::make_pair("FRAGMENT_BUFFER_ARRAY", ""));
    }
    if (fragmentBufferMode == FragmentBufferMode::BUFFER_ARRAY
            || fragmentBufferMode == FragmentBufferMode::BUFFER_REFERENCE_ARRAY) {
        preprocessorDefines.insert(std::make_pair("NUM_FRAGMENT_BUFFERS", std::to_string(cachedNumFragmentBuffers)));
        preprocessorDefines.insert(std::make_pair("NUM_FRAGS_PER_BUFFER", std::to_string(maxStorageBufferSize / 12ull) + "u"));
        auto it = preprocessorDefines.find("__extensions");
        std::string extensionString;
        if (it != preprocessorDefines.end()) {
            extensionString = it->second + ";";
        }
        if (fragmentBufferMode == FragmentBufferMode::BUFFER_ARRAY) {
            extensionString += "GL_EXT_nonuniform_qualifier";
        } else if (fragmentBufferMode == FragmentBufferMode::BUFFER_REFERENCE_ARRAY) {
            extensionString += "GL_EXT_shader_explicit_arithmetic_types_int64;GL_EXT_buffer_reference";
        }
        preprocessorDefines["__extensions"] = extensionString;
    }

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

    if (tileWidth == 1 && tileHeight == 1) {
        // No tiling
        tilingModeIndex = 0;
    } else if (tileWidth == 2 && tileHeight == 2) {
        tilingModeIndex = 1;
        preprocessorDefines.insert(std::make_pair("ADDRESSING_TILED_2x2", ""));
    } else if (tileWidth == 2 && tileHeight == 8) {
        tilingModeIndex = 2;
        preprocessorDefines.insert(std::make_pair("ADDRESSING_TILED_2x8", ""));
    } else if (tileWidth == 8 && tileHeight == 2) {
        tilingModeIndex = 3;
        preprocessorDefines.insert(std::make_pair("ADDRESSING_TILED_NxM", ""));
    } else if (tileWidth == 4 && tileHeight == 4) {
        tilingModeIndex = 4;
        preprocessorDefines.insert(std::make_pair("ADDRESSING_TILED_NxM", ""));
    } else if (tileWidth == 8 && tileHeight == 8 && !tilingUseMortonCode) {
        tilingModeIndex = 5;
        preprocessorDefines.insert(std::make_pair("ADDRESSING_TILED_NxM", ""));
    } else if (tileWidth == 8 && tileHeight == 8 && tilingUseMortonCode) {
        tilingModeIndex = 6;
        preprocessorDefines.insert(std::make_pair("ADRESSING_MORTON_CODE_8x8", ""));
    } else {
        // Invalid mode, just set to mode 5, too.
        tilingModeIndex = 5;
        preprocessorDefines.insert(std::make_pair("ADDRESSING_TILED_NxM", ""));
    }

    preprocessorDefines.insert(std::make_pair("TILE_N", sgl::toString(tileWidth)));
    preprocessorDefines.insert(std::make_pair("TILE_M", sgl::toString(tileHeight)));
}

void TetMeshVolumeRenderer::setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData) {
    if (fragmentBufferMode == FragmentBufferMode::BUFFER) {
        renderData->setStaticBufferOptional(fragmentBuffer, "FragmentBuffer");
    } else if (fragmentBufferMode == FragmentBufferMode::BUFFER_ARRAY) {
        renderData->setStaticBufferArrayOptional(fragmentBuffers, "FragmentBuffer");
    } else {
        renderData->setStaticBufferOptional(fragmentBufferReferenceBuffer, "FragmentBuffer");
    }
    renderData->setStaticBuffer(startOffsetBuffer, "StartOffsetBuffer");
    renderData->setStaticBufferOptional(fragmentCounterBuffer, "FragCounterBuffer");
    renderData->setStaticBufferOptional(uniformDataBuffer, "UniformDataBuffer");

    if (showDepthComplexity) {
        renderData->setStaticBuffer(depthComplexityCounterBuffer, "DepthComplexityCounterBuffer");
    }

    // For resolve pass.
    renderData->setStaticBufferOptional(vertexPositionGradientBuffer, "VertexPositionGradientBuffer");
    renderData->setStaticBufferOptional(vertexColorGradientBuffer, "VertexColorGradientBuffer");
    renderData->setStaticImageViewOptional(outputImageView, "colorImageOpt");
    renderData->setStaticImageViewOptional(colorAdjointImage, "adjointColors");
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

    // Delete old data first (-> refcount 0)
    fragmentBuffers = {};
    fragmentBuffer = {};
    fragmentBufferReferenceBuffer = {};
    resolveRasterPass->clearFragmentBuffer();
    gatherRasterPass->clearFragmentBuffer();
    clearRasterPass->clearFragmentBuffer();
    if (adjointRasterPass) {
        adjointRasterPass->clearFragmentBuffer();
    }

    // We only need buffer arrays when the maximum allocation is larger than our budget.
    if (maxDeviceMemoryBudget < maxStorageBufferSize) {
        fragmentBufferMode = FragmentBufferMode::BUFFER;
        resolveRasterPass->setShaderDirty();
        gatherRasterPass->setShaderDirty();
        clearRasterPass->setShaderDirty();
        if (adjointRasterPass) {
            adjointRasterPass->setShaderDirty();
        }
    }
    size_t maxSingleBufferAllocation = std::min(maxDeviceMemoryBudget, maxStorageBufferSize);

    if (fragmentBufferMode == FragmentBufferMode::BUFFER) {
        if (fragmentBufferSizeBytes > maxSingleBufferAllocation) {
            sgl::Logfile::get()->writeError(
                    std::string() + "Fragment buffer size was larger than maximum allocation size ("
                    + std::to_string(maxSingleBufferAllocation) + "). Clamping to maximum allocation size.",
                    false);
            fragmentBufferSize = maxSingleBufferAllocation / 12ull;
            fragmentBufferSizeBytes = fragmentBufferSize * 12ull;
        } else {
            sgl::Logfile::get()->writeInfo(
                    std::string() + "Fragment buffer size GiB: "
                    + std::to_string(double(fragmentBufferSizeBytes) / 1024.0 / 1024.0 / 1024.0));
        }

        numFragmentBuffers = 1;
        cachedNumFragmentBuffers = 1;
        fragmentBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), fragmentBufferSizeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
    } else {
        if (fragmentBufferSizeBytes > maxDeviceMemoryBudget) {
            sgl::Logfile::get()->writeError(
                    std::string() + "Fragment buffer size was larger than maximum allocation size ("
                    + std::to_string(maxDeviceMemoryBudget) + "). Clamping to maximum allocation size.",
                    false);
            fragmentBufferSize = maxDeviceMemoryBudget / 12ull;
            fragmentBufferSizeBytes = fragmentBufferSize * 12ull;
        } else {
            sgl::Logfile::get()->writeInfo(
                    std::string() + "Fragment buffer size GiB: "
                    + std::to_string(double(fragmentBufferSizeBytes) / 1024.0 / 1024.0 / 1024.0));
        }

        numFragmentBuffers = sgl::sizeceil(fragmentBufferSizeBytes, maxStorageBufferSize);
        size_t fragmentBufferSizeBytesLeft = fragmentBufferSizeBytes;
        for (size_t i = 0; i < numFragmentBuffers; i++) {
            VkBufferUsageFlags flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            if (fragmentBufferMode == FragmentBufferMode::BUFFER_REFERENCE_ARRAY) {
                flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            }
            fragmentBuffers.emplace_back(std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), std::min(fragmentBufferSizeBytesLeft, maxStorageBufferSize),
                    flags, VMA_MEMORY_USAGE_GPU_ONLY));
            fragmentBufferSizeBytesLeft -= maxStorageBufferSize;
        }

        if (numFragmentBuffers != cachedNumFragmentBuffers) {
            cachedNumFragmentBuffers = numFragmentBuffers;
            resolveRasterPass->setShaderDirty();
            gatherRasterPass->setShaderDirty();
            clearRasterPass->setShaderDirty();
            if (adjointRasterPass) {
                adjointRasterPass->setShaderDirty();
            }
        }

        if (fragmentBufferMode == FragmentBufferMode::BUFFER_REFERENCE_ARRAY) {
            auto* bufferReferences = new uint64_t[numFragmentBuffers];
            for (size_t i = 0; i < numFragmentBuffers; i++) {
                bufferReferences[i] = fragmentBuffers.at(i)->getVkDeviceAddress();
            }
            fragmentBufferReferenceBuffer = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), sizeof(uint64_t) * numFragmentBuffers,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY);
            fragmentBufferReferenceBuffer->updateData(
                    sizeof(uint64_t) * numFragmentBuffers, bufferReferences, renderer->getVkCommandBuffer());
            renderer->insertBufferMemoryBarrier(
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    fragmentBufferReferenceBuffer);
            delete[] bufferReferences;
        }
    }
}

void TetMeshVolumeRenderer::onClearColorChanged() {
    resolveRasterPass->setAttachmentClearColor(clearColor.getFloatColorRGBA());
    if ((clearColor.getA() == 0) != (alphaMode == AlphaMode::STRAIGHT)) {
        resolveRasterPass->setShaderDirty();
        alphaMode = clearColor.getA() == 0 ? AlphaMode::STRAIGHT : AlphaMode::PREMUL;
        resolveRasterPass->setBlendMode(
                alphaMode == AlphaMode::PREMUL
                ? sgl::vk::BlendMode::BACK_TO_FRONT_PREMUL_ALPHA : sgl::vk::BlendMode::OVERWRITE);
    }
}

//#define USE_ORTHO_PROJ

void TetMeshVolumeRenderer::render() {
    auto imageSettings = outputImageView->getImage()->getImageSettings();
#ifdef USE_ORTHO_PROJ
    glm::mat4 projMatTest = (*camera)->getProjectionMatrix();
    projMat = glm::orthoRH_ZO(
            -0.5f, 0.5f, -0.5f, 0.5f,
            (*camera)->getNearClipDistance(), (*camera)->getFarClipDistance());
    //projMat = glm::perspectiveRH_ZO(
    //        (*camera)->getFOVy(), (*camera)->getAspectRatio(),
    //        (*camera)->getNearClipDistance(), (*camera)->getFarClipDistance());
    projMat[1][1] = -projMat[1][1]; // coordinateOrigin == CoordinateOrigin::TOP_LEFT
    //glm::mat4 viewProjMat = projMat * (*camera)->getViewMatrix();
    //projMat = (*camera)->getProjectionMatrix();
    //glm::vec4 p(0.1f, 0.1f, -0.2f, 1.0f);
    //glm::vec4 pt0 = projMatTest * p;
    //glm::vec4 pt1 = projMat * p;
    glm::mat4 inverseViewProjMat = glm::inverse(projMat * (*camera)->getViewMatrix());
    uniformData.inverseViewProjectionMatrix = inverseViewProjMat;
#else
    uniformData.inverseViewProjectionMatrix = (*camera)->getInverseViewProjMatrix();
#endif

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

    clear();
    gather();
    resolve();
}

void TetMeshVolumeRenderer::renderAdjoint() {
    adjointRasterPass->render();
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
    if (showDepthComplexity) {
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                depthComplexityCounterBuffer);
    }
}

void TetMeshVolumeRenderer::gather() {
#ifdef USE_ORTHO_PROJ
    renderer->setProjectionMatrix(projMat);
#else
    renderer->setProjectionMatrix((*camera)->getProjectionMatrix());
#endif
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
    if (propertyEditor.addCombo(
            "Fragment Buffer Mode", (int*)&fragmentBufferMode,
            FRAGMENT_BUFFER_MODE_NAMES, IM_ARRAYSIZE(FRAGMENT_BUFFER_MODE_NAMES))) {
        renderer->syncWithCpu();
        renderer->getDevice()->waitIdle();
        resolveRasterPass->setShaderDirty();
        gatherRasterPass->setShaderDirty();
        clearRasterPass->setShaderDirty();
        if (adjointRasterPass) {
            adjointRasterPass->setShaderDirty();
        }
        reallocateFragmentBuffer();
        reRender = true;
    }
    if (propertyEditor.addCheckbox("Use Quality Metric", &showTetQuality)) {
        tetMesh->setTetQualityMetric(tetQualityMetric);
        resolveRasterPass->setShaderDirty();
        reRender = true;
    }
    if (showTetQuality && propertyEditor.addCheckbox("Use Shading", &useShading)) {
        resolveRasterPass->setShaderDirty();
        reRender = true;
    }
    if (showTetQuality && propertyEditor.addCombo(
            "Tet Quality Metric", (int*)&tetQualityMetric,
            TET_QUALITY_METRIC_NAMES, IM_ARRAYSIZE(TET_QUALITY_METRIC_NAMES))) {
        tetMesh->setTetQualityMetric(tetQualityMetric);
        reRender = true;
    }
    if (propertyEditor.addSliderFloat("Attenuation", &attenuationCoefficient, 1.0f, 1000.0f)) {
        reRender = true;
    }
    bool depthComplexityJustChanged = false;
    if (propertyEditor.addCheckbox("Show Depth Complexity", &showDepthComplexity)) {
        resolveRasterPass->setShaderDirty();
        gatherRasterPass->setShaderDirty();
        clearRasterPass->setShaderDirty();
        if (adjointRasterPass) {
            adjointRasterPass->setShaderDirty();
        }
        reRender = true;
        depthComplexityJustChanged = true;
        createDepthComplexityBuffers();
    }
    if (showDepthComplexity && !depthComplexityJustChanged) {
        std::string totalNumFragmentsString = sgl::numberToCommaString(int64_t(totalNumFragments));
        propertyEditor.addText("#Fragments", totalNumFragmentsString);
        propertyEditor.addText(
                "Average Used",
                sgl::toString(double(totalNumFragments) / double(usedLocations), 2));
        propertyEditor.addText(
                "Average All",
                sgl::toString(double(totalNumFragments) / double(bufferSize), 2));
        propertyEditor.addText(
                "Max. Complexity", sgl::toString(maxComplexity) + " / " + sgl::toString(expectedMaxDepthComplexity));
        propertyEditor.addText(
                "Memory",
                sgl::getNiceMemoryString(totalNumFragments * 12ull, 2) + " / "
                + sgl::getNiceMemoryString(fragmentBufferSize * 12ull, 2));
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
