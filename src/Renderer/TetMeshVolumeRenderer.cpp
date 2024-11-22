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

#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Data.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>
#ifndef DISABLE_IMGUI
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/Widgets/NumberFormatting.hpp>
#endif

#include "Tet/TetMesh.hpp"
#include "TetMeshVolumeRenderer.hpp"

TetMeshVolumeRenderer::TetMeshVolumeRenderer(
        sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow)
        : renderer(renderer), camera(camera), transferFunctionWindow(transferFunctionWindow) {
    ;
}

TetMeshVolumeRenderer::~TetMeshVolumeRenderer() {
    ;
}
void TetMeshVolumeRenderer::setCoarseToFineTargetNumTets(uint32_t _coarseToFineMaxNumTets) {
    useCoarseToFine = true;
    coarseToFineMaxNumTets = _coarseToFineMaxNumTets;
}

void TetMeshVolumeRenderer::setTetMeshData(const TetMeshPtr& _tetMesh) {
    tetMesh = _tetMesh;

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

    createDepthComplexityBuffers();
    reallocateFragmentBuffer();
}

void TetMeshVolumeRenderer::recreateSwapchainExternal(
        uint32_t width, uint32_t height, size_t _fragmentBufferSize, const sgl::vk::BufferPtr& _fragmentBuffer,
        const sgl::vk::BufferPtr& _startOffsetBuffer, const sgl::vk::BufferPtr& _fragmentCounterBuffer) {
    windowWidth = int(width);
    windowHeight = int(height);
    paddedWindowWidth = windowWidth, paddedWindowHeight = windowHeight;
    getScreenSizeWithTiling(paddedWindowWidth, paddedWindowHeight);
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
    if (showDepthComplexity) {
        renderData->setStaticBufferOptional(depthComplexityCounterBuffer, "DepthComplexityCounterBuffer");
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

#ifndef DISABLE_IMGUI
void TetMeshVolumeRenderer::renderGuiShared(sgl::PropertyEditor& propertyEditor) {
    auto rendererType = getRendererType();
    /*
     * Currently, quality metrics and shading are not yet supported for projection renderer.
     * - VolumeRendererPassType::RESOLVE is only for PPLL, otherwise triangle generation pass needs to be adapted
     *   for quality metric.
     * - Shading is not easy to implement with the projection pass at all.
     */
    if (propertyEditor.addCheckbox("Use Quality Metric", &showTetQuality)) {
        tetMesh->setTetQualityMetric(tetQualityMetric);
        setShadersDirty(
                rendererType == RendererType::PPLL || rendererType == RendererType::INTERSECTION
                ? VolumeRendererPassType::RESOLVE : VolumeRendererPassType::GATHER);
        reRender = true;
    }
    if (rendererType == RendererType::PPLL) {
        if (showTetQuality && propertyEditor.addCheckbox("Use Shading", &useShading)) {
            setShadersDirty(VolumeRendererPassType::RESOLVE);
            reRender = true;
        }
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
        setShadersDirty(VolumeRendererPassType::ALL);
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
                "Max. Complexity", sgl::toString(maxComplexity) + " / " + getMaxDepthComplexityString());
        renderGuiMemory(propertyEditor);
    }
    propertyEditor.addText("#Tets", std::to_string(tetMesh->getNumCells()));
    propertyEditor.addText("#Vertices", std::to_string(tetMesh->getNumVertices()));
}

void TetMeshVolumeRenderer::renderGuiMemory(sgl::PropertyEditor& propertyEditor) {
    /// We have no limit, except for subclass @see TestMeshRendererPPLL::renderGuiMemory.
    propertyEditor.addText(
            "Fragment throughput",
            sgl::getNiceMemoryString(totalNumFragments * 12ull, 2));
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
#endif

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
