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

#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Data.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>
#ifndef DISABLE_IMGUI
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/Widgets/NumberFormatting.hpp>
#endif

#ifdef BUILD_PYTHON_MODULE
#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif
#ifdef SUPPORT_HIP_INTEROP
#include <Graphics/Vulkan/Utils/InteropHIP.hpp>
#endif
#ifdef SUPPORT_SYCL_INTEROP
#include <Graphics/Vulkan/Utils/InteropLevelZero.hpp>
#endif
#ifdef SUPPORT_CUDA_INTEROP
#include <c10/cuda/CUDAStream.h>
#endif
#ifdef SUPPORT_HIP_INTEROP
#include <c10/hip/HIPStream.h>
#endif
#ifdef SUPPORT_SYCL_INTEROP
#include <c10/xpu/XPUStream.h>
#include <torch/csrc/api/include/torch/xpu.h>
#endif
#endif

#include "Tet/TetMesh.hpp"
#include "TetMeshVolumeRenderer.hpp"

TetMeshVolumeRenderer::TetMeshVolumeRenderer(
        sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow)
        : renderer(renderer), camera(camera), transferFunctionWindow(transferFunctionWindow) {
    clipPlaneDataBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), sizeof(ClipPlaneData), &clipPlaneData,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

TetMeshVolumeRenderer::~TetMeshVolumeRenderer() {
    renderer->getDevice()->waitIdle();
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
    checkRecreateTerminationIndexImage();
}

void TetMeshVolumeRenderer::setAdjointPassData(
        sgl::vk::ImageViewPtr _colorAdjointImage, sgl::vk::ImageViewPtr _adjointPassBackbuffer) {
    colorAdjointImage = std::move(_colorAdjointImage);
    adjointPassBackbuffer = std::move(_adjointPassBackbuffer);
}

void TetMeshVolumeRenderer::checkRecreateTerminationIndexImage() {
    if (!useEarlyRayTermination) {
        terminationIndexImageView = {};
        return;
    }
    if (!outputImageView) {
        return;
    }
    bool recreateImage = false;
    const auto& imageSettingsColor = outputImageView->getImage()->getImageSettings();
    if (!terminationIndexImageView) {
        recreateImage = true;
    } else {
        const auto& imageSettingsTIdx = terminationIndexImageView->getImage()->getImageSettings();
        if (imageSettingsColor.width != imageSettingsTIdx.width
                || imageSettingsColor.height != imageSettingsTIdx.height) {
            recreateImage = true;
        }
    }
    if (recreateImage) {
        sgl::vk::ImageSettings imageSettings{};
        imageSettings.width = outputImageView->getImage()->getImageSettings().width;
        imageSettings.height = outputImageView->getImage()->getImageSettings().height;
        imageSettings.format = VK_FORMAT_R32_UINT;
        imageSettings.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;
        sgl::vk::Device* device = renderer->getDevice();
        terminationIndexImageView = {};
        terminationIndexImageView = std::make_shared<sgl::vk::ImageView>(std::make_shared<sgl::vk::Image>(
                device, imageSettings));
        if (renderer->getIsCommandBufferInRecordingState()) {
            renderer->transitionImageLayout(terminationIndexImageView, VK_IMAGE_LAYOUT_GENERAL);
        } else {
            auto commandBuffer = device->beginSingleTimeCommands();
            terminationIndexImageView->transitionImageLayout(VK_IMAGE_LAYOUT_GENERAL, commandBuffer);
            device->endSingleTimeCommands(commandBuffer);
        }
    }
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

#ifdef BUILD_PYTHON_MODULE
void TetMeshVolumeRenderer::setUseComputeInterop(bool _useComputeInterop) {
    useComputeInterop = _useComputeInterop;
}

void TetMeshVolumeRenderer::setUsedDeviceType(torch::DeviceType _usedDeviceType) {
    usedDeviceType = _usedDeviceType;
}

void TetMeshVolumeRenderer::setViewportSize(uint32_t viewportWidth, uint32_t viewportHeight) {
    outputImageView = {};
    colorAdjointImage = {};
    adjointPassBackbuffer = {};
#ifdef SUPPORT_COMPUTE_INTEROP
    colorImageBuffer = {};
    colorAdjointImageBuffer = {};
    colorImageBufferCu = {};
    colorAdjointImageBufferCu = {};
#endif
    colorImageBufferCpu = {};
    colorAdjointImageBufferCpu = {};
    colorImageBufferCpuPtr = nullptr;
    colorAdjointImageBufferCpuPtr = nullptr;

    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = viewportWidth;
    imageSettings.height = viewportHeight;
    imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageSettings.usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;

    sgl::vk::Device* device = renderer->getDevice();
    outputImageView = std::make_shared<sgl::vk::ImageView>(std::make_shared<sgl::vk::Image>(device, imageSettings));
    checkRecreateTerminationIndexImage();

    const size_t imageSize = viewportWidth * viewportHeight * sizeof(float) * 4;

#ifdef SUPPORT_COMPUTE_INTEROP
    if (useComputeInterop) {
        colorImageBuffer = std::make_shared<sgl::vk::Buffer>(
                device, imageSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY, true, true);
        colorImageBufferCu = std::make_shared<sgl::vk::BufferComputeApiExternalMemoryVk>(colorImageBuffer);
    } else {
#endif
        colorImageBufferCpu = std::make_shared<sgl::vk::Buffer>(
                device, imageSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        colorImageBufferCpuPtr = colorImageBufferCpu->mapMemory();
#ifdef SUPPORT_COMPUTE_INTEROP
    }
#endif

    if (tetMesh && tetMesh->getUseGradients()) {
        imageSettings.usage =
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        colorAdjointImage = std::make_shared<sgl::vk::ImageView>(
                std::make_shared<sgl::vk::Image>(device, imageSettings));
        renderer->insertImageMemoryBarrier(
                colorAdjointImage->getImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        imageSettings.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
        adjointPassBackbuffer = std::make_shared<sgl::vk::ImageView>(std::make_shared<sgl::vk::Image>(
                device, imageSettings));

#ifdef SUPPORT_COMPUTE_INTEROP
        if (useComputeInterop) {
            colorAdjointImageBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, imageSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY, true, true);
            colorAdjointImageBufferCu = std::make_shared<sgl::vk::BufferComputeApiExternalMemoryVk>(
                    colorAdjointImageBuffer);
        } else {
#endif
            colorAdjointImageBufferCpu = std::make_shared<sgl::vk::Buffer>(
                    device, imageSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            colorAdjointImageBufferCpuPtr = colorAdjointImageBufferCpu->mapMemory();
#ifdef SUPPORT_COMPUTE_INTEROP
        }
#endif

        setAdjointPassData(colorAdjointImage, adjointPassBackbuffer);
    }
}

torch::Tensor TetMeshVolumeRenderer::getImageTensor() {
    auto imageWidth = outputImageView->getImage()->getImageSettings().width;
    auto imageHeight = outputImageView->getImage()->getImageSettings().height;
    bool useGradients = tetMesh->getUseGradients();

#ifdef SUPPORT_COMPUTE_INTEROP
    if (useComputeInterop) {
        torch::Tensor imageTensor = torch::from_blob(
                colorImageBufferCu->getDevicePtr<float>(),
                { int(imageHeight), int(imageWidth), int(4) },
                torch::TensorOptions().dtype(torch::kFloat32).device(usedDeviceType).requires_grad(useGradients));
        if (useGradients) {
            torch::Tensor imageAdjointTensor = torch::from_blob(
                    colorAdjointImageBufferCu->getDevicePtr<float>(),
                    { int(imageHeight), int(imageWidth), int(4) },
                    torch::TensorOptions().dtype(torch::kFloat32).device(usedDeviceType));
            imageTensor.mutable_grad() = imageAdjointTensor;
        }
        return imageTensor;
    } else {
#endif
        torch::Tensor imageTensor = torch::from_blob(
                colorImageBufferCpuPtr,
                { int(imageHeight), int(imageWidth), int(4) },
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).requires_grad(useGradients));
        if (useGradients) {
            torch::Tensor imageAdjointTensor = torch::from_blob(
                    colorAdjointImageBufferCpuPtr,
                    { int(imageHeight), int(imageWidth), int(4) },
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
            imageTensor.mutable_grad() = imageAdjointTensor;
        }
        return imageTensor;
#ifdef SUPPORT_COMPUTE_INTEROP
    }
#endif
}

void TetMeshVolumeRenderer::copyOutputImageToBuffer() {
    renderer->transitionImageLayout(outputImageView, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
#ifdef SUPPORT_COMPUTE_INTEROP
    if (useComputeInterop) {
        outputImageView->getImage()->copyToBuffer(colorImageBuffer, renderer->getVkCommandBuffer());
    } else {
#endif
        outputImageView->getImage()->copyToBuffer(colorImageBufferCpu, renderer->getVkCommandBuffer());
#ifdef SUPPORT_COMPUTE_INTEROP
    }
#endif
    renderer->transitionImageLayout(outputImageView, VK_IMAGE_LAYOUT_GENERAL);
}

void TetMeshVolumeRenderer::copyAdjointBufferToImagePreCheck(void* devicePtr) {
#ifdef SUPPORT_COMPUTE_INTEROP
    if (useComputeInterop) {
        if (devicePtr != colorAdjointImageBufferCu->getDevicePtr<void>()) {
            //sgl::Logfile::get()->writeError(
            //        "Error in TetMeshVolumeRenderer::copyAdjointBufferToImagePreCheck: "
            //        "Mismatch in internal adjoint buffer device address and tensor content.", false);
            //cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            //CUresult cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
            //        colorAdjointImageBufferCu->getCudaDevicePtr(), reinterpret_cast<CUdeviceptr>(devicePtr),
            //        colorAdjointImageBuffer->getSizeInBytes(), stream);
            //sgl::vk::checkCUresult(cuResult, "Error in cuMemcpyAsync: ");
            sgl::vk::StreamWrapper stream{};
#ifdef SUPPORT_CUDA_INTEROP
            if (usedDeviceType == torch::DeviceType::CUDA) {
                stream.cuStream = at::cuda::getCurrentCUDAStream();
            }
#endif
#ifdef SUPPORT_HIP_INTEROP
            if (usedDeviceType == torch::DeviceType::HIP) {
                stream.hipStream = at::hip::getCurrentHIPStream();
            }
#endif
#ifdef SUPPORT_SYCL_INTEROP
            if (usedDeviceType == torch::DeviceType::XPU) {
                auto& syclQueue = at::xpu::getCurrentXPUStream().queue();
                sgl::vk::setLevelZeroGlobalStateFromSyclQueue(syclQueue);
                stream.zeCommandList = sgl::vk::syclStreamToZeCommandList(syclQueue);
            }
#endif
            colorAdjointImageBufferCu->copyFromDevicePtrAsync(devicePtr, stream);
        }
    } else {
#endif
        if (devicePtr != colorAdjointImageBufferCpuPtr) {
            colorAdjointImageBufferCpu->copyHostMemoryToAllocation(devicePtr);
        }
#ifdef SUPPORT_COMPUTE_INTEROP
    }
#endif
}

void TetMeshVolumeRenderer::copyAdjointBufferToImage() {
    renderer->insertImageMemoryBarrier(
            colorAdjointImage->getImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
#ifdef SUPPORT_COMPUTE_INTEROP
    if (useComputeInterop) {
        colorAdjointImage->getImage()->copyFromBuffer(colorAdjointImageBuffer, renderer->getVkCommandBuffer());
    } else {
#endif
        colorAdjointImage->getImage()->copyFromBuffer(colorAdjointImageBufferCpu, renderer->getVkCommandBuffer());
#ifdef SUPPORT_COMPUTE_INTEROP
    }
#endif
    renderer->insertImageMemoryBarrier(
            colorAdjointImage->getImage(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}
#endif

void TetMeshVolumeRenderer::setUseLinearRGB(bool _useLinearRGB) {
}

void TetMeshVolumeRenderer::setClearColor(const sgl::Color& _clearColor) {
    clearColor = _clearColor;
    onClearColorChanged();
}

void TetMeshVolumeRenderer::setUseEarlyRayTermination(bool _useEarlyRayTermination) {
    if (useEarlyRayTermination != _useEarlyRayTermination) {
        useEarlyRayTermination = _useEarlyRayTermination;
        setShadersDirty(VolumeRendererPassType::ALL);
        checkRecreateTerminationIndexImage();
        reRender = true;
    }
}

void TetMeshVolumeRenderer::setEarlyRayOutThresh(float _thresh) {
    earlyRayOutThresh = _thresh;
}

void TetMeshVolumeRenderer::setEarlyRayOutAlpha(float _alpha) {
    earlyRayOutThresh = 1.0f - _alpha;
}

void TetMeshVolumeRenderer::getVulkanShaderPreprocessorDefines(
        std::map<std::string, std::string>& preprocessorDefines) {
    if (showDepthComplexity) {
        preprocessorDefines.insert(std::make_pair("SHOW_DEPTH_COMPLEXITY", ""));
    }
    if (useClipPlane) {
        preprocessorDefines.insert(std::make_pair("USE_CLIP_PLANE", ""));
    }

    if (useEarlyRayTermination && tetMesh && tetMesh->getUseGradients()) {
        preprocessorDefines.insert(std::make_pair("USE_TERMINATION_INDEX", ""));
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
    if (!renderData->getShaderStages()->hasDescriptorBinding(0, "VertexDepthGradientBuffer")) {
        // For the adjoint projected rasterization pass, we instead want to bind triangle gradient buffers.
        renderData->setStaticBufferOptional(tetMesh->getVertexPositionGradientBuffer(), "VertexPositionGradientBuffer");
        renderData->setStaticBufferOptional(tetMesh->getVertexColorGradientBuffer(), "VertexColorGradientBuffer");
    }
    renderData->setStaticImageViewOptional(outputImageView, "colorImageOpt");
    renderData->setStaticImageViewOptional(colorAdjointImage, "adjointColors");

    if (useEarlyRayTermination) {
        renderData->setStaticImageViewOptional(terminationIndexImageView, "terminationIndexImage");
    }

    // For clip plane data.
    if (useClipPlane) {
        renderData->setStaticBufferOptional(clipPlaneDataBuffer, "ClipPlaneDataBuffer");
    }
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

    if (propertyEditor.addCheckbox("Use Early Ray Out", &useEarlyRayTermination)) {
        setShadersDirty(VolumeRendererPassType::ALL);
        checkRecreateTerminationIndexImage();
        reRender = true;
    }
    if (useEarlyRayTermination) {
        if (propertyEditor.addSliderFloat(
                "Early Ray Alpha Thresh", &earlyRayOutThresh, 1e-6f, 1e-2f, "%.1e",
                ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat)) {
            reRender = true;
        }
    }

    if (propertyEditor.addCheckbox("Use Clip Plane", &useClipPlane)) {
        setShadersDirty(VolumeRendererPassType::ALL);
        reRender = true;
    }
    if (useClipPlane) {
        bool clipPlaneOptionChanged = false;
        if (propertyEditor.addSliderFloat3("Clip Plane Normal", &clipPlaneData.clipPlaneNormal.x, -1.0f, 1.0f)) {
            reRender = true;
            clipPlaneOptionChanged = true;
        }
        if (propertyEditor.addSliderFloat("Clip Plane Distance", &clipPlaneData.clipPlaneDistance, -0.5f, 0.5f)) {
            reRender = true;
            clipPlaneOptionChanged = true;
        }
        if (clipPlaneOptionChanged) {
            clipPlaneDataBuffer->updateData(sizeof(ClipPlaneData), &clipPlaneData, renderer->getVkCommandBuffer());
        }
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
