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

#include <Utils/AppSettings.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Data.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>
#ifndef DISABLE_IMGUI
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/Widgets/NumberFormatting.hpp>
#include <utility>
#endif

#if defined(BUILD_PYTHON_MODULE) && defined(SUPPORT_CUDA_INTEROP)
#include <c10/cuda/CUDAStream.h>
#endif

#include "Tet/RegularGrid.hpp"
#include "RegularGridRendererDVR.hpp"

class RegularGridDvrPass : public sgl::vk::ComputePass {
public:
    explicit RegularGridDvrPass(RegularGridRendererDVR* volumeRenderer, sgl::vk::BufferPtr rendererUniformDataBuffer)
            : ComputePass(volumeRenderer->getRenderer()), volumeRenderer(volumeRenderer),
              rendererUniformDataBuffer(std::move(rendererUniformDataBuffer)) {
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        shaderStages = sgl::vk::ShaderManager->getShaderStages({ "RegularGridDVR.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        const auto& regularGridData = volumeRenderer->getRegularGridData();
        computeData->setStaticBuffer(rendererUniformDataBuffer, "RendererUniformDataBuffer");
        auto scalarField = std::make_shared<sgl::vk::Texture>(
                regularGridData->getFieldImageView(), volumeRenderer->getImageSampler());
        computeData->setStaticTexture(scalarField, "scalarField");
        computeData->setStaticImageView(volumeRenderer->getOutputImageView(), "outputImage");
        computeData->setStaticTexture(
                volumeRenderer->getTransferFunctionWindow()->getTransferFunctionMapTextureVulkan(),
                "transferFunctionTexture");
        computeData->setStaticBuffer(
                volumeRenderer->getTransferFunctionWindow()->getMinMaxUboVulkan(),
                "MinMaxUniformBuffer");
    }
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {
    }
    void _render() override {
        const auto& imageSettings = volumeRenderer->getOutputImageView()->getImage()->getImageSettings();
        renderer->dispatch(computeData, sgl::uiceil(imageSettings.width, 16), sgl::uiceil(imageSettings.height, 16), 1);
    }

private:
    RegularGridRendererDVR* volumeRenderer;
    sgl::vk::BufferPtr rendererUniformDataBuffer;
};

RegularGridRendererDVR::RegularGridRendererDVR(
        sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow)
        : renderer(renderer), camera(camera), transferFunctionWindow(transferFunctionWindow) {
    rendererUniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), sizeof(RenderSettingsData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    regularGridDvrPass = std::make_shared<RegularGridDvrPass>(this, rendererUniformDataBuffer);
}

RegularGridRendererDVR::~RegularGridRendererDVR() {
    renderer->getDevice()->waitIdle();
}
void RegularGridRendererDVR::setRegularGridData(const RegularGridPtr& _regularGrid) {
    regularGrid = _regularGrid;
    regularGridDvrPass->setDataDirty();
    reRender = true;
}

bool RegularGridRendererDVR::loadTransferFunctionFromFile(const std::string& filePath) {
    std::string localFilePath = sgl::AppSettings::get()->getDataDirectory() + "TransferFunctions/" + filePath;
    if (sgl::FileUtils::get()->exists(localFilePath)) {
        return transferFunctionWindow->loadFunctionFromFile(localFilePath);
    } else {
        return transferFunctionWindow->loadFunctionFromFile(filePath);
    }
}

void RegularGridRendererDVR::setOutputImage(sgl::vk::ImageViewPtr& colorImage) {
    outputImageView = colorImage;
}

void RegularGridRendererDVR::recreateSwapchain(uint32_t width, uint32_t height) {
    //windowWidth = int(width);
    //windowHeight = int(height);
}

#if defined(BUILD_PYTHON_MODULE) && defined(SUPPORT_CUDA_INTEROP)
void RegularGridRendererDVR::setViewportSize(uint32_t viewportWidth, uint32_t viewportHeight) {
    outputImageView = {};
    colorImageBuffer = {};
    colorImageBufferCu = {};

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

    colorImageBuffer = std::make_shared<sgl::vk::Buffer>(
            device, viewportWidth * viewportHeight * sizeof(float) * 4,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, true, true);
    colorImageBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(colorImageBuffer);
}

torch::Tensor RegularGridRendererDVR::getImageTensor() {
    auto imageWidth = outputImageView->getImage()->getImageSettings().width;
    auto imageHeight = outputImageView->getImage()->getImageSettings().height;
    torch::Tensor imageTensor = torch::from_blob(
            reinterpret_cast<float*>(colorImageBufferCu->getCudaDevicePtr()),
            { int(imageHeight), int(imageWidth), int(4) },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(useGradients));
    return imageTensor;
}

void RegularGridRendererDVR::copyOutputImageToBuffer() {
    renderer->transitionImageLayout(outputImageView, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    outputImageView->getImage()->copyToBuffer(colorImageBuffer, renderer->getVkCommandBuffer());
    renderer->transitionImageLayout(outputImageView, VK_IMAGE_LAYOUT_GENERAL);
}
#endif

void RegularGridRendererDVR::setUseLinearRGB(bool _useLinearRGB) {
}

void RegularGridRendererDVR::setClearColor(const sgl::Color& _clearColor) {
    clearColor = _clearColor;
    onClearColorChanged();
}

void RegularGridRendererDVR::onClearColorChanged() {
    reRender = true;
}

void RegularGridRendererDVR::setInterpolationMode(RegularGridInterpolationMode _regularGridInterpolationMode) {
    if (regularGridInterpolationMode != _regularGridInterpolationMode) {
        createImageSampler();
        regularGridInterpolationMode = _regularGridInterpolationMode;
    }
}

void RegularGridRendererDVR::createImageSampler() {
    sgl::vk::ImageSamplerSettings samplerSettings{};
    if (regularGridInterpolationMode == RegularGridInterpolationMode::NEAREST) {
        samplerSettings.minFilter = VK_FILTER_NEAREST;
        samplerSettings.magFilter = VK_FILTER_NEAREST;
    } else if (regularGridInterpolationMode == RegularGridInterpolationMode::LINEAR) {
        samplerSettings.minFilter = VK_FILTER_LINEAR;
        samplerSettings.magFilter = VK_FILTER_LINEAR;
    }
    imageSampler = std::make_shared<sgl::vk::ImageSampler>(renderer->getDevice(), samplerSettings, 0.0f);
}

void RegularGridRendererDVR::render() {
    renderSettingsData.backgroundColor = clearColor.getFloatColorRGBA();
    renderSettingsData.attenuationCoefficient = attenuationCoefficient;
    rendererUniformDataBuffer->updateData(
            sizeof(RenderSettingsData), &renderSettingsData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    regularGridDvrPass->render();
}
