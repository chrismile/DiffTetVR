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

#include <iostream>
#include <random>
#include <utility>
#include <thread>

#include <Math/Math.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Scene/RenderTarget.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/imgui.h>
#include <ImGui/imgui_stdlib.h>
#include <ImGui/imgui_custom.h>

#include "Tet/TetMesh.hpp"
#include "Tet/Writers/VtkWriter.hpp"
#include "Renderer/TetMeshVolumeRenderer.hpp"
#include "LossPass.hpp"
#include "OptimizerPass.hpp"
#include "TetRegularizerPass.hpp"
#include "Optimizer.hpp"

ImGuiVulkanImage::ImGuiVulkanImage(sgl::vk::Renderer* renderer, sgl::vk::TexturePtr _texture)
        : renderer(renderer), texture(std::move(_texture)) {
    descriptorSetImGui = ImGui_ImplVulkan_AddTexture(
            texture->getImageSampler()->getVkSampler(),
            texture->getImageView()->getVkImageView(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

ImGuiVulkanImage::~ImGuiVulkanImage() {
    if (descriptorSetImGui) {
        sgl::ImGuiWrapper::get()->freeDescriptorSet(descriptorSetImGui);
        descriptorSetImGui = nullptr;
    }
}

ImTextureID ImGuiVulkanImage::getImGuiTextureId() {
    if (!descriptorSetImGui) {
        sgl::Logfile::get()->throwError("Error in ImGuiVulkanImage::getImGuiTextureId: Texture not initialized.");
    }
    texture->getImage()->transitionImageLayout(
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
    return reinterpret_cast<ImTextureID>(descriptorSetImGui);
}

ImVec2 ImGuiVulkanImage::getTextureSizeImVec2() {
    return {float(texture->getImage()->getImageSettings().width), float(texture->getImage()->getImageSettings().height)};
}


TetMeshOptimizer::TetMeshOptimizer(
        sgl::vk::Renderer* renderer, std::function<void(const TetMeshPtr&)> setTetMeshCallback,
        bool hasDataSets, std::function<std::string()> renderGuiDataSetSelectionMenuCallback,
        sgl::TransferFunctionWindow* transferFunctionWindow)
        : renderer(renderer), setTetMeshCallback(std::move(setTetMeshCallback)), hasDataSets(hasDataSets),
          renderGuiDataSetSelectionMenuCallback(std::move(renderGuiDataSetSelectionMenuCallback)),
          transferFunctionWindow(transferFunctionWindow) {
    camera = std::make_shared<sgl::Camera>();
    tetMeshVolumeRendererGT = std::make_shared<TetMeshVolumeRenderer>(renderer, &camera, transferFunctionWindow);
    tetMeshVolumeRendererOpt = std::make_shared<TetMeshVolumeRenderer>(renderer, &camera, transferFunctionWindow);
    lossPass = std::make_shared<LossPass>(renderer);
    tetRegularizerPass = std::make_shared<TetRegularizerPass>(renderer);
    optimizerPassPositions = std::make_shared<OptimizerPass>(renderer);
    optimizerPassColors = std::make_shared<OptimizerPass>(renderer);
}

void TetMeshOptimizer::openDialog() {
    ImGui::OpenPopup("Optimize Tet Mesh");
    isOptimizationSettingsDialogOpen = true;
}

void TetMeshOptimizer::renderGuiDialog() {
    bool shallStartOptimization = false;
    bool hasResult = false;
    bool hasStopped = false;
    if (ImGui::BeginPopupModal(
            "Optimize Tet Mesh", &isOptimizationSettingsDialogOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (hasDataSets && ImGui::Button("...##gt-sel")) {
            ImGui::OpenPopup("SelectDataSetPopupGT");
        }
        if (ImGui::BeginPopup("SelectDataSetPopupGT")) {
            std::string selection = renderGuiDataSetSelectionMenuCallback();
            if (!selection.empty()) {
                settings.dataSetFileNameGT = selection;
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        ImGui::InputText("Data Set (GT)", &settings.dataSetFileNameGT);

        if (hasDataSets && ImGui::Button("...##opt-sel")) {
            ImGui::OpenPopup("SelectDataSetPopupOpt");
        }
        if (ImGui::BeginPopup("SelectDataSetPopupOpt")) {
            std::string selection = renderGuiDataSetSelectionMenuCallback();
            if (!selection.empty()) {
                settings.dataSetFileNameOpt = selection;
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        ImGui::InputText("Data Set (Opt.)", &settings.dataSetFileNameOpt);

        ImGui::Combo(
                "Optimizer", (int*)&settings.optimizerType, OPTIMIZER_TYPE_NAMES, IM_ARRAYSIZE(OPTIMIZER_TYPE_NAMES));
        ImGui::Combo("Loss", (int*)&settings.lossType, LOSS_TYPE_NAMES, IM_ARRAYSIZE(LOSS_TYPE_NAMES));
        ImGui::SliderInt("Epochs", &settings.maxNumEpochs, 1, 1000);
        if (showPreview) {
            ImGui::SliderInt("Delay (ms)", &previewDelay, 0, 1000);
        }
        ImGui::SameLine();
        ImGui::Checkbox("Show Preview", &showPreview);

        const char* const PARAMETER_NAMES[] = { "Positions", "Colors" };
        for (int i = 0; i < 2; i++) {
            if (i == 0 && !settings.optimizePositions) {
                continue;
            }
            if (i == 1 && !settings.optimizeColors) {
                continue;
            }
            OptimizerSettings& optSettings =
                    i == 0 ? settings.optimizerSettingsPositions : settings.optimizerSettingsColors;
            std::string name = std::string() + "Optimizer Settings (" + PARAMETER_NAMES[i] + ")";
            if (ImGui::CollapsingHeader(name.c_str(), nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
                std::string alphaId = std::string("alpha##alpha-") + PARAMETER_NAMES[i];
                ImGui::SliderFloat(alphaId.c_str(), &optSettings.learningRate, 0.0f, 1.0f, "%.4f");
                if (settings.optimizerType == OptimizerType::ADAM) {
                    std::string beta1Id = std::string("beta1##beta1-") + PARAMETER_NAMES[i];
                    std::string beta2Id = std::string("beta2##beta2-") + PARAMETER_NAMES[i];
                    ImGui::SliderFloat(beta1Id.c_str(), &optSettings.beta1, 0.0f, 1.0f);
                    ImGui::SliderFloat(beta2Id.c_str(), &optSettings.beta2, 0.0f, 1.0f);
                }
            }
        }

        if (ImGui::CollapsingHeader("Tet Regularizer", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat(
                    "Weight lambda##lambda-regularizer", &settings.tetRegularizerSettings.lambda, 0.0f, 1.0f, "%.4f");
            ImGui::SliderFloat("Softplus beta##beta-regularizer", &settings.tetRegularizerSettings.beta, 1.0f, 50.0f);
        }

        if (ImGui::CollapsingHeader(
                "DVR Settings", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderIntPowerOfTwo("Image Width", (int*)&settings.imageWidth, 16, 4096);
            ImGui::SliderIntPowerOfTwo("Image Height", (int*)&settings.imageHeight, 16, 4096);
            ImGui::SliderFloat("Attenuation", &settings.attenuationCoefficient, 0.0f, 200.0f);
            ImGui::Checkbox("Sample Random View", &settings.sampleRandomView);
        }

        if (settings.optimizePositions) {
            ImGui::Checkbox("Export Position Gradients Field", &settings.exportPositionGradients);
            if (settings.exportPositionGradients) {
                ImGui::InputText("File Path", &settings.exportFileNameGradientField);
                ImGui::Checkbox("Binary VTK", &settings.isBinaryVtk);
            }
        }

        if (ImGui::Button("OK", ImVec2(120, 0))) {
            shallStartOptimization = true;
        }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }

        if (shallStartOptimization) {
            ImGui::OpenPopup("Optimization Progress");
            isOptimizationProgressDialogOpen = true;
            startRequest();
        }

        if (ImGui::BeginPopupModal(
                "Optimization Progress", &isOptimizationProgressDialogOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
            float progress = getProgress();
            ImGui::Text(
                    "Progress: Epoch %d of %d...",
                    int(std::round(progress * float(settings.maxNumEpochs))), settings.maxNumEpochs);
            ImGui::ProgressSpinner(
                    "##progress-spinner-tfopt", -1.0f, -1.0f, 4.0f,
                    ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
            ImGui::SameLine();
            ImGui::ProgressBar(progress, ImVec2(300, 0));
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                hasStopped = true;
            }
            if (!getHasResult()) {
                updateRequest();
            }
            hasResult = getHasResult();
            if (showPreview && settings.maxNumEpochs > 0) {
                if (previewDelay > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(int(previewDelay)));
                }
                renderer->insertImageMemoryBarriers(
                        std::vector<sgl::vk::ImagePtr>{ colorImageGT->getImage(), colorImageOpt->getImage() },
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
                ImVec2 sizeContent = colorImageGTImGui->getTextureSizeImVec2();
                ImTextureID textureIdGT = colorImageGTImGui->getImGuiTextureId();
                ImGui::Image(
                        textureIdGT, sizeContent, ImVec2(0, 0), ImVec2(1, 1));
                ImTextureID textureIdOpt = colorImageOptImGui->getImGuiTextureId();
                ImGui::Image(
                        textureIdOpt, sizeContent, ImVec2(0, 0), ImVec2(1, 1));
            }
            if (hasResult || hasStopped) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::EndPopup();
    }

    if (hasResult && !hasStopped) {
        if (settings.optimizerSettingsPositions.learningRate > 0.0f) {
            tetMeshOpt->setVerticesChangedOnDevice(true);
        }
        setTetMeshCallback(tetMeshOpt);
        needsReRender = true;
    }
    if (hasResult || hasStopped) {
        tetMeshGT = {};
        tetMeshOpt = {};
        hasRequest = false;
    }
}

float TetMeshOptimizer::getProgress() {
    return float(currentEpoch) / float(settings.maxNumEpochs);
}

void TetMeshOptimizer::startRequest() {
    renderer->syncWithCpu();
    hasRequest = true;
    currentEpoch = 0;

    auto* device = renderer->getDevice();
    //sgl::AABB3 aabb = tetMeshOpt->getBoundingBoxRendering();
    float fovy = std::atan(1.0f / 2.0f) * 2.0f;
    //float aspect = float(settings.imageWidth) / float(settings.imageHeight);
    //glm::mat4 projectionMatrix = glm::perspectiveRH_ZO(fovy, aspect, 0.001f, 100.0f);
    camera->setFOVy(fovy);
    camera->setNearClipDistance(0.001f);
    camera->setFarClipDistance(100.0f);
    auto renderTarget = std::make_shared<sgl::RenderTarget>(int(settings.imageWidth), int(settings.imageHeight));
    camera->setRenderTarget(renderTarget, false);
    camera->onResolutionChanged({});
    int paddedViewportWidth = 0, paddedViewportHeight = 0;
    std::map<std::string, std::string> preprocessorDefinesRenderer;
    tetMeshVolumeRendererOpt->getScreenSizeWithTiling(paddedViewportWidth, paddedViewportHeight);
    tetMeshVolumeRendererOpt->getVulkanShaderPreprocessorDefines(preprocessorDefinesRenderer);

    vtkWriter = {};
    if (settings.exportPositionGradients && settings.optimizePositions) {
        vtkWriter = std::make_shared<VtkWriter>();
        vtkWriter->initializeWriter(settings.exportFileNameGradientField, settings.isBinaryVtk);
    }

    optimizerPassPositions->setOptimizerType(settings.optimizerType);
    optimizerPassPositions->setSettings(
            settings.lossType, settings.optimizerSettingsPositions.learningRate,
            settings.optimizerSettingsPositions.beta1, settings.optimizerSettingsPositions.beta2,
            settings.optimizerSettingsPositions.epsilon, false);
    optimizerPassColors->setOptimizerType(settings.optimizerType);
    optimizerPassColors->setSettings(
            settings.lossType, settings.optimizerSettingsColors.learningRate,
            settings.optimizerSettingsColors.beta1, settings.optimizerSettingsColors.beta2,
            settings.optimizerSettingsColors.epsilon, true);
    lossPass->setSettings(
            settings.lossType, settings.imageWidth, settings.imageHeight,
            uint32_t(paddedViewportWidth), uint32_t(paddedViewportHeight), preprocessorDefinesRenderer);
    lossPass->updateUniformBuffer();
    tetRegularizerPass->setSettings(settings.tetRegularizerSettings.lambda, settings.tetRegularizerSettings.beta);
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    if (viewportWidth != settings.imageWidth || viewportHeight != settings.imageHeight) {
        colorImageGT = {};
        colorImageOpt = {};
        colorAdjointTexture = {};
        adjointPassBackbuffer = {};

        sgl::vk::ImageSettings imageSettings{};
        imageSettings.width = settings.imageWidth;
        imageSettings.height = settings.imageHeight;
        imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageSettings.usage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;
        colorImageGT = std::make_shared<sgl::vk::Texture>(device, imageSettings);
        colorImageOpt = std::make_shared<sgl::vk::Texture>(device, imageSettings);
        colorAdjointTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings);
        renderer->insertImageMemoryBarrier(
                colorAdjointTexture->getImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        //colorAdjointTexture->getImage()->transitionImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        imageSettings.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
        adjointPassBackbuffer = std::make_shared<sgl::vk::ImageView>(std::make_shared<sgl::vk::Image>(
                device, imageSettings));
    }

    if (viewportWidth != settings.imageWidth || viewportHeight != settings.imageHeight
            || usePreviewCached != showPreview) {
        colorImageGTImGui = {};
        colorImageOptImGui = {};
        colorImageGTImGui = std::make_shared<ImGuiVulkanImage>(renderer, colorImageGT);
        colorImageOptImGui = std::make_shared<ImGuiVulkanImage>(renderer, colorImageOpt);
    }

    tetMeshGT = std::make_shared<TetMesh>(device, transferFunctionWindow);
    tetMeshOpt = std::make_shared<TetMesh>(device, transferFunctionWindow);
    bool dataLoadedGT = tetMeshGT->loadFromFile(settings.dataSetFileNameGT);
    bool dataLoadedOpt = tetMeshOpt->loadFromFile(settings.dataSetFileNameOpt);
    if (!dataLoadedGT || !dataLoadedOpt) {
        hasRequest = false;
        tetMeshGT = {};
        tetMeshOpt = {};
    }

    auto vertexPositionBuffer = tetMeshOpt->getVertexPositionBuffer();
    auto vertexColorBuffer = tetMeshOpt->getVertexColorBuffer();
    sgl::vk::BufferSettings bufferSettings{};
    bufferSettings.usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferSettings.sizeInBytes = vertexPositionBuffer->getSizeInBytes();
    vertexPositionGradientBuffer = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);
    bufferSettings.sizeInBytes = vertexColorBuffer->getSizeInBytes();
    vertexColorGradientBuffer = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);
    optimizerPassPositions->setBuffers(vertexPositionBuffer, vertexPositionGradientBuffer);
    optimizerPassColors->setBuffers(vertexColorBuffer, vertexColorGradientBuffer);
    auto cellIndicesBuffer = tetMeshOpt->getCellIndicesBuffer();
    tetRegularizerPass->setBuffers(cellIndicesBuffer, vertexPositionBuffer, vertexPositionGradientBuffer);

    // TODO: Make sure data is freed before allocating new data.
    tetMeshVolumeRendererGT->setClearColor(sgl::Color(0, 0, 0, 0));
    tetMeshVolumeRendererGT->setTetMeshData(tetMeshGT);
    tetMeshVolumeRendererGT->setOutputImage(colorImageGT->getImageView());
    tetMeshVolumeRendererGT->recreateSwapchain(settings.imageWidth, settings.imageHeight);
    fragmentBufferSize = tetMeshVolumeRendererGT->getFragmentBufferSize();
    fragmentBuffer = tetMeshVolumeRendererGT->getFragmentBuffer();
    startOffsetBuffer = tetMeshVolumeRendererGT->getStartOffsetBuffer();
    fragmentCounterBuffer = tetMeshVolumeRendererGT->getFragmentCounterBuffer();

    tetMeshVolumeRendererOpt->setClearColor(sgl::Color(0, 0, 0, 0));
    tetMeshVolumeRendererOpt->setTetMeshData(tetMeshOpt);
    tetMeshVolumeRendererOpt->setOutputImage(colorImageOpt->getImageView());
    tetMeshVolumeRendererOpt->setAdjointPassData(
            colorAdjointTexture->getImageView(), adjointPassBackbuffer,
            vertexPositionGradientBuffer, vertexColorGradientBuffer);
    tetMeshVolumeRendererOpt->recreateSwapchainExternal(
            settings.imageWidth, settings.imageHeight, fragmentBufferSize,
            fragmentBuffer, startOffsetBuffer, fragmentCounterBuffer);

    if (viewportWidth != settings.imageWidth || viewportHeight != settings.imageHeight) {
        lossPass->setImageViews(
                colorImageGT->getImageView(), colorImageOpt->getImageView(), colorAdjointTexture->getImageView(),
                tetMeshVolumeRendererGT->getStartOffsetBuffer(), tetMeshVolumeRendererOpt->getStartOffsetBuffer());
    }

    viewportWidth = settings.imageWidth;
    viewportHeight = settings.imageHeight;
    usePreviewCached = showPreview;
}

bool TetMeshOptimizer::getHasResult() {
    return hasRequest && currentEpoch == settings.maxNumEpochs;
}

void TetMeshOptimizer::sampleCameraPoses() {
    const glm::vec3 globalUp(0.0f, 1.0f, 0.0f);
    //for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    float theta;
    float phi;
    if (settings.sampleRandomView) {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<float> dist(0, 1);
        theta = sgl::TWO_PI * dist(generator);
        phi = std::acos(1.0f - 2.0f * dist(generator));
    } else {
        theta = 0.0f;
        phi = 0.0f;
    }
    glm::vec3 cameraPosition(std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi));
    glm::vec3 cameraForward = glm::normalize(cameraPosition);
    glm::vec3 cameraRight = glm::normalize(glm::cross(globalUp, cameraForward));
    glm::vec3 cameraUp = glm::normalize(glm::cross(cameraForward, cameraRight));
    glm::mat4 rotationMatrix;
    for (int i = 0; i < 4; i++) {
        rotationMatrix[0][i] = i < 3 ? cameraRight[i] : 0.0f;
        rotationMatrix[1][i] = i < 3 ? cameraUp[i] : 0.0f;
        rotationMatrix[2][i] = i < 3 ? cameraForward[i] : 0.0f;
        rotationMatrix[3][i] = i < 3 ? 0.0f : 1.0f;
    }
    glm::mat4 inverseViewMatrix = sgl::matrixTranslation(cameraPosition) * rotationMatrix;
    camera->overwriteViewMatrix(glm::inverse(inverseViewMatrix));
    // TODO
    //batchSettingsArray.at(batchIdx) = inverseViewMatrix;
    //}
    //batchSettingsBuffer->updateData(
    //        sizeof(glm::mat4) * batchSize, batchSettingsArray.data(), renderer->getVkCommandBuffer());

    /*auto inverseProjectionMatrix = dvrSettings.inverseProjectionMatrix;
    auto inverseViewMatrix = batchSettingsArray.front();
    uint32_t imageWidth = settings.imageWidth;
    uint32_t imageHeight = settings.imageHeight;
    glm::uvec2 gl_GlobalInvocationID(settings.imageWidth / 2, settings.imageHeight / 2);
    glm::vec3 rayOrigin = inverseViewMatrix[3];
    glm::vec2 fragNdc = 2.0f * ((glm::vec2(gl_GlobalInvocationID) + glm::vec2(0.5f)) / glm::vec2(imageWidth, imageHeight)) - 1.0f;
    glm::vec3 rayTarget = (inverseProjectionMatrix * glm::vec4(fragNdc, 1.0, 1.0));
    glm::vec3 normalizedTarget = normalize(rayTarget);
    glm::vec3 rayDirection = (inverseViewMatrix * glm::vec4(normalizedTarget, 0.0));
    std::cout << "o: " << rayOrigin.x << ", " << rayOrigin.y << ", " << rayOrigin.z << std::endl;
    std::cout << "d: " << rayDirection.x << ", " << rayDirection.y << ", " << rayDirection.z << std::endl;
    std::cout << "m: " << rayTarget[0] << " " << rayTarget[1] << " " << rayTarget[2] << std::endl;*/
    //for (int i = 0; i < 4; i++) {
    //    std::cout << "m: " << inverseViewMatrix[i][0] << " " << inverseViewMatrix[i][1] << " "
    //            << inverseViewMatrix[i][2] << " " << inverseViewMatrix[i][3] << std::endl;
    //}
}

void TetMeshOptimizer::updateRequest() {
    // Randomly sample camera poses.
    sampleCameraPoses();
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Run the forward passes.
    tetMeshVolumeRendererGT->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    tetMeshVolumeRendererOpt->render();
    renderer->insertImageMemoryBarriers(
            std::vector<sgl::vk::ImagePtr>{ colorImageGT->getImage(), colorImageOpt->getImage() },
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    // Compute the image loss.
    lossPass->render();
    renderer->insertImageMemoryBarrier(
            colorAdjointTexture->getImage(),
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    // Clear the gradients.
    if (settings.optimizePositions) {
        vertexPositionGradientBuffer->fill(0, renderer->getVkCommandBuffer());
        VkPipelineStageFlags destStage =
                settings.tetRegularizerSettings.lambda > 0.0f
                ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, destStage,
                vertexPositionGradientBuffer);
    }
    if (settings.optimizeColors) {
        vertexColorGradientBuffer->fill(0, renderer->getVkCommandBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                vertexColorGradientBuffer);
    }

    // Compute the tet regularizer loss/gradients.
    if (settings.optimizePositions && settings.tetRegularizerSettings.lambda > 0.0f) {
        tetRegularizerPass->render();
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                vertexPositionGradientBuffer);
    }

    // Compute the gradients wrt. the transfer function entries for the image loss.
    tetMeshVolumeRendererOpt->renderAdjoint();
    if (settings.optimizePositions) {
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vertexPositionGradientBuffer);
    }
    if (settings.optimizeColors) {
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vertexColorGradientBuffer);
    }

    if (settings.exportPositionGradients && settings.optimizePositions) {
        const auto& cellIndices = tetMeshOpt->getCellIndices();
        auto vertexPositionBuffer = tetMeshOpt->getVertexPositionBuffer();
        if (!vertexPositionStagingBuffer
            || vertexPositionStagingBuffer->getSizeInBytes() != vertexPositionBuffer->getSizeInBytes()) {
            vertexPositionStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), vertexPositionBuffer->getSizeInBytes(),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        }
        if (!vertexPositionGradientStagingBuffer
            || vertexPositionGradientStagingBuffer->getSizeInBytes() != vertexPositionGradientBuffer->getSizeInBytes()) {
            vertexPositionGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), vertexPositionGradientBuffer->getSizeInBytes(),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        }
        renderer->insertMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        vertexPositionBuffer->copyDataTo(
                vertexPositionStagingBuffer, 0, 0, vertexPositionStagingBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        vertexPositionGradientBuffer->copyDataTo(
                vertexPositionGradientStagingBuffer, 0, 0, vertexPositionGradientBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        renderer->syncWithCpu();

        auto* vertexPositions = reinterpret_cast<glm::vec3*>(vertexPositionStagingBuffer->mapMemory());
        auto* vertexPositionGradients = reinterpret_cast<glm::vec3*>(vertexPositionGradientStagingBuffer->mapMemory());
        const auto numVertices = int(vertexPositionBuffer->getSizeInBytes() / sizeof(glm::vec3));
        vtkWriter->writeNextTimeStep(
                cellIndices, vertexPositions, vertexPositionGradients, numVertices);
        vertexPositionStagingBuffer->unmapMemory();
        vertexPositionGradientStagingBuffer->unmapMemory();
    }

    /*bool debugOutput = false;
    if (debugOutput) {
        auto vertexPositionGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), vertexPositionGradientBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        auto vertexColorGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), vertexColorGradientBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        vertexPositionGradientBuffer->copyDataTo(
                vertexPositionGradientStagingBuffer, 0, 0, vertexPositionGradientBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        vertexColorGradientBuffer->copyDataTo(
                vertexColorGradientStagingBuffer, 0, 0, vertexColorGradientBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        renderer->syncWithCpu();

        uint32_t numEntriesPos = vertexPositionGradientBuffer->getSizeInBytes() / sizeof(float);
        uint32_t numEntriesCol = vertexColorGradientBuffer->getSizeInBytes() / sizeof(float);
        auto* vertexPositionGradients = reinterpret_cast<float*>(vertexPositionGradientStagingBuffer->mapMemory());
        auto* vertexColorGradients = reinterpret_cast<float*>(vertexColorGradientStagingBuffer->mapMemory());
        std::cout << "vertexPositionGradients:" << std::endl;
        for (uint32_t i = 0; i < numEntriesPos; i++) {
            std::cout << vertexPositionGradients[i] << std::endl;
        }
        std::cout << std::endl << "vertexColorGradients:" << std::endl;
        for (uint32_t i = 0; i < numEntriesCol; i++) {
            std::cout << vertexColorGradients[i] << std::endl;
        }
        std::cout << std::endl;
        vertexPositionGradientStagingBuffer->unmapMemory();
        vertexColorGradientStagingBuffer->unmapMemory();
    }*/

    // Run the optimizer.
    if (settings.optimizePositions) {
        optimizerPassPositions->setEpochIndex(currentEpoch);
        optimizerPassPositions->render();
    }
    if (settings.optimizeColors) {
        optimizerPassColors->setEpochIndex(currentEpoch);
        optimizerPassColors->render();
    }
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    /*if (debugOutput) {
        auto vertexPositionBuffer = tetMeshGT->getVertexPositionBuffer();
        auto vertexColorBuffer = tetMeshGT->getVertexColorBuffer();
        auto vertexPositionStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), vertexPositionBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        auto vertexColorStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), vertexColorBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        vertexPositionBuffer->copyDataTo(
                vertexPositionStagingBuffer, 0, 0, vertexPositionBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        vertexColorBuffer->copyDataTo(
                vertexColorStagingBuffer, 0, 0, vertexColorBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        renderer->syncWithCpu();

        uint32_t numEntriesPos = vertexPositionBuffer->getSizeInBytes() / sizeof(float);
        uint32_t numEntriesCol = vertexColorBuffer->getSizeInBytes() / sizeof(float);
        auto* vertexPositions = reinterpret_cast<float*>(vertexPositionStagingBuffer->mapMemory());
        auto* vertexColors = reinterpret_cast<float*>(vertexColorStagingBuffer->mapMemory());
        std::cout << "vertexPositions:" << std::endl;
        for (uint32_t i = 0; i < numEntriesPos; i++) {
            std::cout << vertexPositions[i] << std::endl;
        }
        std::cout << std::endl << "vertexColors:" << std::endl;
        for (uint32_t i = 0; i < numEntriesCol; i++) {
            std::cout << vertexColors[i] << std::endl;
        }
        std::cout << std::endl;
        vertexPositionStagingBuffer->unmapMemory();
        vertexColorStagingBuffer->unmapMemory();
    }*/

    currentEpoch++;
}
