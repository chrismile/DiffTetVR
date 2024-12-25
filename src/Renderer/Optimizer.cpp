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
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Scene/RenderTarget.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#ifndef DISABLE_IMGUI
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/imgui.h>
#include <ImGui/imgui_stdlib.h>
#include <ImGui/imgui_custom.h>
#endif

#include "Tet/TetMesh.hpp"
#include "Tet/Loaders/LoadersUtil.hpp"
#include "Tet/Loaders/TetMeshLoader.hpp"
#include "Tet/Writers/VtkWriter.hpp"
#include "Renderer/TetMeshRendererPPLL.hpp"
#include "Renderer/TetMeshRendererProjection.hpp"
#include "Renderer/TetMeshRendererIntersection.hpp"
#include "LossPass.hpp"
#include "OptimizerPass.hpp"
#include "TetRegularizerPass.hpp"
#include "Optimizer.hpp"

#ifndef DISABLE_IMGUI
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
#endif


TetMeshOptimizer::TetMeshOptimizer(
        sgl::vk::Renderer* renderer, std::function<void(const TetMeshPtr&, float)> setTetMeshCallback,
        bool hasDataSets, std::function<std::string()> renderGuiDataSetSelectionMenuCallback,
        sgl::TransferFunctionWindow* transferFunctionWindow)
        : renderer(renderer), setTetMeshCallback(std::move(setTetMeshCallback)), hasDataSets(hasDataSets),
          renderGuiDataSetSelectionMenuCallback(std::move(renderGuiDataSetSelectionMenuCallback)),
          transferFunctionWindow(transferFunctionWindow) {
    camera = std::make_shared<sgl::Camera>();
    lossPass = std::make_shared<LossPass>(renderer);
    tetRegularizerPass = std::make_shared<TetRegularizerPass>(renderer);
    optimizerPassPositions = std::make_shared<OptimizerPass>(renderer);
    optimizerPassColors = std::make_shared<OptimizerPass>(renderer);
}

void TetMeshOptimizer::setTetMeshRendererType(TetMeshRendererType _tetMeshRendererType) {
    if (tetMeshRendererType != _tetMeshRendererType || !tetMeshVolumeRendererGT) {
        tetMeshRendererType = _tetMeshRendererType;
        if (tetMeshRendererType == TetMeshRendererType::PPLL) {
            tetMeshVolumeRendererGT = std::make_shared<TetMeshRendererPPLL>(renderer, &camera, transferFunctionWindow);
            tetMeshVolumeRendererOpt = std::make_shared<TetMeshRendererPPLL>(renderer, &camera, transferFunctionWindow);
        } else if (tetMeshRendererType == TetMeshRendererType::PROJECTION) {
            tetMeshVolumeRendererGT = std::make_shared<TetMeshRendererProjection>(renderer, &camera, transferFunctionWindow);
            tetMeshVolumeRendererOpt = std::make_shared<TetMeshRendererProjection>(renderer, &camera, transferFunctionWindow);
        } else if (tetMeshRendererType == TetMeshRendererType::INTERSECTION) {
            tetMeshVolumeRendererGT = std::make_shared<TetMeshRendererIntersection>(renderer, &camera, transferFunctionWindow);
            tetMeshVolumeRendererOpt = std::make_shared<TetMeshRendererIntersection>(renderer, &camera, transferFunctionWindow);
        }
    }
}

#ifndef DISABLE_IMGUI
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
                std::string fileExtension = sgl::FileUtils::get()->getFileExtensionLower(settings.dataSetFileNameGT);
                auto tetMeshTest = std::make_shared<TetMesh>(renderer->getDevice(), transferFunctionWindow);
                TetMeshLoader* tetMeshLoader = tetMeshTest->createTetMeshLoaderByExtension(fileExtension);
                size_t numCells = 0, numVertices = 0;
                if (tetMeshLoader && tetMeshLoader->peekSizes(settings.dataSetFileNameGT, numCells, numVertices)) {
                    numCellsGT = uint32_t(numCells);
                }
                delete tetMeshLoader;
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        ImGui::InputText("Data Set (GT)", &settings.dataSetFileNameGT);
        if (numCellsGT > 0) {
            ImGui::SameLine();
            if (numCellsGT == 1) {
                ImGui::Text("1 tet");
            } else {
                ImGui::Text("%u tets", numCellsGT);
            }
        }

        if (hasDataSets && ImGui::Button("...##opt-sel")) {
            ImGui::OpenPopup("SelectDataSetPopupOpt");
        }
        if (ImGui::BeginPopup("SelectDataSetPopupOpt")) {
            std::string selection = renderGuiDataSetSelectionMenuCallback();
            if (!selection.empty()) {
                settings.dataSetFileNameOpt = selection;
                std::string fileExtension = sgl::FileUtils::get()->getFileExtensionLower(settings.dataSetFileNameOpt);
                auto tetMeshTest = std::make_shared<TetMesh>(renderer->getDevice(), transferFunctionWindow);
                TetMeshLoader* tetMeshLoader = tetMeshTest->createTetMeshLoaderByExtension(fileExtension);
                size_t numCells = 0, numVertices = 0;
                if (tetMeshLoader && tetMeshLoader->peekSizes(settings.dataSetFileNameOpt, numCells, numVertices)) {
                    numCellsOpt = uint32_t(numCells);
                }
                delete tetMeshLoader;
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        ImGui::InputText("Data Set (Opt.)", &settings.dataSetFileNameOpt);
        if (numCellsOpt > 0) {
            ImGui::SameLine();
            if (numCellsOpt == 1) {
                ImGui::Text("1 tet");
            } else {
                ImGui::Text("%u tets", numCellsOpt);
            }
        }

        ImGui::Combo(
                "Optimizer", (int*)&settings.optimizerType, OPTIMIZER_TYPE_NAMES, IM_ARRAYSIZE(OPTIMIZER_TYPE_NAMES));
        ImGui::Combo("Loss", (int*)&settings.lossType, LOSS_TYPE_NAMES, IM_ARRAYSIZE(LOSS_TYPE_NAMES));
        ImGui::SliderInt("Epochs", &settings.maxNumEpochs, 1, 1000);
        if (showPreview) {
            ImGui::SliderInt("Delay (ms)", &previewDelay, 0, 1000);
        }
        ImGui::SameLine();
        ImGui::Checkbox("Show Preview", &showPreview);
        ImGui::Checkbox("Fix Boundary", &settings.fixBoundary);

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
                std::string lrDecayId = std::string("Exp. Decay##lrdec-") + PARAMETER_NAMES[i];
                ImGui::SliderFloat(lrDecayId.c_str(), &optSettings.lrDecayRate, 0.0f, 1.0f, "%.4f");
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

        ImGui::Checkbox("Use Coarse to Fine", &settings.useCoarseToFine);
        if (settings.useCoarseToFine && ImGui::CollapsingHeader(
                "Coarse to Fine Settings", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderInt("Max. Tets", (int*)&settings.maxNumTets, 1, 128*128*128*6);
            ImGui::Combo(
                    "Split Gradient Type", (int*)&settings.splitGradientType,
                    SPLIT_GRADIENT_TYPE_NAMES, IM_ARRAYSIZE(SPLIT_GRADIENT_TYPE_NAMES));
            ImGui::SliderFloat("#Splits Ratio", &settings.numSplitsRatio, 0.0f, 1.0f);
            ImGui::Checkbox("Use Init. Grid", &settings.useConstantInitGrid);
            if (settings.useConstantInitGrid) {
                ImGui::Combo(
                        "Init Grid Type", (int*)&settings.initGridType,
                        INIT_GRID_TYPE_NAMES, IM_ARRAYSIZE(INIT_GRID_TYPE_NAMES));
                ImGui::SliderInt3("Init Grid Size", (int*)&settings.initGridResolution.x, 1, 128);
                if (settings.initGridType == InitGridType::MESHING_FTETWILD) {
                    ImGui::SliderDouble("Ideal Edge Length", &settings.fTetWildParams.relativeIdealEdgeLength, 0.01, 0.5);
                    ImGui::SliderDouble("Epsilon", &settings.fTetWildParams.epsilon, 1e-4, 1e-2);
                    ImGui::Checkbox("Skip Simplify", &settings.fTetWildParams.skipSimplify);
                    ImGui::Checkbox("Coarsen", &settings.fTetWildParams.coarsen);
                } else if (settings.initGridType == InitGridType::MESHING_TETGEN) {
                    ImGui::Checkbox("Use Steiner Points", &settings.tetGenParams.useSteinerPoints);
                    if (settings.tetGenParams.useSteinerPoints) {
                        ImGui::Checkbox("Use Radius-Edge Ratio", &settings.tetGenParams.useRadiusEdgeRatioBound);
                        if (settings.tetGenParams.useRadiusEdgeRatioBound) {
                            ImGui::SliderDouble("Radius-Edge Ratio Bound", &settings.tetGenParams.radiusEdgeRatioBound, 0.1f, 2.0f);
                        }
                        ImGui::Checkbox("Use Max. Volume", &settings.tetGenParams.useMaximumVolumeConstraint);
                        if (settings.tetGenParams.useMaximumVolumeConstraint) {
                            ImGui::SliderDouble("Max. Tetrahedron Volume", &settings.tetGenParams.maximumTetrahedronVolume, 0.1f, 2.0f);
                        }
                    }
                    ImGui::Checkbox("Coarsen", &settings.tetGenParams.coarsen);
                    ImGui::SliderDouble("Max. Dihedral Angle", &settings.tetGenParams.maximumDihedralAngle, 10.0f, 180.0f);
                    ImGui::SliderInt("Mesh Opt. Level", &settings.tetGenParams.meshOptimizationLevel, 0, 10);
                    ImGui::Checkbox("Use Edge/Face Flips", &settings.tetGenParams.useEdgeAndFaceFlips);
                    ImGui::Checkbox("Use Vertex Smoothing", &settings.tetGenParams.useVertexSmoothing);
                    ImGui::Checkbox("Use Vertex Ins./Del.", &settings.tetGenParams.useVertexInsertionAndDeletion);
                }
            }
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
            startRequest();
            if (hasRequest) {
                ImGui::OpenPopup("Optimization Progress");
                isOptimizationProgressDialogOpen = true;
            }
        }

        if (ImGui::BeginPopupModal(
                "Optimization Progress", &isOptimizationProgressDialogOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
            //const int coarseToFineEpochIndex = coarseToFineEpoch % 3;
            const float progressBarWidth = settings.useCoarseToFine ? 400.0f : 300.0f;
            if (settings.useCoarseToFine) {
                auto progress = float(double(tetMeshOpt->getNumCells()) / double(settings.maxNumTets));
                progress = std::min(progress, 1.0f);
                ImGui::Text("Progress: #Tets %u of %u...", uint32_t(tetMeshOpt->getNumCells()), settings.maxNumTets);
                ImGui::ProgressSpinner(
                        "##progress-spinner-tets", -1.0f, -1.0f, 4.0f,
                        ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
                ImGui::SameLine();
                ImGui::ProgressBar(progress, ImVec2(progressBarWidth, 0));
            }
            float progress = getProgress();
            ImGui::Text(
                    "Progress: Epoch %d of %d...",
                    int(std::round(progress * float(settings.maxNumEpochs))), settings.maxNumEpochs);
            ImGui::ProgressSpinner(
                    "##progress-spinner-epochs", -1.0f, -1.0f, 4.0f,
                    ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
            ImGui::SameLine();
            ImGui::ProgressBar(progress, ImVec2(progressBarWidth, 0));
            if (!settings.useCoarseToFine) {
                ImGui::SameLine();
            }
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                hasStopped = true;
            }
            if (!getHasResult()) {
                updateRequest();
            }
            hasResult = getHasResult();
            if (showPreview && settings.maxNumEpochs > 0
                    && (!hasResult || (this->currentEpoch != 0 && this->coarseToFineEpoch != 0))) {
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
        auto timeEnd = std::chrono::system_clock::now();
        auto elapsedLoad = std::chrono::duration_cast<std::chrono::microseconds >(timeEnd - timeStart);
        std::cout << "Elapsed time optimization: " << (double(elapsedLoad.count()) * 1e-6) << "s" << std::endl;
        tetMeshOpt->setVerticesChangedOnDevice(true);
        tetMeshOpt->setUseGradients(false);
        setTetMeshCallback(tetMeshOpt, settings.attenuationCoefficient);
        needsReRender = true;
    }
    if (hasResult || hasStopped) {
        tetMeshGT = {};
        tetMeshOpt = {};
        hasRequest = false;
    }
}
#endif

float TetMeshOptimizer::getProgress() {
    return float(currentEpoch) / float(settings.maxNumEpochs);
}

void TetMeshOptimizer::startRequest() {
    renderer->syncWithCpu();
    hasRequest = true;
    currentEpoch = 0;
    coarseToFineEpoch = 0;

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
            settings.optimizerSettingsPositions.epsilon, false, settings.fixBoundary);
    optimizerPassColors->setOptimizerType(settings.optimizerType);
    optimizerPassColors->setSettings(
            settings.lossType, settings.optimizerSettingsColors.learningRate,
            settings.optimizerSettingsColors.beta1, settings.optimizerSettingsColors.beta2,
            settings.optimizerSettingsColors.epsilon, true, settings.fixBoundary);
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

#ifndef DISABLE_IMGUI
    if (viewportWidth != settings.imageWidth || viewportHeight != settings.imageHeight
            || usePreviewCached != showPreview || tetMeshRendererTypeCached != tetMeshRendererType) {
        colorImageOptPreview = {};
        colorImageGTImGui = {};
        colorImageOptImGui = {};
        colorImageGTImGui = std::make_shared<ImGuiVulkanImage>(renderer, colorImageGT);
        if (tetMeshRendererType == TetMeshRendererType::PPLL) {
            colorImageOptImGui = std::make_shared<ImGuiVulkanImage>(renderer, colorImageOpt);
        } else {
            auto imageSettings = colorImageOpt->getImage()->getImageSettings();
            imageSettings.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            colorImageOptPreview = std::make_shared<sgl::vk::Texture>(device, imageSettings);
            colorImageOptImGui = std::make_shared<ImGuiVulkanImage>(renderer, colorImageOptPreview);
        }
    }
#endif

    tetMeshGT = std::make_shared<TetMesh>(device, transferFunctionWindow);
    tetMeshOpt = std::make_shared<TetMesh>(device, transferFunctionWindow);
    if (settings.useCoarseToFine) {
        tetMeshOpt->setForceUseOvmRepresentation();
    }
    tetMeshOpt->setUseGradients(true);
    bool dataLoadedGT = tetMeshGT->loadFromFile(settings.dataSetFileNameGT);
    bool dataLoadedOpt;
    if (settings.useConstantInitGrid) {
        dataLoadedOpt = true;
        if (settings.initGridType == InitGridType::DECOMPOSED_HEX_MESH) {
            tetMeshOpt->setHexMeshConst(
                    tetMeshGT->getBoundingBox(), settings.initGridResolution.x,
                    settings.initGridResolution.y, settings.initGridResolution.z,
                    glm::vec4(0.5f, 0.5f, 0.5f, 0.1f));
        } else if (settings.initGridType == InitGridType::MESHING_FTETWILD) {
            dataLoadedOpt = tetMeshOpt->setTetrahedralizedGridFTetWild(
                    tetMeshGT->getBoundingBox(), settings.initGridResolution.x,
                    settings.initGridResolution.y, settings.initGridResolution.z,
                    glm::vec4(0.5f, 0.5f, 0.5f, 0.1f), settings.fTetWildParams);
        } else { // settings.initGridType == InitGridType::TETGEN
            dataLoadedOpt = tetMeshOpt->setTetrahedralizedGridTetGen(
                    tetMeshGT->getBoundingBox(), settings.initGridResolution.x,
                    settings.initGridResolution.y, settings.initGridResolution.z,
                    glm::vec4(0.5f, 0.5f, 0.5f, 0.1f), settings.tetGenParams);
        }
    } else {
        dataLoadedOpt = tetMeshOpt->loadFromFile(settings.dataSetFileNameOpt);
    }
    if (!dataLoadedGT || !dataLoadedOpt) {
        hasRequest = false;
        tetMeshGT = {};
        tetMeshOpt = {};
        return;
    }
    numCellsInit = uint32_t(tetMeshOpt->getNumCells());

    onVertexBuffersRecreated();

    // TODO: Make sure data is freed before allocating new data.
    tetMeshVolumeRendererGT->setClearColor(sgl::Color(0, 0, 0, 0));
    if (settings.useCoarseToFine) {
        tetMeshVolumeRendererGT->setCoarseToFineTargetNumTets(settings.maxNumTets);
    }
    tetMeshVolumeRendererGT->setTetMeshData(tetMeshGT);
    tetMeshVolumeRendererGT->setAttenuationCoefficient(settings.attenuationCoefficient);
    tetMeshVolumeRendererGT->setOutputImage(colorImageGT->getImageView());
    tetMeshVolumeRendererGT->recreateSwapchain(settings.imageWidth, settings.imageHeight);
    fragmentBufferSize = tetMeshVolumeRendererGT->getFragmentBufferSize();
    fragmentBuffer = tetMeshVolumeRendererGT->getFragmentBuffer();
    startOffsetBuffer = tetMeshVolumeRendererGT->getStartOffsetBuffer();
    fragmentCounterBuffer = tetMeshVolumeRendererGT->getFragmentCounterBuffer();

    tetMeshVolumeRendererOpt->setUseExternalFragmentBuffer(true);
    tetMeshVolumeRendererOpt->setClearColor(sgl::Color(0, 0, 0, 0));
    tetMeshVolumeRendererOpt->setTetMeshData(tetMeshOpt);
    tetMeshVolumeRendererOpt->setAttenuationCoefficient(settings.attenuationCoefficient);
    tetMeshVolumeRendererOpt->setOutputImage(colorImageOpt->getImageView());
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
    tetMeshRendererTypeCached = tetMeshRendererType;
    timeStart = std::chrono::system_clock::now();
}

void TetMeshOptimizer::onVertexBuffersRecreated() {
    auto vertexPositionBuffer = tetMeshOpt->getVertexPositionBuffer();
    auto vertexColorBuffer = tetMeshOpt->getVertexColorBuffer();
    auto vertexBoundaryBitBuffer = tetMeshOpt->getVertexBoundaryBitBuffer();
    auto vertexColorGradientBuffer = tetMeshOpt->getVertexColorGradientBuffer();
    auto vertexPositionGradientBuffer = tetMeshOpt->getVertexPositionGradientBuffer();
    optimizerPassPositions->setBuffers(vertexPositionBuffer, vertexPositionGradientBuffer, vertexBoundaryBitBuffer);
    optimizerPassColors->setBuffers(vertexColorBuffer, vertexColorGradientBuffer, vertexBoundaryBitBuffer);
    auto cellIndicesBuffer = tetMeshOpt->getCellIndicesBuffer();
    tetRegularizerPass->setBuffers(cellIndicesBuffer, vertexPositionBuffer, vertexPositionGradientBuffer);
    tetMeshVolumeRendererOpt->setAdjointPassData(colorAdjointTexture->getImageView(), adjointPassBackbuffer);
}

bool TetMeshOptimizer::getHasResult() {
    if (!hasRequest) {
        return false;
    }
    if (!settings.useCoarseToFine) {
        return currentEpoch == settings.maxNumEpochs;
    }
    return tetMeshOpt->getNumCells() >= size_t(settings.maxNumTets) && coarseToFineEpoch % 3 == 2;
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
    if (tetMeshRendererType == TetMeshRendererType::PPLL) {
        renderer->insertImageMemoryBarriers(
                std::vector<sgl::vk::ImagePtr>{ colorImageGT->getImage(), colorImageOpt->getImage() },
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    } else {
        renderer->insertImageMemoryBarrier(
                colorImageGT->getImage(),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        renderer->insertImageMemoryBarrier(
                colorImageOpt->getImage(),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        renderer->insertImageMemoryBarrier(
                colorImageOptPreview->getImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
        colorImageOpt->getImage()->copyToImage(
                colorImageOptPreview->getImage(), VK_IMAGE_ASPECT_COLOR_BIT, renderer->getVkCommandBuffer());
        renderer->insertImageMemoryBarrier(
                colorImageOpt->getImage(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        renderer->insertImageMemoryBarrier(
                colorImageOptPreview->getImage(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        /*renderer->insertImageMemoryBarriers(
                std::vector<sgl::vk::ImagePtr>{ colorImageGT->getImage(), colorImageOpt->getImage() },
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);*/
    }

    // Compute the image loss.
    lossPass->render();
    renderer->insertImageMemoryBarrier(
            colorAdjointTexture->getImage(),
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    const int coarseToFineEpochIndex = coarseToFineEpoch % 3;
    bool clearGrads =
            !settings.useCoarseToFine || currentEpoch == 0
            || coarseToFineEpochIndex != COARSE_TO_FINE_EPOCH_GATHER;

    // Clear the gradients.
    auto vertexPositionGradientBuffer = tetMeshOpt->getVertexPositionGradientBuffer();
    auto vertexColorGradientBuffer = tetMeshOpt->getVertexColorGradientBuffer();
    if (settings.optimizePositions && clearGrads) {
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
    if (settings.optimizeColors && clearGrads) {
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

    // Compute the gradients wrt. the tet vertices for the image loss.
    if (coarseToFineEpochIndex == COARSE_TO_FINE_EPOCH_GATHER
            && (settings.splitGradientType == SplitGradientType::ABS_POSITION
                    || settings.splitGradientType == SplitGradientType::ABS_COLOR)) {
        tetMeshVolumeRendererOpt->setUseAbsGrad(true);
    }
    tetMeshVolumeRendererOpt->renderAdjoint();
    tetMeshVolumeRendererOpt->setUseAbsGrad(false);
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

    // TODO
    bool writeNow =
            (!settings.useCoarseToFine && settings.optimizePositions)
            || (settings.useCoarseToFine && coarseToFineEpochIndex == COARSE_TO_FINE_EPOCH_GATHER
                    && currentEpoch == settings.maxNumEpochs - 1);
    if (settings.exportPositionGradients && writeNow) {
        const auto& cellIndices = tetMeshOpt->getCellIndices();
        auto vertexPositionBuffer = tetMeshOpt->getVertexPositionBuffer();
        auto vertexColorBuffer = tetMeshOpt->getVertexColorBuffer();
        if (!vertexPositionStagingBuffer
                || vertexPositionStagingBuffer->getSizeInBytes() != vertexPositionBuffer->getSizeInBytes()) {
            vertexPositionStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), vertexPositionBuffer->getSizeInBytes(),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        }
        if (!vertexColorStagingBuffer
                || vertexColorStagingBuffer->getSizeInBytes() != vertexColorBuffer->getSizeInBytes()) {
            vertexColorStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), vertexColorBuffer->getSizeInBytes(),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        }
        if (!vertexPositionGradientStagingBuffer
                || vertexPositionGradientStagingBuffer->getSizeInBytes() != vertexPositionGradientBuffer->getSizeInBytes()) {
            vertexPositionGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), vertexPositionGradientBuffer->getSizeInBytes(),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        }
        if (!vertexColorGradientStagingBuffer
                || vertexColorGradientStagingBuffer->getSizeInBytes() != vertexColorGradientBuffer->getSizeInBytes()) {
            vertexColorGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), vertexColorGradientBuffer->getSizeInBytes(),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        }
        renderer->insertMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        vertexPositionBuffer->copyDataTo(
                vertexPositionStagingBuffer, 0, 0, vertexPositionStagingBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        vertexColorBuffer->copyDataTo(
                vertexColorStagingBuffer, 0, 0, vertexColorStagingBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        vertexPositionGradientBuffer->copyDataTo(
                vertexPositionGradientStagingBuffer, 0, 0, vertexPositionGradientBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        vertexColorGradientBuffer->copyDataTo(
                vertexColorGradientStagingBuffer, 0, 0, vertexColorGradientBuffer->getSizeInBytes(),
                renderer->getVkCommandBuffer());
        renderer->syncWithCpu();

        auto* vertexPositions = reinterpret_cast<glm::vec3*>(vertexPositionStagingBuffer->mapMemory());
        auto* vertexColors = reinterpret_cast<glm::vec4*>(vertexColorStagingBuffer->mapMemory());
        auto* vertexPositionGradients = reinterpret_cast<glm::vec3*>(vertexPositionGradientStagingBuffer->mapMemory());
        auto* vertexColorGradients = reinterpret_cast<glm::vec4*>(vertexColorGradientStagingBuffer->mapMemory());
        const auto numVertices = int(vertexPositionBuffer->getSizeInBytes() / sizeof(glm::vec3));
        vtkWriter->writeNextTimeStep(
                cellIndices, vertexPositions, vertexColors,
                vertexPositionGradients, vertexColorGradients, numVertices);
        vertexPositionStagingBuffer->unmapMemory();
        vertexColorStagingBuffer->unmapMemory();
        vertexPositionGradientStagingBuffer->unmapMemory();
        vertexColorGradientStagingBuffer->unmapMemory();
    }

    /*bool debugOutput = true;
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
    }

    if (debugOutput) {
        auto vertexPositionBuffer = tetMeshOpt->getVertexPositionBuffer();
        auto vertexColorBuffer = tetMeshOpt->getVertexColorBuffer();
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
        std::cout << "vertexPositions (A):" << std::endl;
        for (uint32_t i = 0; i < numEntriesPos; i++) {
            std::cout << vertexPositions[i] << std::endl;
        }
        std::cout << std::endl << "vertexColors (A):" << std::endl;
        for (uint32_t i = 0; i < numEntriesCol; i++) {
            std::cout << vertexColors[i] << std::endl;
        }
        std::cout << std::endl;
        vertexPositionStagingBuffer->unmapMemory();
        vertexColorStagingBuffer->unmapMemory();
    }*/

    // Run the optimizer.
    bool optimizePos =
            !settings.useCoarseToFine || coarseToFineEpochIndex == COARSE_TO_FINE_EPOCH_COLOR_POS;
    bool optimizeCol =
            !settings.useCoarseToFine || coarseToFineEpochIndex != COARSE_TO_FINE_EPOCH_GATHER;
    if (settings.optimizePositions && optimizePos) {
        //const int epochNum = (coarseToFineEpoch / 3) * settings.maxNumEpochs + currentEpoch;
        // Restart from epoch 0 when buffers are recreated.
        const int epochNum = currentEpoch;
        optimizerPassPositions->setEpochIndex(epochNum);
        optimizerPassPositions->render();
    }
    if (settings.optimizeColors && optimizeCol) {
        // Restart from epoch 0 when buffers are recreated.
        //const int epochNum = (coarseToFineEpoch / 3 * 2 + coarseToFineEpochIndex) * settings.maxNumEpochs + currentEpoch;
        const int epochNum = currentEpoch;
        optimizerPassColors->setEpochIndex(epochNum);
        optimizerPassColors->render();
    }
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    /*if (debugOutput) {
        auto vertexPositionBuffer = tetMeshOpt->getVertexPositionBuffer();
        auto vertexColorBuffer = tetMeshOpt->getVertexColorBuffer();
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
        std::cout << "vertexPositions (B):" << std::endl;
        for (uint32_t i = 0; i < numEntriesPos; i++) {
            std::cout << vertexPositions[i] << std::endl;
        }
        std::cout << std::endl << "vertexColors (B):" << std::endl;
        for (uint32_t i = 0; i < numEntriesCol; i++) {
            std::cout << vertexColors[i] << std::endl;
        }
        std::cout << std::endl;
        vertexPositionStagingBuffer->unmapMemory();
        vertexColorStagingBuffer->unmapMemory();
    }*/

    currentEpoch++;

    if (settings.useCoarseToFine && currentEpoch == settings.maxNumEpochs
            && (tetMeshOpt->getNumCells() < size_t(settings.maxNumTets) || coarseToFineEpoch % 3 < 2)) {
        coarseToFineEpoch++;
        currentEpoch = 0;
        if (coarseToFineEpochIndex == 0) {
            const auto tetFactor = float(double(numCellsInit) / double(tetMeshOpt->getNumCells()));
            float lrPos = settings.optimizerSettingsPositions.learningRate;
            lrPos *= std::pow(settings.optimizerSettingsPositions.lrDecayRate, float(coarseToFineEpoch / 3));
            lrPos *= tetFactor;
            float lrCol = settings.optimizerSettingsColors.learningRate;
            lrCol *= std::pow(settings.optimizerSettingsColors.lrDecayRate, float(coarseToFineEpoch / 3));
            lrCol *= tetFactor;
            std::cout << "lrCol: " << lrCol << ", lrPos: " << lrPos << std::endl;
            optimizerPassPositions->setSettings(
                    settings.lossType, lrPos,
                    settings.optimizerSettingsPositions.beta1, settings.optimizerSettingsPositions.beta2,
                    settings.optimizerSettingsPositions.epsilon, false, settings.fixBoundary);
            optimizerPassColors->setSettings(
                    settings.lossType, lrCol,
                    settings.optimizerSettingsColors.beta1, settings.optimizerSettingsColors.beta2,
                    settings.optimizerSettingsColors.epsilon, true, settings.fixBoundary);
            optimizerPassPositions->updateUniformBuffer();
            optimizerPassColors->updateUniformBuffer();
        }
        if (coarseToFineEpochIndex == COARSE_TO_FINE_EPOCH_GATHER) {
            renderer->insertMemoryBarrier(
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            tetMeshOpt->splitByLargestGradientMagnitudes(renderer, settings.splitGradientType, settings.numSplitsRatio);
            tetMeshVolumeRendererOpt->setTetMeshData(tetMeshOpt);

            // Set new vertex and color and respective gradient buffers.
            std::cout << "Recreate" << std::endl;
            onVertexBuffersRecreated();
        }
    }
    if (!settings.useCoarseToFine && settings.optimizerSettingsPositions.lrDecayRate != 1.0f) {
        float lrPos = settings.optimizerSettingsPositions.learningRate;
        lrPos *= std::pow(settings.optimizerSettingsPositions.lrDecayRate, float(currentEpoch));
        std::cout << "lrPos: " << lrPos << std::endl;
        optimizerPassPositions->setSettings(
                settings.lossType, lrPos,
                settings.optimizerSettingsPositions.beta1, settings.optimizerSettingsPositions.beta2,
                settings.optimizerSettingsPositions.epsilon, false, settings.fixBoundary);
        optimizerPassPositions->updateUniformBuffer();
    }
    if (!settings.useCoarseToFine && settings.optimizerSettingsColors.lrDecayRate != 1.0f) {
        float lrCol = settings.optimizerSettingsColors.learningRate;
        lrCol *= std::pow(settings.optimizerSettingsColors.lrDecayRate, float(currentEpoch));
        std::cout << "lrCol: " << lrCol << std::endl;
        optimizerPassColors->setSettings(
                settings.lossType, lrCol,
                settings.optimizerSettingsColors.beta1, settings.optimizerSettingsColors.beta2,
                settings.optimizerSettingsColors.epsilon, true, settings.fixBoundary);
        optimizerPassColors->updateUniformBuffer();
    }
}
