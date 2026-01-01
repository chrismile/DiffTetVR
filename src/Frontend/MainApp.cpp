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

#include <stack>
#include <csignal>

#include <Utils/StringUtils.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/Dialog.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/Events/EventManager.hpp>
#include <Input/Keyboard.hpp>
#include <Graphics/Window.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Utils/DeviceSelectionVulkan.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>
#include <ImGui/imgui_internal.h>
#include <ImGui/imgui_custom.h>
#include <ImGui/imgui_stdlib.h>

#include "Tet/TetMesh.hpp"
#include "Tet/Loaders/DataSetList.hpp"
#include "Tet/Meshing/TetMeshing.hpp"
#include "Renderer/Optimizer.hpp"
#include "Renderer/TetMeshRendererPPLL.hpp"
#include "Renderer/TetMeshRendererProjection.hpp"
#include "Renderer/TetMeshRendererIntersection.hpp"
#include "DataView.hpp"
#include "MainApp.hpp"

void vulkanErrorCallback() {
#ifdef SGL_INPUT_API_V2
    sgl::AppSettings::get()->captureMouse(false);
#else
    SDL_CaptureMouse(SDL_FALSE);
#endif
    std::cerr << "Application callback" << std::endl;
}

#ifdef __linux__
void signalHandler(int signum) {
#ifdef SGL_INPUT_API_V2
    sgl::AppSettings::get()->captureMouse(false);
#else
    SDL_CaptureMouse(SDL_FALSE);
#endif
    std::cerr << "Interrupt signal (" << signum << ") received." << std::endl;
    exit(signum);
}
#endif

MainApp::MainApp()
        : sceneData(
            camera, clearColor, screenshotTransparentBackground, recording, useCameraFlight) {
    sgl::AppSettings::get()->getVulkanInstance()->setDebugCallback(&vulkanErrorCallback);
    clearColor = sgl::Color(0, 0, 0, 255);
    clearColorSelection = ImColor(0, 0, 0, 255);
    std::string clearColorString;
    if (sgl::AppSettings::get()->getSettings().getValueOpt("clearColor", clearColorString)) {
        std::vector<std::string> clearColorStringParts;
        sgl::splitString(clearColorString, ',', clearColorStringParts);
        if (clearColorStringParts.size() == 3 || clearColorStringParts.size() == 4) {
            clearColor.setR(uint8_t(sgl::fromString<int>(clearColorStringParts.at(0))));
            clearColor.setG(uint8_t(sgl::fromString<int>(clearColorStringParts.at(1))));
            clearColor.setB(uint8_t(sgl::fromString<int>(clearColorStringParts.at(2))));
            clearColorSelection = ImColor(clearColor.getR(), clearColor.getG(), clearColor.getB(), 255);
        }
    }

    //viewManager = new ViewManager(&clearColor, rendererVk);

    checkpointWindow.setStandardWindowSize(1254, 390);
    checkpointWindow.setStandardWindowPosition(841, 53);

    propertyEditor.setInitWidthValues(sgl::ImGuiWrapper::get()->getScaleDependentSize(280.0f));

    camera->setNearClipDistance(0.01f);
    camera->setFarClipDistance(100.0f);

    useDockSpaceMode = true;
    sgl::AppSettings::get()->getSettings().getValueOpt("useDockSpaceMode", useDockSpaceMode);
    sgl::AppSettings::get()->getSettings().getValueOpt("useFixedSizeViewport", useFixedSizeViewport);
    sgl::AppSettings::get()->getSettings().getValueOpt("fixedViewportSizeX", fixedViewportSize.x);
    sgl::AppSettings::get()->getSettings().getValueOpt("fixedViewportSizeY", fixedViewportSize.y);
    fixedViewportSizeEdit = fixedViewportSize;
    showPropertyEditor = true;
    sgl::ImGuiWrapper::get()->setUseDockSpaceMode(useDockSpaceMode);
    //useDockSpaceMode = false;

#ifdef NDEBUG
    showFpsOverlay = false;
#else
    showFpsOverlay = true;
#endif
    sgl::AppSettings::get()->getSettings().getValueOpt("showFpsOverlay", showFpsOverlay);
    sgl::AppSettings::get()->getSettings().getValueOpt("showCoordinateAxesOverlay", showCoordinateAxesOverlay);

    if (usePerformanceMeasurementMode) {
        useCameraFlight = true;
    }
    if (useCameraFlight && recording) {
        sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
        window->setWindowSize(recordingResolution.x, recordingResolution.y);
        realTimeCameraFlight = false;
    }

    fileDialogInstance = IGFD_Create();
    customDataSetFileName = sgl::FileUtils::get()->getUserDirectory();
    loadAvailableDataSetInformation();
    isProgramStart = false;

    const std::string tetMeshDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "DataSets/";
    sgl::FileUtils::get()->ensureDirectoryExists(tetMeshDataSetsDirectory);
#ifdef __linux__
    datasetsWatch.setPath(tetMeshDataSetsDirectory + "datasets.json", false);
    datasetsWatch.initialize();
#endif

    useLinearRGB = false;
    onClearColorChanged();
    deviceSelector = device->getDeviceSelector();

    createTetMeshRenderer();
    dataView = std::make_shared<DataView>(camera, rendererVk, tetMeshVolumeRenderer);
    dataView->useLinearRGB = useLinearRGB;
    if (useDockSpaceMode) {
        cameraHandle = dataView->camera;
    } else {
        cameraHandle = camera;
    }

    resolutionChanged(sgl::EventPtr());

    if (!recording && !usePerformanceMeasurementMode) {
        // Just for convenience...
        int desktopWidth = 0;
        int desktopHeight = 0;
        int refreshRate = 60;
        sgl::AppSettings::get()->getDesktopDisplayMode(desktopWidth, desktopHeight, refreshRate);
        sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
        if (desktopWidth == 3840 && desktopHeight == 2160 && !window->getWindowSettings().isMaximized) {
            window->setWindowSize(2186, 1358);
        }
    }
    if (!sgl::AppSettings::get()->getSettings().hasKey("cameraNavigationMode")) {
        cameraNavigationMode = sgl::CameraNavigationMode::TURNTABLE;
        updateCameraNavigationMode();
    }

    tetMeshOptimizer = new TetMeshOptimizer(
            rendererVk, [this](const TetMeshPtr& _tetMesh, float _attenuationCoefficient) {
                tetMesh = _tetMesh;
                tetMeshVolumeRenderer->setAttenuationCoefficient(_attenuationCoefficient);
                tetMeshVolumeRenderer->setTetMeshData(tetMesh);
            },
            dataSetNames.size() > 1, [this]() -> std::string {
                int i = this->renderGuiDataSetSelectionMenu();
                if (i >= 0) {
                    return dataSetInformationList.at(i - NUM_MANUAL_LOADERS)->filenames.front();
                } else {
                    return "";
                }
            }, &transferFunctionWindow);
    tetMeshOptimizer->setTetMeshRendererType(tetMeshRendererType);

#ifdef __linux__
    signal(SIGSEGV, signalHandler);
#endif
}

MainApp::~MainApp() {
    device->waitIdle();

    delete tetMeshOptimizer;
    tetMeshVolumeRenderer = {};
    dataView = {};

    /*delete tfOptimization;
    volumeRenderers = {};
    volumeData = {};
    dataViews.clear();
    delete viewManager;
    viewManager = nullptr;*/

    IGFD_Destroy(fileDialogInstance);

/*#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::getIsNvrtcFunctionTableInitialized()) {
        sgl::freeNvrtcFunctionTable();
    }
    if (sgl::getIsCudaDeviceApiFunctionTableInitialized()) {
        if (cuContext) {
            CUresult cuResult = sgl::g_cudaDeviceApiFunctionTable.cuCtxDestroy(cuContext);
            sgl::checkCUresult(cuResult, "Error in cuCtxDestroy: ");
            cuContext = {};
        }
        sgl::freeCudaDeviceApiFunctionTable();
    }
#endif*/

    for (int i = 0; i < int(nonBlockingMsgBoxHandles.size()); i++) {
        auto& handle = nonBlockingMsgBoxHandles.at(i);
        if (handle->ready(0)) {
            nonBlockingMsgBoxHandles.erase(nonBlockingMsgBoxHandles.begin() + i);
            i--;
        } else {
            handle->kill();
        }
    }
    nonBlockingMsgBoxHandles.clear();

    std::string clearColorString =
            std::to_string(int(clearColor.getR())) + ","
            + std::to_string(int(clearColor.getG())) + ","
            + std::to_string(int(clearColor.getB()));
    sgl::AppSettings::get()->getSettings().addKeyValue("clearColor", clearColorString);
    sgl::AppSettings::get()->getSettings().addKeyValue("useDockSpaceMode", useDockSpaceMode);
    if (!usePerformanceMeasurementMode) {
        sgl::AppSettings::get()->getSettings().addKeyValue("useFixedSizeViewport", useFixedSizeViewport);
        sgl::AppSettings::get()->getSettings().addKeyValue("fixedViewportSizeX", fixedViewportSize.x);
        sgl::AppSettings::get()->getSettings().addKeyValue("fixedViewportSizeY", fixedViewportSize.y);
    }
    sgl::AppSettings::get()->getSettings().addKeyValue("showFpsOverlay", showFpsOverlay);
    sgl::AppSettings::get()->getSettings().addKeyValue("showCoordinateAxesOverlay", showCoordinateAxesOverlay);
}

void MainApp::createTetMeshRenderer() {
    float oldAttenuation = 0.0f;
    if (tetMeshVolumeRenderer) {
        oldAttenuation = tetMeshVolumeRenderer->getAttenuationCoefficient();
    }
    tetMeshVolumeRenderer = {};
    if (tetMeshRendererType == TetMeshRendererType::PPLL) {
        tetMeshVolumeRenderer = std::make_shared<TetMeshRendererPPLL>(
                rendererVk, &cameraHandle, &transferFunctionWindow);
    } else if (tetMeshRendererType == TetMeshRendererType::PROJECTION) {
        tetMeshVolumeRenderer = std::make_shared<TetMeshRendererProjection>(
                rendererVk, &cameraHandle, &transferFunctionWindow);
    } else if (tetMeshRendererType == TetMeshRendererType::INTERSECTION) {
        tetMeshVolumeRenderer = std::make_shared<TetMeshRendererIntersection>(
                rendererVk, &cameraHandle, &transferFunctionWindow);
    }
    tetMeshVolumeRenderer->setUseLinearRGB(useLinearRGB);
    tetMeshVolumeRenderer->setClearColor(clearColor);
    if (oldAttenuation > 0.0f) {
        tetMeshVolumeRenderer->setAttenuationCoefficient(oldAttenuation);
    }
    //tetMeshVolumeRenderer->setFileDialogInstance(fileDialogInstance);
    if (tetMesh) {
        tetMeshVolumeRenderer->setTetMeshData(tetMesh);
    }
    if (!useDockSpaceMode) {
        sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
        auto width = uint32_t(window->getWidth());
        auto height = uint32_t(window->getHeight());
        if (width > 0 && height > 0 && sceneTextureVk) {
            tetMeshVolumeRenderer->setOutputImage(sceneTextureVk->getImageView());
            tetMeshVolumeRenderer->recreateSwapchain(width, height);
        }
    }
    if (dataView) {
        dataView->setTetMeshVolumeRenderer(tetMeshVolumeRenderer);
    }
    if (tetMeshOptimizer) {
        tetMeshOptimizer->setTetMeshRendererType(tetMeshRendererType);
    }
}

void MainApp::resolutionChanged(sgl::EventPtr event) {
    SciVisApp::resolutionChanged(event);

    sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
    auto width = uint32_t(window->getWidth());
    auto height = uint32_t(window->getHeight());

    if (!useDockSpaceMode) {
        tetMeshVolumeRenderer->setOutputImage(sceneTextureVk->getImageView());
        tetMeshVolumeRenderer->recreateSwapchain(width, height);
    }
}

void MainApp::updateColorSpaceMode() {
    SciVisApp::updateColorSpaceMode();
    tetMeshVolumeRenderer->setUseLinearRGB(useLinearRGB);
    if (dataView) {
        dataView->useLinearRGB = useLinearRGB;
        dataView->viewportWidth = 0;
        dataView->viewportHeight = 0;
    }
}

void MainApp::render() {
    // Test Code.
    static bool isFirstFrame = true;
    if (isFirstFrame) {
        //selectedDataSetIndex = 0;
        //customDataSetFileName = "/home/christoph/datasets/Toy/chord/linear_4x4.nc";
        //loadVolumeDataSet({ customDataSetFileName });
        tetMesh = std::make_shared<TetMesh>(device, &transferFunctionWindow);
        tetMesh->loadTestData(TestCase::SINGLE_TETRAHEDRON);
        tetMeshVolumeRenderer->setTetMeshData(tetMesh);
        isFirstFrame = false;
        isStartupFile = true;
        boundingBox = tetMesh->getBoundingBox();
        cameraPath.fromCirclePath(
                boundingBox, "",
                usePerformanceMeasurementMode
                ? CAMERA_PATH_TIME_PERFORMANCE_MEASUREMENT : CAMERA_PATH_TIME_RECORDING,
                usePerformanceMeasurementMode);
    }

    SciVisApp::preRender();
    if (dataView) {
        dataView->saveScreenshotDataIfAvailable();
    }

    if (!useDockSpaceMode) {
        reRender = reRender || tetMeshVolumeRenderer->needsReRender();

        if (reRender || continuousRendering) {
            SciVisApp::prepareReRender();

            if (tetMesh) {
                rendererVk->setProjectionMatrix(camera->getProjectionMatrix());
                rendererVk->setViewMatrix(camera->getViewMatrix());
                rendererVk->setModelMatrix(sgl::matrixIdentity());
                tetMeshVolumeRenderer->render();
            }

            reRender = false;
        }
    }

    SciVisApp::postRender();

    if (useDockSpaceMode && !uiOnScreenshot && recording && !isFirstRecordingFrame) {
        rendererVk->transitionImageLayout(
                dataView->compositedDataViewTexture->getImage(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        videoWriter->pushFramebufferImage(dataView->compositedDataViewTexture->getImage());
        rendererVk->transitionImageLayout(
                dataView->compositedDataViewTexture->getImage(),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
}

void MainApp::renderGui() {
    focusedWindowIndex = -1;
    mouseHoverWindowIndex = -1;

#ifdef SGL_INPUT_API_V2
    if (sgl::Keyboard->keyPressed(ImGuiKey_O) && sgl::Keyboard->getModifier(ImGuiKey_ModCtrl)) {
        openFileDialog();
    }
    if (sgl::Keyboard->keyPressed(ImGuiKey_S) && sgl::Keyboard->getModifier(ImGuiKey_ModCtrl)) {
        openSaveTetMeshFileDialog();
    }
#else
    if (sgl::Keyboard->keyPressed(SDLK_o) && (sgl::Keyboard->getModifier() & (KMOD_LCTRL | KMOD_RCTRL)) != 0) {
        openFileDialog();
    }
    if (sgl::Keyboard->keyPressed(SDLK_s) && (sgl::Keyboard->getModifier() & (KMOD_LCTRL | KMOD_RCTRL)) != 0) {
        openSaveTetMeshFileDialog();
    }
#endif

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseDataSetFile", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathNameString(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilterString(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            std::string currentPath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            if (selection.count != 0) {
                filename += selection.table[0].fileName;
            }
            IGFD_Selection_DestroyContent(&selection);

            std::string filenameLower = sgl::toLowerCopy(filename);
            if (checkHasValidExtension(filenameLower)) {
                fileDialogDirectory = sgl::FileUtils::get()->getPathToFile(filename);
                selectedDataSetIndex = 0;
                customDataSetFileName = filename;
                isStartupFile = false;
                loadTetMeshDataSet(getSelectedDataSetFilename());
            } else {
                sgl::Logfile::get()->writeError(
                        "The selected file name has an unknown extension \""
                        + sgl::FileUtils::get()->getFileExtension(filenameLower) + "\".");
            }
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseSaveFile", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathNameString(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilterString(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            std::string currentPath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            std::string currentFileName;
            if (filter == ".*") {
                currentFileName = IGFD_GetCurrentFileNameRawString(fileDialogInstance);
            } else {
                currentFileName = IGFD_GetCurrentFileNameString(fileDialogInstance);
            }
            if (selection.count != 0 && selection.table[0].fileName == currentFileName) {
                filename += selection.table[0].fileName;
            } else {
                filename += currentFileName;
            }
            IGFD_Selection_DestroyContent(&selection);

            saveTestMeshFileDialogDirectory = sgl::FileUtils::get()->getPathToFile(filename);

            std::string filenameLower = sgl::toLowerCopy(filename);
            if (sgl::endsWith(filenameLower, ".bintet")
#ifdef USE_OPEN_VOLUME_MESH
                    || sgl::endsWith(filenameLower, ".ovm")
                    || sgl::endsWith(filenameLower, ".ovmb")
#endif
                    || sgl::endsWith(filenameLower, ".txt")) {
                tetMesh->saveToFile(filename);
            } else {
                sgl::Logfile::get()->writeError(
                        "The selected file name has an unsupported extension \""
                        + sgl::FileUtils::get()->getFileExtension(filenameLower) + "\".");
            }
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    if (useDockSpaceMode) {
        static bool isProgramStartup = true;
        ImGuiID dockSpaceId = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
        if (isProgramStartup) {
            ImGuiDockNode* centralNode = ImGui::DockBuilderGetNode(dockSpaceId);
            if (centralNode->IsEmpty()) {
                auto* window = sgl::AppSettings::get()->getMainWindow();
                const ImVec2 dockSpaceSize(float(window->getWidth()), float(window->getHeight()));
                ImGui::DockBuilderSetNodeSize(dockSpaceId, dockSpaceSize);

                ImGuiID dockLeftId, dockMainId;
                ImGui::DockBuilderSplitNode(
                        dockSpaceId, ImGuiDir_Left, 0.29f, &dockLeftId, &dockMainId);
                ImGui::DockBuilderSetNodeSize(dockLeftId, ImVec2(dockSpaceSize.x * 0.29f, dockSpaceSize.y));
                ImGui::DockBuilderDockWindow("Tet Mesh Volume Renderer", dockMainId);

                ImGuiID dockLeftUpId, dockLeftDownId;
                ImGui::DockBuilderSplitNode(
                        dockLeftId, ImGuiDir_Up, 0.8f,
                        &dockLeftUpId, &dockLeftDownId);
                ImGui::DockBuilderDockWindow("Property Editor", dockLeftUpId);

                ImGui::DockBuilderDockWindow("Transfer Function", dockLeftDownId);
                ImGui::DockBuilderDockWindow("Camera Checkpoints", dockLeftDownId);

                ImGui::DockBuilderFinish(dockLeftId);
                ImGui::DockBuilderFinish(dockSpaceId);
            }
            isProgramStartup = false;
        }

        renderGuiMenuBar();

        if (showRendererWindow) {
            bool isViewOpen = true;
            sgl::ImGuiWrapper::get()->setNextWindowStandardSize(800, 600);
            if (ImGui::Begin("Tet Mesh Volume Renderer", &isViewOpen)) {
                if (ImGui::IsWindowFocused()) {
                    focusedWindowIndex = 0;
                }
                sgl::ImGuiWrapper::get()->setWindowViewport(0, ImGui::GetWindowViewport());
                sgl::ImGuiWrapper::get()->setWindowViewport(0, ImGui::GetWindowViewport());
                sgl::ImGuiWrapper::get()->setWindowPosAndSize(0, ImGui::GetWindowPos(), ImGui::GetWindowSize());

                ImVec2 sizeContent = ImGui::GetContentRegionAvail();
                if (useFixedSizeViewport) {
                    sizeContent = ImVec2(float(fixedViewportSize.x), float(fixedViewportSize.y));
                }
                if (int(sizeContent.x) != int(dataView->viewportWidth)
                        || int(sizeContent.y) != int(dataView->viewportHeight)
                        || tetMeshVolumeRenderer->getRendererType() != dataView->cachedRendererType) {
                    dataView->resize(int(sizeContent.x), int(sizeContent.y));
                    // Handled by resize().
                    //if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0) {
                    //    tetMeshVolumeRenderer->setOutputImage(dataView->dataViewTexture->getImageView());
                    //    tetMeshVolumeRenderer->recreateSwapchain(
                    //            dataView->viewportWidth, dataView->viewportHeight);
                    //}
                    dataView->cachedRendererType = tetMeshVolumeRenderer->getRendererType();
                    reRender = true;
                }

                reRender = reRender || tetMeshVolumeRenderer->needsReRender();

                if (reRender || continuousRendering) {
                    if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0) {
                        dataView->beginRender();
                        if (tetMesh) {
                            tetMeshVolumeRenderer->render();
                        }
                        dataView->endRender();
                    }

                    reRender = false;
                }

                if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0) {
                    if (!uiOnScreenshot && screenshot) {
                        printNow = true;
                        std::string screenshotFilename =
                                saveDirectoryScreenshots + saveFilenameScreenshots
                                + "_" + sgl::toString(screenshotNumber);
                        screenshotFilename += ".png";

                        dataView->screenshotReadbackHelper->setScreenshotTransparentBackground(
                                screenshotTransparentBackground);
                        dataView->saveScreenshot(screenshotFilename);
                        screenshot = false;

                        printNow = false;
                        screenshot = true;
                    }

                    if (isViewOpen) {
                        ImTextureID textureId = dataView->getImGuiTextureId();
                        ImGui::Image(
                                textureId, sizeContent, ImVec2(0, 0), ImVec2(1, 1));
                        if (ImGui::IsItemHovered()) {
                            mouseHoverWindowIndex = 0;
                        }
                    }

                    if (showFpsOverlay) {
                        renderGuiFpsOverlay();
                    }
                    if (showCoordinateAxesOverlay) {
                        renderGuiCoordinateAxesOverlay(dataView->camera);
                    }
                }
            }
            ImGui::End();
        }

        if (!uiOnScreenshot && screenshot) {
            screenshot = false;
            screenshotNumber++;
        }
        reRender = false;
    }

    if (tetMeshVolumeRenderer->getShowTetQuality() && transferFunctionWindow.renderGui()) {
        reRender = true;
        if (transferFunctionWindow.getTransferFunctionMapRebuilt()) {
            tetMeshVolumeRenderer->onTransferFunctionMapRebuilt();
            sgl::EventManager::get()->triggerEvent(std::make_shared<sgl::Event>(
                    ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT));
        }
    }

    if (checkpointWindow.renderGui()) {
        fovDegree = camera->getFOVy() / sgl::PI * 180.0f;
        reRender = true;
        hasMoved();
    }

    if (showPropertyEditor) {
        renderGuiPropertyEditorWindow();
    }
}

void MainApp::onClearColorChanged() {
    clearColor.setA(screenshotTransparentBackground ? 0 : 255);
    transferFunctionWindow.setClearColor(clearColor);
    coordinateAxesOverlayWidget.setClearColor(clearColor);
    if (tetMeshVolumeRenderer) {
        rendererVk->syncWithCpu();
        tetMeshVolumeRenderer->setClearColor(clearColor);
    }
    reRender = true;
}

void MainApp::renderGuiGeneralSettingsPropertyEditor() {
    if (propertyEditor.addColorEdit3("Clear Color", (float*)&clearColorSelection, 0)) {
        clearColor = sgl::colorFromFloat(
                clearColorSelection.x, clearColorSelection.y, clearColorSelection.z, clearColorSelection.w);
        onClearColorChanged();
    }

    newDockSpaceMode = useDockSpaceMode;
    if (propertyEditor.addCheckbox("Use Docking Mode", &newDockSpaceMode)) {
        scheduledDockSpaceModeChange = true;
    }

    if (propertyEditor.addCheckbox("Fixed Size Viewport", &useFixedSizeViewport)) {
        reRender = true;
    }
    if (useFixedSizeViewport) {
        if (propertyEditor.addSliderInt2Edit("Viewport Size", &fixedViewportSizeEdit.x, 1, 8192)
                == ImGui::EditMode::INPUT_FINISHED) {
            fixedViewportSize = fixedViewportSizeEdit;
            reRender = true;
        }
    }
}

void MainApp::loadAvailableDataSetInformation() {
    const std::string tetMeshDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "DataSets/";
    bool datasetsJsonExists = sgl::FileUtils::get()->exists(tetMeshDataSetsDirectory + "datasets.json");
    DataSetInformationPtr dataSetInformationRootNew;
    if (datasetsJsonExists) {
        dataSetInformationRootNew = loadDataSetList(tetMeshDataSetsDirectory + "datasets.json", isFileWatchReload);
        if (!dataSetInformationRootNew && !isProgramStart) {
            return;
        }
    } else if (!isProgramStart) {
        return;
    }

    if (!isProgramStart && currentlyLoadedDataSetIndex >= NUM_MANUAL_LOADERS) {
        customDataSetFileName = dataSetInformationList.at(currentlyLoadedDataSetIndex - NUM_MANUAL_LOADERS)->filenames.front();
        selectedDataSetIndex = 0;
        currentlyLoadedDataSetIndex = 0;
    }

    dataSetInformationRoot = {};
    dataSetInformationList.clear();
    dataSetNames.clear();
    dataSetNames.emplace_back("Local file...");
    selectedDataSetIndex = 0;

    dataSetInformationRoot = dataSetInformationRootNew;
    if (!datasetsJsonExists || !dataSetInformationRoot) {
        return;
    }

    std::stack<std::pair<DataSetInformationPtr, size_t>> dataSetInformationStack;
    dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, 0));
    while (!dataSetInformationStack.empty()) {
        std::pair<DataSetInformationPtr, size_t> dataSetIdxPair = dataSetInformationStack.top();
        DataSetInformationPtr dataSetInformationParent = dataSetIdxPair.first;
        size_t idx = dataSetIdxPair.second;
        dataSetInformationStack.pop();
        while (idx < dataSetInformationParent->children.size()) {
            DataSetInformationPtr dataSetInformationChild =
                    dataSetInformationParent->children.at(idx);
            idx++;
            if (dataSetInformationChild->type == DataSetType::NODE) {
                dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, idx));
                dataSetInformationStack.push(std::make_pair(dataSetInformationChild, 0));
                break;
            } else {
                dataSetInformationChild->sequentialIndex = int(dataSetNames.size());
                dataSetInformationList.push_back(dataSetInformationChild);
                dataSetNames.push_back(dataSetInformationChild->name);
            }
        }
    }
}

std::string MainApp::getSelectedDataSetFilename() {
    std::vector<std::string> filenames;
    if (selectedDataSetIndex == 0) {
        filenames.push_back(customDataSetFileName);
    } else {
        for (const std::string& filename : dataSetInformationList.at(
                selectedDataSetIndex - NUM_MANUAL_LOADERS)->filenames) {
            filenames.push_back(filename);
        }
    }
    return filenames.front();
}

void MainApp::openFileDialog() {
    selectedDataSetIndex = 0;
    if (fileDialogDirectory.empty() || !sgl::FileUtils::get()->directoryExists(fileDialogDirectory)) {
        fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory() + "DataSets/";
        if (!sgl::FileUtils::get()->exists(fileDialogDirectory)) {
            fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory();
        }
    }
    IGFD_OpenModal(
            fileDialogInstance,
            "ChooseDataSetFile", "Choose a File",
#ifdef USE_OPEN_VOLUME_MESH
            ".*,.bintet,.txt,.ovm,.ovmb,.vtk,.msh,.mesh",
#else
           ".*,.bintet,.txt,.msh,.mesh",
#endif
            fileDialogDirectory.c_str(),
            "", 1, nullptr,
            ImGuiFileDialogFlags_None);
}

void MainApp::openSaveTetMeshFileDialog() {
    if (saveTestMeshFileDialogDirectory.empty() || !sgl::FileUtils::get()->directoryExists(saveTestMeshFileDialogDirectory)) {
        saveTestMeshFileDialogDirectory = sgl::AppSettings::get()->getDataDirectory() + "DataSets/";
        if (!sgl::FileUtils::get()->exists(saveTestMeshFileDialogDirectory)) {
            saveTestMeshFileDialogDirectory = sgl::AppSettings::get()->getDataDirectory();
        }
    }
    IGFD_OpenModal(
            fileDialogInstance,
            "ChooseSaveFile", "Choose a File",
#ifdef USE_OPEN_VOLUME_MESH
            ".*,.bintet,.txt,.ovm,.ovmb",
#else
            ".*,.bintet,.txt",
#endif
            saveTestMeshFileDialogDirectory.c_str(),
            "", 1, nullptr,
            ImGuiFileDialogFlags_ConfirmOverwrite);
}

int MainApp::renderGuiDataSetSelectionMenu() {
    int selectedDataSetIndexLocal = -1;
    if (dataSetInformationRoot) {
        std::stack<std::pair<DataSetInformationPtr, size_t>> dataSetInformationStack;
        dataSetInformationStack.emplace(dataSetInformationRoot, 0);
        while (!dataSetInformationStack.empty()) {
            std::pair<DataSetInformationPtr, size_t> dataSetIdxPair = dataSetInformationStack.top();
            DataSetInformationPtr dataSetInformationParent = dataSetIdxPair.first;
            size_t idx = dataSetIdxPair.second;
            dataSetInformationStack.pop();
            while (idx < dataSetInformationParent->children.size()) {
                DataSetInformationPtr dataSetInformationChild =
                        dataSetInformationParent->children.at(idx);
                if (dataSetInformationChild->type == DataSetType::NODE) {
                    if (ImGui::BeginMenu(dataSetInformationChild->name.c_str())) {
                        dataSetInformationStack.emplace(dataSetInformationRoot, idx + 1);
                        dataSetInformationStack.emplace(dataSetInformationChild, 0);
                        break;
                    }
                } else {
                    if (ImGui::MenuItem(dataSetInformationChild->name.c_str())) {
                        selectedDataSetIndexLocal = int(dataSetInformationChild->sequentialIndex);
                    }
                }
                idx++;
            }

            if (idx == dataSetInformationParent->children.size() && !dataSetInformationStack.empty()) {
                ImGui::EndMenu();
            }
        }
    }
    return selectedDataSetIndexLocal;
}

void MainApp::renderGuiMenuBar() {
    bool openOptimizerDialog = false;
    bool openRemoveTransparentTetsDialog = false;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Tet Mesh...", "CTRL+O")) {
                openFileDialog();
            }

            if (tetMesh && ImGui::MenuItem("Save Tet Mesh...", "CTRL+S")) {
                openSaveTetMeshFileDialog();
            }

            if (ImGui::BeginMenu("Datasets")) {
                for (int i = 1; i < NUM_MANUAL_LOADERS; i++) {
                    if (ImGui::MenuItem(dataSetNames.at(i).c_str())) {
                        selectedDataSetIndex = i;
                    }
                }

                int selectedDataSetIndexLocal = renderGuiDataSetSelectionMenu();
                if (selectedDataSetIndexLocal >= 0) {
                    selectedDataSetIndex = selectedDataSetIndexLocal;
                    isStartupFile = false;
                    loadTetMeshDataSet(getSelectedDataSetFilename());
                }

                ImGui::EndMenu();
            }

            if (ImGui::MenuItem("Quit", "CTRL+Q")) {
                quit();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window")) {
            if (ImGui::MenuItem("Tet Mesh Volume Renderer", nullptr, showRendererWindow)) {
                showRendererWindow = !showRendererWindow;
            }
            if (ImGui::MenuItem("FPS Overlay", nullptr, showFpsOverlay)) {
                showFpsOverlay = !showFpsOverlay;
            }
            if (ImGui::MenuItem("Coordinate Axes Overlay", nullptr, showCoordinateAxesOverlay)) {
                showCoordinateAxesOverlay = !showCoordinateAxesOverlay;
            }
            if (ImGui::MenuItem("Property Editor", nullptr, showPropertyEditor)) {
                showPropertyEditor = !showPropertyEditor;
            }
            if (ImGui::MenuItem("Checkpoint Window", nullptr, checkpointWindow.getShowWindow())) {
                checkpointWindow.setShowWindow(!checkpointWindow.getShowWindow());
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Tools")) {
            if (tetMesh && ImGui::MenuItem("Optimizer...")) {
                openOptimizerDialog = true;
            }

            if (ImGui::MenuItem("Print Camera State")) {
                std::cout << "Position: (" << camera->getPosition().x << ", " << camera->getPosition().y
                          << ", " << camera->getPosition().z << ")" << std::endl;
                std::cout << "Look At: (" << camera->getLookAtLocation().x << ", " << camera->getLookAtLocation().y
                          << ", " << camera->getLookAtLocation().z << ")" << std::endl;
                std::cout << "Yaw: " << camera->getYaw() << std::endl;
                std::cout << "Pitch: " << camera->getPitch() << std::endl;
                std::cout << "FoVy: " << (camera->getFOVy() / sgl::PI * 180.0f) << std::endl;
            }

            if (tetMesh && ImGui::MenuItem("Unlink Tets...")) {
                tetMesh->unlinkTets();
            }

            if (tetMesh && ImGui::MenuItem("Remove Transparent Tets...")) {
                openRemoveTransparentTetsDialog = true;
            }

            deviceSelector->renderGuiMenu();

            ImGui::EndMenu();
        }

        //if (dataRequester.getIsProcessingRequest()) {
        //    ImGui::SetCursorPosX(ImGui::GetWindowContentRegionWidth() - ImGui::GetTextLineHeight());
        //    ImGui::ProgressSpinner(
        //            "##progress-spinner", -1.0f, -1.0f, 4.0f,
        //            ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
        //}

        ImGui::EndMainMenuBar();
    }

    deviceSelector->renderGuiDialog();
    if (deviceSelector->getShallRestartApp()) {
        quit();
    }
    if (openOptimizerDialog) {
        tetMeshOptimizer->openDialog();
    }
    tetMeshOptimizer->renderGuiDialog();
    if (tetMeshOptimizer->getNeedsReRender()) {
        reRender = true;
    }
    if (openRemoveTransparentTetsDialog) {
        ImGui::OpenPopup("Remove Transparent Tets");
        isRemoveTransparentTetsDialogOpen = true;
    }
    if (isRemoveTransparentTetsDialogOpen) {
        if (ImGui::BeginPopupModal(
                "Remove Transparent Tets", &isRemoveTransparentTetsDialogOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::InputFloat("Alpha Threshold", &removeTetsAlphaThreshold);
            if (ImGui::Button("OK")) {
                tetMesh->removeTransparentTets(removeTetsAlphaThreshold);
                tetMeshVolumeRenderer->setTetMeshData(tetMesh);
                reRender = true;
                isRemoveTransparentTetsDialogOpen = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                isRemoveTransparentTetsDialogOpen = false;
            }
            ImGui::EndPopup();
        }
    }
}

void MainApp::renderGuiPropertyEditorBegin() {
    if (!useDockSpaceMode) {
        renderGuiFpsCounter();

        if (ImGui::Combo(
                "Data Set", &selectedDataSetIndex, dataSetNames.data(),
                int(dataSetNames.size()))) {
            if (selectedDataSetIndex >= NUM_MANUAL_LOADERS) {
                isStartupFile = false;
                loadTetMeshDataSet(getSelectedDataSetFilename());
            }
        }

        if (selectedDataSetIndex == 0) {
            ImGui::InputText("##datasetfilenamelabel", &customDataSetFileName);
            ImGui::SameLine();
            if (ImGui::Button("Load File")) {
                isStartupFile = false;
                loadTetMeshDataSet(getSelectedDataSetFilename());
            }
        }

        ImGui::Separator();
    }
}

void MainApp::renderGuiPropertyEditorCustomNodes() {
    if (propertyEditor.beginNode("Tet Mesh Volume Renderer")) {
        if (propertyEditor.addCombo(
                "Renderer", (int*)&tetMeshRendererType,
                TET_MESH_RENDERER_TYPES, IM_ARRAYSIZE(TET_MESH_RENDERER_TYPES))) {
            rendererVk->getDevice()->waitIdle();
            // syncWithCpu is necessary to avoid destroyed bound VkPipeline errors when continuous rendering is on.
            rendererVk->syncWithCpu();
            createTetMeshRenderer();
        }
        tetMeshVolumeRenderer->renderGuiPropertyEditorNodes(propertyEditor);
        propertyEditor.endNode();
    }
}

void MainApp::update(float dt) {
    sgl::SciVisApp::update(dt);

#ifdef __linux__
    datasetsWatch.update([this] {
        isFileWatchReload = true;
        this->loadAvailableDataSetInformation();
        isFileWatchReload = false;
    });
#endif

    if (scheduledDockSpaceModeChange) {
        useDockSpaceMode = newDockSpaceMode;
        scheduledDockSpaceModeChange = false;
        if (useDockSpaceMode) {
            cameraHandle = dataView->camera;
        } else {
            cameraHandle = camera;
        }

        device->waitGraphicsQueueIdle();
        resolutionChanged(sgl::EventPtr());
    }

    updateCameraFlight(tetMesh.get() != nullptr, usesNewState);

    checkLoadingRequestFinished();

    transferFunctionWindow.update(dt);

    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard || recording || focusedWindowIndex != -1) {
        moveCameraKeyboard(dt);
    }

    if (!io.WantCaptureMouse || mouseHoverWindowIndex != -1) {
        moveCameraMouse(dt);
    }
}

void MainApp::hasMoved() {
    dataView->syncCamera();
    tetMeshVolumeRenderer->onHasMoved();
}

void MainApp::onCameraReset() {
}

bool MainApp::checkHasValidExtension(const std::string& filenameLower) {
    if (sgl::endsWith(filenameLower, ".bintet")
#ifdef USE_OPEN_VOLUME_MESH
            || sgl::endsWith(filenameLower, ".ovm")
            || sgl::endsWith(filenameLower, ".ovmb")
            || sgl::endsWith(filenameLower, ".vtk")
#endif
            || sgl::endsWith(filenameLower, ".txt")
            || sgl::endsWith(filenameLower, ".msh")
            || sgl::endsWith(filenameLower, ".mesh")) {
        return true;
    }
    return false;
}

void MainApp::onFileDropped(const std::string& droppedFileName) {
    std::string filenameLower = sgl::toLowerCopy(droppedFileName);
    if (checkHasValidExtension(filenameLower)) {
        device->waitIdle();
        fileDialogDirectory = sgl::FileUtils::get()->getPathToFile(droppedFileName);
        selectedDataSetIndex = 0;
        customDataSetFileName = droppedFileName;
        isStartupFile = false;
        loadTetMeshDataSet(getSelectedDataSetFilename());
    } else {
        sgl::Logfile::get()->writeError(
                "The dropped file name has an unknown extension \""
                + sgl::FileUtils::get()->getFileExtension(filenameLower) + "\".");
    }
}



// --- Visualization pipeline ---

void MainApp::loadTetMeshDataSet(const std::string& fileName, bool blockingDataLoading) {
    if (fileName.empty()) {
        tetMesh = TetMeshPtr();
        return;
    }
    currentlyLoadedDataSetIndex = selectedDataSetIndex;

    DataSetInformation selectedDataSetInformation;
    if (selectedDataSetIndex >= NUM_MANUAL_LOADERS && !dataSetInformationList.empty()) {
        selectedDataSetInformation = *dataSetInformationList.at(selectedDataSetIndex - NUM_MANUAL_LOADERS);
    } else {
        selectedDataSetInformation.filenames = { fileName };
    }

    glm::mat4 transformationMatrix = sgl::matrixIdentity();
    //glm::mat4* transformationMatrixPtr = nullptr;
    if (selectedDataSetInformation.hasCustomTransform) {
        transformationMatrix *= selectedDataSetInformation.transformMatrix;
        //transformationMatrixPtr = &transformationMatrix;
    }
    if (rotateModelBy90DegreeTurns != 0) {
        transformationMatrix *= glm::rotate(rotateModelBy90DegreeTurns * sgl::HALF_PI, modelRotationAxis);
        //transformationMatrixPtr = &transformationMatrix;
    }

    TetMeshPtr tetMesh(new TetMesh(device, &transferFunctionWindow));

    if (blockingDataLoading) {
        // Add check if action triggered reload of startup file (which is actually a test case, not a true file).
        bool dataLoaded = false;
        if (isStartupFile) {
            tetMesh->loadTestData(TestCase::SINGLE_TETRAHEDRON);
        } else {
            dataLoaded = tetMesh->loadFromFile(fileName);
        }

        if (dataLoaded) {
            this->tetMesh = tetMesh;
            newMeshLoaded = true;
            boundingBox = tetMesh->getBoundingBox();

            tetMeshVolumeRenderer->setTetMeshData(tetMesh);
            tetMeshVolumeRenderer->setUseLinearRGB(useLinearRGB);
            reRender = true;

            const std::string& meshDescriptorName = fileName;
            checkpointWindow.onLoadDataSet(meshDescriptorName);

            if (true) { // useCameraFlight
                std::string cameraPathFilename =
                        saveDirectoryCameraPaths + sgl::FileUtils::get()->getPathAsList(meshDescriptorName).back()
                        + ".binpath";
                if (sgl::FileUtils::get()->exists(cameraPathFilename)) {
                    cameraPath.fromBinaryFile(cameraPathFilename);
                } else {
                    cameraPath.fromCirclePath(
                            boundingBox, meshDescriptorName,
                            usePerformanceMeasurementMode
                            ? CAMERA_PATH_TIME_PERFORMANCE_MEASUREMENT : CAMERA_PATH_TIME_RECORDING,
                            usePerformanceMeasurementMode);
                }
            }
        }
    } else {
        //dataRequester.queueRequest(tetMesh, fileName, selectedDataSetInformation, transformationMatrixPtr);
    }
}

void MainApp::checkLoadingRequestFinished() {
    /*TetMeshPtr tetMesh;
    DataSetInformation loadedDataSetInformation;

    //if (!tetMesh) {
    //    tetMesh = dataRequester.getLoadedData(loadedDataSetInformation);
    //}

    if (tetMesh) {
        this->tetMesh = tetMesh;
        newMeshLoaded = true;
        //modelBoundingBox = tetMesh->getModelBoundingBox();

        std::string meshDescriptorName = tetMesh->getFileName();
        checkpointWindow.onLoadDataSet(meshDescriptorName);

        if (true) {
            std::string cameraPathFilename =
                    saveDirectoryCameraPaths + sgl::FileUtils::get()->getPathAsList(meshDescriptorName).back()
                    + ".binpath";
            if (sgl::FileUtils::get()->exists(cameraPathFilename)) {
                cameraPath.fromBinaryFile(cameraPathFilename);
            } else {
                cameraPath.fromCirclePath(
                        modelBoundingBox, meshDescriptorName,
                        usePerformanceMeasurementMode
                        ? CAMERA_PATH_TIME_PERFORMANCE_MEASUREMENT : CAMERA_PATH_TIME_RECORDING,
                        usePerformanceMeasurementMode);
            }
        }
    }*/
}

void MainApp::reloadDataSet() {
    loadTetMeshDataSet(getSelectedDataSetFilename());
}
