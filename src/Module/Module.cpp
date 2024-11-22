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

#include <pybind11/functional.h>

#include <torch/script.h>
#include <torch/types.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#if defined(SUPPORT_CUDA_INTEROP) && CUDA_VERSION < 11020
#error Please install CUDA >= 11.2 for timeline semaphore support.
#endif

#include <Utils/AppSettings.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Utils/SyncObjects.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>

#include <Tet/TetMesh.hpp>
#include <Renderer/TetMeshVolumeRenderer.hpp>
#include <Renderer/OptimizerDefines.hpp>
#include <Renderer/Optimizer.hpp>

#ifdef __linux__
#include <dlfcn.h>
#endif

#ifdef _WIN32
#define _WIN32_IE 0x0400
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <shlobj.h>
#include <shlwapi.h>
#include <windef.h>
#include <windows.h>
#endif

class ApplicationState {
public:
    ApplicationState();
    ~ApplicationState();
    void checkSettings(
            uint32_t widthIn, uint32_t heightIn, uint32_t widthOut, uint32_t heightOut, uint32_t numColorChannels,
            caffe2::TypeMeta dtypeColor, caffe2::TypeMeta dtypeDepth, caffe2::TypeMeta dtypeMotionVector,
            float exposure);

    sgl::vk::Device* device = nullptr;
    sgl::vk::Renderer* renderer = nullptr;
    sgl::TransferFunctionWindow* transferFunctionWindow = nullptr;
    TetMeshOptimizer* optimizer = nullptr;

    uint32_t cachedWidth = 0, cachedHeight = 0;

    // For invocation of Vulkan command buffer in CUDA stream.
    sgl::vk::SemaphoreVkCudaDriverApiInteropPtr renderReadySemaphore;
    sgl::vk::SemaphoreVkCudaDriverApiInteropPtr renderFinishedSemaphore;
    uint64_t timelineValue = 0;
    sgl::vk::ImageViewPtr colorImageVk;
    sgl::vk::BufferPtr colorBufferVk;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr colorBufferCu;
};
static ApplicationState* sState = nullptr;

void vulkanErrorCallbackHeadless() {
    std::cerr << "Application callback" << std::endl;
}

static const char* argv[] = { "." }; //< Just pass something as argv.

static std::string getLibraryPath() {
#if defined(_WIN32)
    // See: https://stackoverflow.com/questions/6924195/get-dll-path-at-runtime
    WCHAR modulePath[MAX_PATH];
    HMODULE hmodule{};
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPWSTR>(&getLibraryPath), &hmodule)) {
        GetModuleFileNameW(hmodule, modulePath, MAX_PATH);
        PathRemoveFileSpecW(modulePath); //< Needs linking with shlwapi.lib.
        return sgl::wideStringArrayToStdString(modulePath);
    } else if (_DEBUG) {
        sgl::Logfile::get()->writeError(
                std::string() + "Error when calling GetModuleHandle: " + std::to_string(GetLastError()));
    }
    return "";
#elif defined(__linux__)
    // See: https://stackoverflow.com/questions/33151264/get-dynamic-library-directory-in-c-linux
    Dl_info dlInfo{};
    dladdr((void*)getLibraryPath, &dlInfo);
    std::string moduleDir = sgl::FileUtils::get()->getPathToFile(dlInfo.dli_fname);
    return moduleDir;
#else
    return ".";
#endif
}

ApplicationState::ApplicationState() {
    // Initialize the filesystem utilities.
    sgl::FileUtils::get()->initialize("pysrg", 1, argv);

    std::string dataDirectory = getLibraryPath();
    if (!sgl::endsWith(dataDirectory, "/") && !sgl::endsWith(dataDirectory, "\\")) {
        dataDirectory += "/";
    }
    dataDirectory += "Data";
    sgl::AppSettings::get()->setDataDirectory(dataDirectory);
    sgl::AppSettings::get()->initializeDataDirectory();

    sgl::AppSettings::get()->setSaveSettings(false);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-debugContext", true);
    sgl::AppSettings::get()->setRenderSystem(sgl::RenderSystem::VULKAN);
    std::vector<const char*> optionalInstanceExtensions;
#ifdef SUPPORT_DLSS
    getInstanceDlssSupportInfo(optionalInstanceExtensions);
    sgl::AppSettings::get()->setRequiredVulkanInstanceExtensions(optionalInstanceExtensions);
#endif
    sgl::AppSettings::get()->createHeadless();

    std::vector<const char*> optionalDeviceExtensions;
#ifdef SUPPORT_CUDA_INTEROP
    optionalDeviceExtensions = sgl::vk::Device::getCudaInteropDeviceExtensions();
#endif
    sgl::vk::DeviceFeatures requestedDeviceFeatures{};

    sgl::vk::Instance* instance = sgl::AppSettings::get()->getVulkanInstance();
    sgl::AppSettings::get()->getVulkanInstance()->setDebugCallback(&vulkanErrorCallbackHeadless);
    device = new sgl::vk::Device;

#ifdef SUPPORT_DLSS
    auto physicalDeviceCheckCallback = [instance](
            VkPhysicalDevice physicalDevice,
            VkPhysicalDeviceProperties physicalDeviceProperties,
            std::vector<const char*>& requiredDeviceExtensions,
            std::vector<const char*>& optionalDeviceExtensions,
            sgl::vk::DeviceFeatures& requestedDeviceFeatures) {
        if (physicalDeviceProperties.apiVersion < VK_API_VERSION_1_1) {
            return false;
        }
        getPhysicalDeviceDlssSupportInfo(
                instance, physicalDevice, optionalDeviceExtensions, requestedDeviceFeatures);
        if (physicalDeviceProperties.apiVersion >= VK_API_VERSION_1_2) {
            bool needsDeviceAddressFeature = false;
            for (size_t i = 0; i < optionalDeviceExtensions.size();) {
                const char* extensionName = optionalDeviceExtensions.at(i);
                if (strcmp(extensionName, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
                    optionalDeviceExtensions.erase(optionalDeviceExtensions.begin() + i);
                    needsDeviceAddressFeature = true;
                    continue;
                }
                if (strcmp(extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
                    optionalDeviceExtensions.erase(optionalDeviceExtensions.begin() + i);
                    needsDeviceAddressFeature = true;
                    continue;
                }
                i++;
            }
            if (needsDeviceAddressFeature) {
                requestedDeviceFeatures.optionalVulkan12Features.bufferDeviceAddress = VK_TRUE;
            }
        }
        return true;
    };
    device->setPhysicalDeviceCheckCallback(physicalDeviceCheckCallback);
#endif

    device->createDeviceHeadless(
            instance, {
                    VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
            },
            optionalDeviceExtensions, requestedDeviceFeatures);
    sgl::AppSettings::get()->setPrimaryDevice(device);
    sgl::AppSettings::get()->initializeSubsystems();

    renderer = new sgl::vk::Renderer(sgl::AppSettings::get()->getPrimaryDevice());
    transferFunctionWindow = new sgl::TransferFunctionWindow;
    // TODO: Callbacks
    optimizer = new TetMeshOptimizer(
            renderer, [](const TetMeshPtr&, float) {},
            false, []() -> std::string { return ""; },
            transferFunctionWindow);
}

ApplicationState::~ApplicationState() {
    device->waitIdle();
    renderReadySemaphore = {};
    renderFinishedSemaphore = {};
    colorImageVk = {};
    colorBufferVk = {};
    colorBufferCu = {};
    delete optimizer;
    delete transferFunctionWindow;
    delete renderer;
#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::vk::freeCudaDeviceApiFunctionTable();
    }
#endif
    sgl::AppSettings::get()->release();
}

static void ensureStateExists() {
    if (!sState) {
        sState = new ApplicationState;
    }
}

void difftetvrCleanup() {
    if (sState) {
        delete sState;
        sState = nullptr;
    }
}

torch::Tensor forward(torch::Tensor X) {
    return X;
}

struct CameraSettings {
    glm::vec3 position;
    uint32_t width, height;
};

TetMeshPtr createTetMesh(CameraSettings cameraSettings, torch::Tensor colorImage) {
    // TODO
    return TetMeshPtr();
}

/*TetMeshPtr createEmptyGridMesh(
        const sgl::AABB3& aabb, uint32_t width, uint32_t height, uint32_t depth) {
    // TODO
    tetMeshOpt->setHexMeshConst(
            tetMeshGT->getBoundingBox(), settings.initGridResolution.x,
            settings.initGridResolution.y, settings.initGridResolution.z,
            glm::vec4(0.5f, 0.5f, 0.5f, 0.1f));
}*/

torch::Tensor renderForward(CameraSettings cameraSettings, torch::Tensor colorImage) {
    // Use enable_grad() on tensor on Python side if backpropagation should be used.
    // https://pytorch.org/docs/stable/generated/torch.enable_grad.html
    return colorImage;
}

torch::Tensor renderAdjoint(CameraSettings cameraSettings, torch::Tensor colorImage) {
    auto adjointImage = colorImage.grad();
    return colorImage;
}

PYBIND11_MODULE(difftetvr, m) {
    py::class_<glm::vec3>(m, "vec3")
            .def_readwrite("x", &glm::vec3::x)
            .def_readwrite("y", &glm::vec3::y)
            .def_readwrite("z", &glm::vec3::z);

    py::class_<sgl::AABB3>(m, "AABB3")
            .def(py::init<>())
            .def(py::init<glm::vec3, glm::vec3>(), py::arg("min"), py::arg("max"))
            .def_readwrite("min", &sgl::AABB3::min)
            .def_readwrite("max", &sgl::AABB3::max)
            .def("get_dimensions", &sgl::AABB3::getDimensions)
            .def("get_extent", &sgl::AABB3::getExtent)
            .def("get_center", &sgl::AABB3::getCenter)
            .def("get_minimum", &sgl::AABB3::getMinimum)
            .def("get_maximum", &sgl::AABB3::getMaximum);

    py::enum_<TestCase>(m, "TestCase", py::arithmetic(), "")
            .value("SINGLE_TETRAHEDRON", TestCase::SINGLE_TETRAHEDRON, "");
    py::class_<TetMesh, TetMeshPtr>(m, "TetMesh")
            .def(py::init([]() {
                ensureStateExists();
                return std::make_shared<TetMesh>(sState->device, sState->transferFunctionWindow);
            }))
            .def("load_test_data", &TetMesh::loadTestData, py::arg("test_case"))
            .def("load_from_file", &TetMesh::loadFromFile, py::arg("file_path"))
            .def("save_to_file", &TetMesh::saveToFile, py::arg("file_path"))
            .def("get_bounding_box", &TetMesh::getBoundingBox)
            .def("set_force_use_ovm_representation", &TetMesh::setForceUseOvmRepresentation, "Coarse to fine strategy.")
            .def("set_hex_mesh_const", &TetMesh::setHexMeshConst,
                 py::arg("aabb"), py::arg("xs"), py::arg("ys"), py::arg("zs"), py::arg("const_color"),
                 "Initialize with tetrahedralized tet mesh with constant color.")
            .def("get_num_cells", &TetMesh::getNumCells)
            .def("get_num_vertices", &TetMesh::getNumVertices)
            .def("unlink_tets", &TetMesh::unlinkTets,
                 "Removes the links between all tets, i.e., a potentially used shared index representation is reversed.");

    py::enum_<OptimizerType>(m, "OptimizerType")
            .value("SGD", OptimizerType::SGD)
            .value("ADAM", OptimizerType::ADAM);
    py::enum_<LossType>(m, "LossType")
            .value("L1", LossType::L1)
            .value("L2", LossType::L2);
    py::enum_<SplitGradientType>(m, "SplitGradientType")
            .value("POSITION", SplitGradientType::POSITION)
            .value("COLOR", SplitGradientType::COLOR)
            .value("ABS_POSITION", SplitGradientType::ABS_POSITION)
            .value("ABS_COLOR", SplitGradientType::ABS_COLOR);
    py::class_<OptimizerSettings>(m, "OptimizerSettings")
            .def(py::init<>())
            .def_readwrite("learning_rate", &OptimizerSettings::learningRate)
            .def_readwrite("lr_decay_rate", &OptimizerSettings::lrDecayRate)
            .def_readwrite("beta1", &OptimizerSettings::beta1)
            .def_readwrite("beta2", &OptimizerSettings::beta2)
            .def_readwrite("epsilon", &OptimizerSettings::epsilon);
    py::class_<TetRegularizerSettings>(m, "TetRegularizerSettings")
            .def(py::init<>())
            .def_readwrite("lambda", &TetRegularizerSettings::lambda)
            .def_readwrite("beta", &TetRegularizerSettings::beta);
    py::class_<OptimizationSettings>(m, "OptimizationSettings")
            .def(py::init<>())
            .def_readwrite("optimizer_type", &OptimizationSettings::optimizerType)
            .def_readwrite("loss_type", &OptimizationSettings::lossType)
            .def_readwrite("optimize_positions", &OptimizationSettings::optimizePositions)
            .def_readwrite("optimize_colors", &OptimizationSettings::optimizeColors)
            .def_readwrite("optimizer_settings_positions", &OptimizationSettings::optimizerSettingsPositions)
            .def_readwrite("optimizer_settings_colors", &OptimizationSettings::optimizerSettingsColors)
            .def_readwrite("tet_regularizer_settings", &OptimizationSettings::tetRegularizerSettings)
            .def_readwrite("max_num_epochs", &OptimizationSettings::maxNumEpochs)
            .def_readwrite("fix_boundary", &OptimizationSettings::fixBoundary)
            .def_readwrite("image_width", &OptimizationSettings::imageWidth)
            .def_readwrite("image_height", &OptimizationSettings::imageHeight)
            .def_readwrite("attenuation_coefficient", &OptimizationSettings::attenuationCoefficient)
            .def_readwrite("sample_random_view", &OptimizationSettings::sampleRandomView)
            .def_readwrite("data_set_file_name_gt", &OptimizationSettings::dataSetFileNameGT, "Selected file name.")
            .def_readwrite("data_set_file_name_opt", &OptimizationSettings::dataSetFileNameOpt, "Selected file name.")
            .def_readwrite("use_coarse_to_fine", &OptimizationSettings::useCoarseToFine)
            .def_readwrite("use_constant_init_grid", &OptimizationSettings::useConstantInitGrid)
            .def_readwrite("init_grid_resolution", &OptimizationSettings::initGridResolution)
            .def_readwrite("max_num_tets", &OptimizationSettings::maxNumTets)
            .def_readwrite("num_splits_ratio", &OptimizationSettings::numSplitsRatio)
            .def_readwrite("split_gradient_type", &OptimizationSettings::splitGradientType)
            .def_readwrite("export_position_gradients", &OptimizationSettings::exportPositionGradients)
            .def_readwrite("export_file_name_gradient_field", &OptimizationSettings::exportFileNameGradientField)
            .def_readwrite("is_binary_vtk", &OptimizationSettings::isBinaryVtk);
    m.def("forward", forward,
        "Forward rendering pass.",
        py::arg("X"));
    m.add_object("_cleanup", py::capsule(difftetvrCleanup));
}
