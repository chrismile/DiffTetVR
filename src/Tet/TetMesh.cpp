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

#include <algorithm>
#include <unordered_map>

#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>

#ifdef USE_OPEN_VOLUME_MESH
#include <OpenVolumeMesh/Geometry/VectorT.hh>
#include <OpenVolumeMesh/Unstable/Topology/TetTopology.hh>
#include <OpenVolumeMesh/Mesh/TetrahedralMesh.hh>
#include <OpenVolumeMesh/Mesh/TetrahedralGeometryKernel.hh>
#include "Loaders/OvmLoader.hpp"
#endif

#include "Loaders/BinTetLoader.hpp"
#include "Loaders/TxtTetLoader.hpp"
#include "CSP/CSPSolver.hpp"
#include "CSP/FlipSolver.hpp"
#include "TetQualityFunctions.hpp"
#include "TetMesh.hpp"

#ifdef USE_OPEN_VOLUME_MESH
struct OvmRepresentationData {
    //OpenVolumeMesh::GeometricTetrahedralMeshV3f ovmMesh;
    OpenVolumeMesh::TetrahedralGeometryKernel<OpenVolumeMesh::Geometry::Vec3f, OpenVolumeMesh::TetrahedralMeshTopologyKernel> ovmMesh;
    OpenVolumeMesh::VertexPropertyPtr<OpenVolumeMesh::Vec4f> vertexColorProp;

};
#endif

template <typename T>
static std::pair<std::vector<std::string>, std::function<TetMeshLoader*()>> registerTetMeshLoader() {
    return { T::getSupportedExtensions(), []() { return new T{}; }};
}

TetMeshLoader* TetMesh::createTetMeshLoaderByExtension(const std::string& fileExtension) {
    auto it = factoriesLoader.find(fileExtension);
    if (it == factoriesLoader.end()) {
        sgl::Logfile::get()->writeError(
                "Error in TetMeshData::createTetMeshLoaderByExtension: Unsupported file extension '."
                + fileExtension + "'.", true);
        return nullptr;
    } else {
        return it->second();
    }
}

template <typename T>
static std::pair<std::vector<std::string>, std::function<TetMeshWriter*()>> registerTetMeshWriter() {
    return { T::getSupportedExtensions(), []() { return new T{}; }};
}

TetMeshWriter* TetMesh::createTetMeshWriterByExtension(const std::string& fileExtension) {
    auto it = factoriesWriter.find(fileExtension);
    if (it == factoriesWriter.end()) {
        sgl::Logfile::get()->throwError(
                "Error in TetMeshData::createTetMeshWriterByExtension: Unsupported file extension '."
                + fileExtension + "'.");
        return nullptr;
    } else {
        return it->second();
    }
}

TetMesh::TetMesh(sgl::vk::Device* device, sgl::TransferFunctionWindow* transferFunctionWindow)
        : device(device), transferFunctionWindow(transferFunctionWindow) {
    // Create the list of tet mesh loaders.
    std::map<std::vector<std::string>, std::function<TetMeshLoader*()>> factoriesLoaderMap = {
            registerTetMeshLoader<BinTetLoader>(),
            registerTetMeshLoader<TxtTetLoader>(),
#ifdef USE_OPEN_VOLUME_MESH
            registerTetMeshLoader<OvmLoader>(),
#endif
    };
    for (auto& factory : factoriesLoaderMap) {
        for (const std::string& extension : factory.first) {
            factoriesLoader.insert(std::make_pair(extension, factory.second));
        }
    }

    // Create the list of tet mesh writers.
    std::map<std::vector<std::string>, std::function<TetMeshWriter*()>> factoriesWriterMap = {
            registerTetMeshWriter<BinTetWriter>(),
            registerTetMeshWriter<TxtTetWriter>(),
#ifdef USE_OPEN_VOLUME_MESH
            registerTetMeshWriter<OvmWriter>(),
#endif
    };
    for (auto& factory : factoriesWriterMap) {
        for (const std::string& extension : factory.first) {
            factoriesWriter.insert(std::make_pair(extension, factory.second));
        }
    }
}

TetMesh::~TetMesh() {
    if (ovmRepresentationData) {
        delete ovmRepresentationData;
        ovmRepresentationData = nullptr;
    }
}

void TetMesh::setUseGradients(bool _useGradient) {
    useGradients = _useGradient;
    if (!useGradients) {
        vertexPositionGradientBuffer = {};
        vertexColorGradientBuffer = {};
    }
}

#if defined(BUILD_PYTHON_MODULE) && defined(SUPPORT_CUDA_INTEROP)
torch::Tensor TetMesh::getVertexPositionTensor() {
    torch::Tensor vertexPositionTensor = torch::from_blob(
            reinterpret_cast<float*>(vertexPositionBufferCu->getCudaDevicePtr()),
            { int(vertexPositions.size()), int(3) },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(useGradients));
    if (useGradients) {
        torch::Tensor vertexPositionGradientTensor = torch::from_blob(
                reinterpret_cast<float*>(vertexPositionGradientBufferCu->getCudaDevicePtr()),
                { int(vertexPositions.size()), int(3) },
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        vertexPositionTensor.mutable_grad() = vertexPositionGradientTensor;
    }
    return vertexPositionTensor;
}

torch::Tensor TetMesh::getVertexColorTensor() {
    torch::Tensor vertexColorTensor = torch::from_blob(
            reinterpret_cast<float*>(vertexColorBufferCu->getCudaDevicePtr()),
            { int(vertexColors.size()), int(4) },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(useGradients));
    if (useGradients) {
        torch::Tensor vertexColorGradientTensor = torch::from_blob(
                reinterpret_cast<float*>(vertexColorGradientBufferCu->getCudaDevicePtr()),
                { int(vertexColors.size()), int(4) },
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        vertexColorTensor.mutable_grad() = vertexColorGradientTensor;
    }
    return vertexColorTensor;
}
#endif

void TetMesh::setTetMeshData(
        const std::vector<uint32_t>& _cellIndices, const std::vector<glm::vec3>& _vertexPositions,
        const std::vector<glm::vec4>& _vertexColors) {
    cellIndices = _cellIndices;
    vertexPositions = _vertexPositions;
    vertexColors = _vertexColors;
    setTetMeshDataInternal();
}

void TetMesh::setTetMeshDataInternal() {
    //for (auto& color : vertexColors) {
    //    color.a = std::max(color.a, 1e-3f);
    //}

    meshNumCells = cellIndices.size() / 4;
    meshNumVertices = vertexPositions.size();
#ifdef USE_OPEN_VOLUME_MESH
    if (ovmRepresentationData) {
        delete ovmRepresentationData;
    }
    ovmRepresentationData = new OvmRepresentationData;
    newData = true;
    verticesDirty = false;
    facesDirty = true;
    cellsDirty = false;
    if (useOvmRepresentation) {
        rebuildInternalRepresentationIfNecessary_Ovm();
    } else {
        rebuildInternalRepresentationIfNecessary_Slim();
    }
#else
    rebuildInternalRepresentationIfNecessary_Slim();
#endif
    uploadDataToDevice();

    boundingBox = {};
    for (const glm::vec3& pt : vertexPositions) {
        boundingBox.combine(pt);
    }

    for (size_t tet = 0; tet < cellIndices.size(); tet += 4) {
        glm::vec3 p0 = vertexPositions.at(cellIndices.at(tet + 0));
        glm::vec3 p1 = vertexPositions.at(cellIndices.at(tet + 1));
        glm::vec3 p2 = vertexPositions.at(cellIndices.at(tet + 2));
        glm::vec3 p3 = vertexPositions.at(cellIndices.at(tet + 3));
        float signVal = glm::sign(glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0));
        if (signVal >= 0.0f) {
            std::cout << "Invalid sign for tet " << (tet / 4) << std::endl;
        }
    }
}

void TetMesh::uploadDataToDevice() {
    updateCellIndicesIfNecessary();
    updateVerticesIfNecessary();
    updateFacesIfNecessary();
    isVisualRepresentationDirty = true;

    std::vector<uint32_t> triangleIndices;
    triangleIndices.reserve(facesSlim.size() * 3);
    for (auto& f : facesSlim) {
        for (uint32_t v_idx : f.vs) {
            triangleIndices.push_back(v_idx);
        }
    }

    std::vector<glm::uvec2> faceToTetMapArray;
    for (auto& f : facesSlim) {
        faceToTetMapArray.emplace_back(f.tetId0, f.tetId1);
    }

    cellIndicesBuffer = {};
    triangleIndexBuffer = {};
    vertexPositionBuffer = {};
    vertexColorBuffer = {};
    faceBoundaryBitBuffer = {};
    vertexBoundaryBitBuffer = {};
    faceToTetMapBuffer = {};
    tetQualityBuffer = {};
    vertexPositionGradientBuffer = {};
    vertexColorGradientBuffer = {};
#if defined(BUILD_PYTHON_MODULE) && defined(SUPPORT_CUDA_INTEROP)
    vertexPositionBufferCu = {};
    vertexColorBufferCu = {};
    vertexPositionGradientBufferCu = {};
    vertexColorGradientBufferCu = {};
#endif

    cellIndicesBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * cellIndices.size(), cellIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    triangleIndexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * triangleIndices.size(), triangleIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    faceBoundaryBitBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * facesBoundarySlim.size(), facesBoundarySlim.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexBoundaryBitBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * verticesBoundarySlim.size(), verticesBoundarySlim.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    faceToTetMapBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::uvec2) * faceToTetMapArray.size(), faceToTetMapArray.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

#if defined(BUILD_PYTHON_MODULE) && defined(SUPPORT_CUDA_INTEROP)
    vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * vertexPositions.size(), vertexPositions.data(),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, true, true);
    vertexColorBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec4) * vertexColors.size(), vertexColors.data(),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, true, true);
    vertexPositionBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(vertexPositionBuffer);
    vertexColorBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(vertexColorBuffer);
    if (useGradients) {
        vertexPositionGradientBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec3) * vertexPositions.size(), vertexPositions.data(),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY, true, true);
        vertexColorGradientBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * vertexColors.size(), vertexColors.data(),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY, true, true);
        vertexPositionGradientBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(vertexPositionGradientBuffer);
        vertexColorGradientBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(vertexColorGradientBuffer);
    }
#else
    vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * vertexPositions.size(), vertexPositions.data(),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexColorBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec4) * vertexColors.size(), vertexColors.data(),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    if (useGradients) {
        vertexPositionGradientBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec3) * vertexPositions.size(), vertexPositions.data(),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        vertexColorGradientBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * vertexColors.size(), vertexColors.data(),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
    }
#endif
}

void TetMesh::setTetQualityMetric(TetQualityMetric _tetQualityMetric) {
    if (tetQualityArray.empty() || tetQualityMetric != _tetQualityMetric) {
        tetQualityMetric = _tetQualityMetric;
        isTetQualityDataDirty = true;
        getTetQualityBuffer();
    }
}

void TetMesh::fetchVertexDataFromDeviceIfNecessary() {
    if (verticesChangedOnDevice) {
        device->waitIdle();
        auto commandBuffer = device->beginSingleTimeCommands();
        sgl::vk::BufferPtr stagingBufferPosition = std::make_shared<sgl::vk::Buffer>(
                device, vertexPositionBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        sgl::vk::BufferPtr stagingBufferColor = std::make_shared<sgl::vk::Buffer>(
                device, vertexColorBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        vertexPositionBuffer->copyDataTo(stagingBufferPosition, commandBuffer);
        vertexColorBuffer->copyDataTo(stagingBufferColor, commandBuffer);
        device->endSingleTimeCommands(commandBuffer);
        auto* positionsPtr = reinterpret_cast<glm::vec3*>(stagingBufferPosition->mapMemory());
        auto* colorsPtr = reinterpret_cast<glm::vec4*>(stagingBufferColor->mapMemory());
        for (size_t i = 0; i < vertexPositions.size(); i++) {
            vertexPositions.at(i) = positionsPtr[i];
            vertexColors.at(i) = colorsPtr[i];
        }
        stagingBufferPosition->unmapMemory();
        stagingBufferColor->unmapMemory();

#ifdef USE_OPEN_VOLUME_MESH
        if (useOvmRepresentation) {
            // Update OpenVolumeMesh data structure.
            auto& ovmMesh = ovmRepresentationData->ovmMesh;
            auto& vertexPositionsOvm = ovmMesh.vertex_positions();
            auto& vertexColorProp = ovmRepresentationData->vertexColorProp;
            for (size_t vertexIdx = 0; vertexIdx < vertexPositions.size(); vertexIdx++) {
                OpenVolumeMesh::VertexHandle vh((int)vertexIdx);
                auto& vp = vertexPositions.at(vertexIdx);
                auto& vc = vertexColors.at(vertexIdx);
                auto& vpo = vertexPositionsOvm.at(vh);
                auto& vco = vertexColorProp.at(vh);
                vpo = OpenVolumeMesh::Vec3f(vp.x, vp.y, vp.z);
                vco = OpenVolumeMesh::Vec4f(vc.x, vc.y, vc.z, vc.w);
            }
        }
#endif

        boundingBox = {};
        for (const glm::vec3& pt : vertexPositions) {
            boundingBox.combine(pt);
        }

        verticesChangedOnDevice = false;
        isTetQualityDataDirty = true;
        verticesDirty = false;
    }
}

sgl::vk::BufferPtr TetMesh::getTetQualityBuffer() {
    fetchVertexDataFromDeviceIfNecessary();

    const size_t numCells = cellIndices.size() / 4;
    if (!tetQualityBuffer || sizeof(float) * numCells != tetQualityBuffer->getSizeInBytes()) {
        tetQualityBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(float) * numCells,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        isTetQualityDataDirty = true;
    }

    if (isTetQualityDataDirty) {
        tetQualityArray.resize(numCells);
        TetQualityMetricFunc* functor = getTetQualityMetricFunc(tetQualityMetric);
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numCells), [&](auto const& r) {
                for (auto tet = r.begin(); tet != r.end(); tet++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel for shared(numCells, functor) default(none)
#endif
        for (size_t tet = 0; tet < numCells; tet++) {
#endif
            size_t tetOffset = size_t(tet) * 4;
            glm::vec3 p0 = vertexPositions.at(cellIndices.at(tetOffset + 0));
            glm::vec3 p1 = vertexPositions.at(cellIndices.at(tetOffset + 1));
            glm::vec3 p2 = vertexPositions.at(cellIndices.at(tetOffset + 2));
            glm::vec3 p3 = vertexPositions.at(cellIndices.at(tetOffset + 3));
            tetQualityArray.at(tet) = functor(p0, p1, p2, p3);
        }
#ifdef USE_TBB
        });
#endif
        transferFunctionWindow->computeHistogram(tetQualityArray);
        isTetQualityDataDirty = false;
        device->waitIdle();
        tetQualityBuffer->uploadData(sizeof(float) * tetQualityArray.size(), tetQualityArray.data());
    }

    return tetQualityBuffer;
}

void TetMesh::loadTestData(TestCase testCase) {
    if (testCase == TestCase::SINGLE_TETRAHEDRON) {
        std::vector<uint32_t> _cellIndices = { 0, 1, 2, 3 };
        /*std::vector<glm::vec3> _vertexPositions = {
                {-0.1f, -0.1f,  0.1f},
                { 0.1f, -0.1f,  0.1f},
                { 0.0f,  0.1f,  0.0f},
                { 0.1f, -0.1f, -0.1f},
        };*/
        std::vector<glm::vec3> _vertexPositions = {
                {-0.1f, -0.1f,  0.1f},
                { 0.1f, -0.1f,  0.1f},
                { 0.0f,  0.3f,  0.0f},
                { 0.1f, -0.1f, -0.1f},
        };

        /*std::vector<glm::vec4> _vertexColors = {
                {0.8f, 0.0f, 0.0f, 0.2f},
                {0.0f, 0.8f, 0.0f, 0.4f},
                {0.0f, 0.0f, 0.8f, 0.1f},
                {0.8f, 0.8f, 0.0f, 0.3f},
        };*/
        /*std::vector<glm::vec4> _vertexColors = {
                {0.8f, 0.0f, 0.0f, 0.4f},
                {0.0f, 0.8f, 0.0f, 0.4f},
                {0.8f, 0.8f, 0.8f, 0.4f},
                {0.0f, 0.0f, 0.8f, 0.4f},
        };*/
        // GW.
        std::vector<glm::vec4> _vertexColors = {
                {1.0f, 1.0f, 1.0f, 0.1f},
                {1.0f, 1.0f, 1.0f, 0.1f},
                {0.0f, 1.0f, 0.0f, 0.1f},
                {1.0f, 1.0f, 1.0f, 0.8f},
        };
        // BW.
        /*std::vector<glm::vec4> _vertexColors = {
                {1.0f, 1.0f, 1.0f, 0.1f},
                {1.0f, 1.0f, 1.0f, 0.1f},
                {1.0f, 1.0f, 1.0f, 0.1f},
                {1.0f, 1.0f, 1.0f, 0.8f},
        };*/
        // Colored.
        /*std::vector<glm::vec4> _vertexColors = {
                {1.0f, 1.0f, 0.0f, 0.8f},
                {1.0f, 0.0f, 0.0f, 0.1f},
                {0.0f, 1.0f, 0.0f, 0.1f},
                {0.0f, 0.0f, 1.0f, 0.1f},
        };*/
        setTetMeshData(_cellIndices, _vertexPositions, _vertexColors);
    }
}

bool TetMesh::loadFromFile(const std::string& filePath) {
    std::string fileExtension = sgl::FileUtils::get()->getFileExtensionLower(filePath);
    TetMeshLoader* tetMeshLoader = createTetMeshLoaderByExtension(fileExtension);
    if (!tetMeshLoader) {
        return false;
    }
    std::vector<uint32_t> _cellIndices;
    std::vector<glm::vec3> _vertexPositions;
    std::vector<glm::vec4> _vertexColors;
    bool retVal;
#ifdef USE_OPEN_VOLUME_MESH
    if (tetMeshLoader->getNeedsOpenVolumeMeshSupport()) {
        if (ovmRepresentationData) {
            delete ovmRepresentationData;
        }
        ovmRepresentationData = new OvmRepresentationData;
        useOvmRepresentation = true;

        retVal = tetMeshLoader->loadFromFileOvm(filePath, ovmRepresentationData->ovmMesh);
        ovmRepresentationData->vertexColorProp =
                ovmRepresentationData->ovmMesh.request_vertex_property<OpenVolumeMesh::Vec4f>("vertexColors");

        newData = false;
        verticesDirty = true;
        facesDirty = true;
        cellsDirty = true;
        uploadDataToDevice();
    } else {
#endif
        retVal = tetMeshLoader->loadFromFile(filePath, _cellIndices, _vertexPositions, _vertexColors);
        if (retVal) {
            if (!forceUseOvmRepresentation) {
                useOvmRepresentation = false;
            }
            setTetMeshData(_cellIndices, _vertexPositions, _vertexColors);
        }
#ifdef USE_OPEN_VOLUME_MESH
    }
#endif
    delete tetMeshLoader;
    return retVal;
}

bool TetMesh::saveToFile(const std::string& filePath) {
    std::string fileExtension = sgl::FileUtils::get()->getFileExtensionLower(filePath);
    TetMeshWriter* tetMeshWriter = createTetMeshWriterByExtension(fileExtension);
    if (!tetMeshWriter) {
        return false;
    }
    updateVerticesIfNecessary();
    updateCellIndicesIfNecessary();
    bool retVal;
#ifdef USE_OPEN_VOLUME_MESH
    if (tetMeshWriter->getNeedsOpenVolumeMeshSupport()) {
        retVal = tetMeshWriter->saveToFileOvm(filePath, ovmRepresentationData->ovmMesh);
    } else {
#endif
        retVal = tetMeshWriter->saveToFile(filePath, cellIndices, vertexPositions, vertexColors);
#ifdef USE_OPEN_VOLUME_MESH
    }
#endif
    delete tetMeshWriter;
    return retVal;
}

const int tetFaceTable[4][3] = {
        // Use consistent winding for faces at the boundary (normals pointing out of the cell - no arbitrary decisions).
        { 0, 1, 2 },
        { 1, 0, 3 },
        { 0, 2, 3 },
        { 2, 1, 3 },
};

struct TempFace {
    uint32_t vertexId[3];
    uint32_t faceId;
    uint32_t tetId;

    inline bool operator==(const TempFace& other) const {
        for (int i = 0; i < 3; i++) {
            if (vertexId[i] != other.vertexId[i]) {
                return false;
            }
        }
        return true;
    }

    inline bool operator!=(const TempFace& other) const {
        for (int i = 0; i < 3; i++) {
            if (vertexId[i] != other.vertexId[i]) {
                return true;
            }
        }
        return false;
    }

    inline bool operator<(const TempFace& other) const {
        for (int i = 0; i < 3; i++) {
            if (vertexId[i] < other.vertexId[i]) {
                return true;
            } else if (vertexId[i] > other.vertexId[i]) {
                return false;
            }
        }
        return faceId < other.faceId;
    }
};

void buildFacesSlim(
        const std::vector<glm::vec3>& vertices, const std::vector<uint32_t>& cellIndices,
        std::vector<FaceSlim>& facesSlim, std::vector<uint32_t>& isBoundaryFace,
        std::vector<uint32_t>& isBoundaryVertex) {
    const uint32_t numTets = cellIndices.size() / 4;
    std::vector<FaceSlim> totalFaces(numTets * 4);
    std::vector<TempFace> tempFaces(numTets * 4);

    FaceSlim face{};
    for (uint32_t cellId = 0; cellId < numTets; ++cellId) {
        for (uint32_t faceIdx = 0; faceIdx < 4; faceIdx++){
            for (uint32_t vertexIdx = 0; vertexIdx < 3; vertexIdx++) {
                face.vs[vertexIdx] = cellIndices.at(cellId * 4 + tetFaceTable[faceIdx][vertexIdx]);
            }

            uint32_t faceId = 4 * cellId + faceIdx;
            totalFaces[faceId] = face;
            std::sort(face.vs, face.vs + 3);
            tempFaces[faceId] = TempFace{
                    { face.vs[0], face.vs[1], face.vs[2] }, faceId, cellId
            };
        }
    }
    std::sort(tempFaces.begin(), tempFaces.end());

    isBoundaryVertex.clear();
    isBoundaryVertex.resize(vertices.size(), false);
    facesSlim.reserve(tempFaces.size() / 2);
    uint32_t numFaces = 0;
    for (uint32_t i = 0; i < tempFaces.size(); ++i) {
        if (i == 0 || tempFaces[i] != tempFaces[i - 1]) {
            face = totalFaces[tempFaces[i].faceId];
            face.tetId0 = tempFaces[i].tetId;
            facesSlim.push_back(face);
            isBoundaryFace.push_back(true);
            numFaces++;
        } else {
            isBoundaryFace[numFaces - 1] = false;
            facesSlim[numFaces - 1].tetId1 = tempFaces[i].tetId;
        }
    }
    for (uint32_t i = 0; i < facesSlim.size(); ++i) {
        if (isBoundaryFace.at(i)) {
            for (unsigned int v : facesSlim.at(i).vs) {
                isBoundaryVertex[v] = true;
            }
        }
    }
}

void TetMesh::rebuildInternalRepresentationIfNecessary_Slim() {
    buildFacesSlim(vertexPositions, cellIndices, facesSlim, facesBoundarySlim, verticesBoundarySlim);
    isVisualRepresentationDirty = true;
}

#ifdef USE_OPEN_VOLUME_MESH
void TetMesh::rebuildInternalRepresentationIfNecessary_Ovm() {
    if (newData) {
        // https://www.graphics.rwth-aachen.de/media/openvolumemesh_static/Documentation/OpenVolumeMesh-Doc-Latest/ovm_tutorial_01.html
        auto& ovmMesh = ovmRepresentationData->ovmMesh;
        std::vector<OpenVolumeMesh::VertexHandle> ovmVertices;
        ovmVertices.reserve(vertexPositions.size());
        for (const glm::vec3& v : vertexPositions) {
            ovmVertices.emplace_back(ovmMesh.add_vertex(OpenVolumeMesh::Vec3f(v.x, v.y, v.z)));
        }

        const uint32_t numTets = cellIndices.size() / 4;
        std::vector<OpenVolumeMesh::VertexHandle> ovmCellVertices(4);
        for (uint32_t cellId = 0; cellId < numTets; ++cellId) {
            for (uint32_t faceIdx = 0; faceIdx < 4; faceIdx++){
                ovmCellVertices.at(faceIdx) = ovmVertices.at(cellIndices.at(cellId * 4 + faceIdx));
            }
            ovmMesh.add_cell(ovmCellVertices);
        }

        ovmRepresentationData->vertexColorProp =
                *ovmMesh.create_persistent_vertex_property<OpenVolumeMesh::Vec4f>("vertexColors");
        auto& vertexColorProp = ovmRepresentationData->vertexColorProp;
        for(OpenVolumeMesh::VertexIter v_it = ovmMesh.vertices_begin(); v_it != ovmMesh.vertices_end(); ++v_it) {
            auto vh = *v_it;
            const auto& vertexColor = vertexColors.at(vh.idx());
            vertexColorProp[vh] = OpenVolumeMesh::Vec4f(vertexColor.x, vertexColor.y, vertexColor.z, vertexColor.w);
        }

        newData = false;
    }
}

/*glm::vec3 getPos(
        const OpenVolumeMesh::PropertyPtr<OpenVolumeMesh::Vec3f, OpenVolumeMesh::Entity::Vertex> vertexPositionsOvm,
        OpenVolumeMesh::VertexHandle vh) {
    auto v = vertexPositionsOvm.at(vh);
    return { v[0], v[1], v[2] };
}*/

/*
 * b: 0, c: 1, d: 2, b': 3, c': 4, d': 5
 * abc: R(0) -> (b, c'), F(1) -> (c, b')
 * acd: R(0) -> (c, d'), F(1) -> (d, c')
 * abd: R(0) -> (d, b'), F(1) -> (b, d')
 */
const uint32_t PRISM_TO_TET_TABLE[8][12] = {
        // RRR, invalid
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        // FRR, splits: (c, b'), (c, d'), (d, b'); meets in: c, b'
        { 0, 1, 3, 2, 3, 4, 5, 1, 1, 2, 5, 3 },
        // RFR, splits: (b, c'), (d, c'), (d, b'); meets in: d, c'
        { 0, 1, 4, 2, 3, 4, 5, 2, 0, 2, 4, 3 },
        // FFR, splits: (c, b'), (d, c'), (d, b'); meets in: d, b'
        { 0, 1, 3, 2, 3, 4, 5, 2, 1, 2, 4, 3 },
        // RRF, splits: (b, c'), (c, d'), (b, d'); meets in: b, d'
        { 0, 1, 5, 2, 3, 4, 5, 0, 0, 1, 4, 5 },
        // FRF, splits: (c, b'), (c, d'), (b, d'); meets in: c, d'
        { 0, 1, 5, 2, 3, 4, 5, 1, 0, 1, 3, 5 },
        // RFF, splits: (b, c'), (d, c'), (b, d'); meets in: b, c'
        { 0, 1, 4, 2, 3, 4, 5, 0, 0, 2, 4, 5 },
        // FFF, invalid
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};
// Original before swaps for winding:
/*const uint32_t PRISM_TO_TET_TABLE[8][12] = {
        // RRR, invalid
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        // FRR, splits: (c, b'), (c, d'), (d, b'); meets in: c, b'
        { 0, 1, 2, 3, 3, 4, 5, 1, 1, 2, 3, 5 },
        // RFR, splits: (b, c'), (d, c'), (d, b'); meets in: d, c'
        { 0, 1, 2, 4, 3, 4, 5, 2, 0, 2, 3, 4 },
        // FFR, splits: (c, b'), (d, c'), (d, b'); meets in: d, b'
        { 0, 1, 2, 3, 3, 4, 5, 2, 1, 2, 3, 4 },
        // RRF, splits: (b, c'), (c, d'), (b, d'); meets in: b, d'
        { 0, 1, 2, 5, 3, 4, 5, 0, 0, 1, 4, 5 },
        // FRF, splits: (c, b'), (c, d'), (b, d'); meets in: c, d'
        { 0, 1, 2, 5, 3, 4, 5, 1, 0, 1, 3, 5 },
        // RFF, splits: (b, c'), (d, c'), (b, d'); meets in: b, c'
        { 0, 1, 2, 4, 3, 4, 5, 0, 0, 2, 4, 5 },
        // FFF, invalid
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};*/

//#define USE_SPLIT_EDGE
#define USE_OVM_SANITY_CHECK

void TetMesh::subdivideAtVertex(uint32_t vertexIndex, float t) {
    auto& ovmMesh = ovmRepresentationData->ovmMesh;
    auto& vertexPositionsOvm = ovmMesh.vertex_positions();
    auto& vertexColorProp = ovmRepresentationData->vertexColorProp;
    OpenVolumeMesh::VertexHandle vh((int)vertexIndex);

#ifdef USE_SPLIT_EDGE
    // Add new vertices along the incident edges & collect old edges to delete.
    std::vector<OpenVolumeMesh::EdgeHandle> edgesToDelete;
    for (auto ve_it = ovmMesh.ve_iter(vh); ve_it.valid(); ve_it++) {
        const auto& eh = ve_it.cur_handle();
        edgesToDelete.push_back(eh);
    }

    for (const auto& eh : edgesToDelete) {
        auto heh = eh.halfedge_handle(0);
        auto verts = ovmMesh.halfedge_vertices(heh);
        if (verts[0] != vh) {
            heh = heh.opposite_handle();
        }
        auto vhe = ovmMesh.split_edge(heh, t);
        const auto& vc0Ovm = vertexColorProp[verts[0]];
        const auto& vc1Ovm = vertexColorProp[verts[1]];
        glm::vec4 vc0(vc0Ovm[0], vc0Ovm[1], vc0Ovm[2], vc0Ovm[3]);
        glm::vec4 vc1(vc1Ovm[0], vc1Ovm[1], vc1Ovm[2], vc1Ovm[3]);
        glm::vec4 vce = glm::mix(vc0, vc1, t);
        vertexColorProp[vhe] = OpenVolumeMesh::Vec4f(vce.x, vce.y, vce.z, vce.w);
    }
    ovmMesh.collect_garbage();
#else
    // Add new vertices along the incident edges & collect old edges to delete.
    std::vector<OpenVolumeMesh::EdgeHandle> edgesToDelete;
    std::unordered_map<OpenVolumeMesh::EdgeHandle, OpenVolumeMesh::VertexHandle> edgeToNewVertexMap;
    for (auto ve_it = ovmMesh.ve_iter(vh); ve_it.valid(); ve_it++) {
        const auto& eh = ve_it.cur_handle();
        edgesToDelete.push_back(eh);
        auto vhs = ovmMesh.edge_vertices(eh);
        auto vh0 = vhs.at(0);
        auto vh1 = vhs.at(1);
        const auto& vp0Ovm = vertexPositionsOvm.at(vh0);
        const auto& vp1Ovm = vertexPositionsOvm.at(vh1);
        glm::vec3 vp0(vp0Ovm[0], vp0Ovm[1], vp0Ovm[2]);
        glm::vec3 vp1(vp1Ovm[0], vp1Ovm[1], vp1Ovm[2]);
        glm::vec3 vpe = glm::mix(vp0, vp1, t);
        const auto& vc0Ovm = vertexColorProp[vh0];
        const auto& vc1Ovm = vertexColorProp[vh1];
        glm::vec4 vc0(vc0Ovm[0], vc0Ovm[1], vc0Ovm[2], vc0Ovm[3]);
        glm::vec4 vc1(vc1Ovm[0], vc1Ovm[1], vc1Ovm[2], vc1Ovm[3]);
        glm::vec4 vce = glm::mix(vc0, vc1, t);
        auto vhe = ovmMesh.add_vertex(OpenVolumeMesh::Vec3f(vpe.x, vpe.y, vpe.z));
        vertexColorProp[vhe] = OpenVolumeMesh::Vec4f(vce.x, vce.y, vce.z, vce.w);
        edgeToNewVertexMap.insert(std::make_pair(eh, vhe));
    }

    // Iterate over all cells incident with the vertex.
    int numIncidentTets = 0;
    for (auto vc_it = ovmMesh.vc_iter(vh); vc_it.valid(); vc_it++) {
        numIncidentTets++;
    }
    auto** tetTopologies = new OpenVolumeMesh::TetTopology*[numIncidentTets];
    int tetIdx = 0;
    std::unordered_map<OpenVolumeMesh::HFH, int> halffaceToIndexMap;
    std::unordered_map<OpenVolumeMesh::CH, int> cellToIndexMap;
    for (auto vc_it = ovmMesh.vc_iter(vh); vc_it.valid(); vc_it++) {
        auto* tt = new OpenVolumeMesh::TetTopology(ovmMesh, *vc_it, vh); // vertex a == vh
        tetTopologies[tetIdx] = tt;
        halffaceToIndexMap.insert(std::make_pair(tt->abc(), tetIdx * 3));
        halffaceToIndexMap.insert(std::make_pair(tt->acd(), tetIdx * 3 + 1));
        halffaceToIndexMap.insert(std::make_pair(tt->adb(), tetIdx * 3 + 2));
        cellToIndexMap.insert(std::make_pair(*vc_it, tetIdx));
        tetIdx++;
    }

    // Build neighborhood graph.
    std::vector<Prism> prisms(numIncidentTets);
    for (tetIdx = 0; tetIdx < numIncidentTets; tetIdx++) {
        auto& prism = prisms.at(tetIdx);
        auto& tt = *tetTopologies[tetIdx]; // vertex a == vh
        std::array<OpenVolumeMesh::HFH, 3> hfhs = {
                tt.abc().opposite_handle(), tt.acd().opposite_handle(), tt.adb().opposite_handle()
        };
        for (int faceIdx = 0; faceIdx < 3; faceIdx++) {
            // ovmMesh.incident_cell(hfhs.at(faceIdx)).is_valid()
            if (hfhs.at(faceIdx).is_valid()) {
                auto itFace = halffaceToIndexMap.find(hfhs.at(faceIdx));
                auto ch = ovmMesh.incident_cell(hfhs.at(faceIdx));
                auto itCell = cellToIndexMap.find(ch);
                prism.neighborFaceIndices.at(faceIdx) = itFace != halffaceToIndexMap.end() ? itFace->second : -1;
                prism.neighbors.at(faceIdx) = itCell != cellToIndexMap.end() ? itCell->second : -1;
            }
        }
    }

    // Solve constraint satisfaction problem.
    auto* cspSolver = new FlipSolver();
    if (!cspSolver->solve(prisms)) {
        throw std::runtime_error("Error: CSP solver failed!");
    }
    if (!checkIsCspFulfilled(prisms)) {
        throw std::runtime_error("Error: CSP condition not fulfilled!");
    }
    delete cspSolver;

    // Collect the new indices of the new cells. Only add them after deleting the old cells.
    std::vector<OpenVolumeMesh::VertexHandle> newCells;
    // Prism vertex handles.
    std::array<OpenVolumeMesh::VertexHandle, 6> pvhs;
    for (tetIdx = 0; tetIdx < numIncidentTets; tetIdx++) {
        auto& prism = prisms.at(tetIdx);
        auto& tt = *tetTopologies[tetIdx]; // vertex a == vh
        pvhs[0] = tt.b();
        pvhs[1] = tt.c();
        pvhs[2] = tt.d();
        pvhs[3] = edgeToNewVertexMap.find(tt.ab().edge_handle())->second;
        pvhs[4] = edgeToNewVertexMap.find(tt.ac().edge_handle())->second;
        pvhs[5] = edgeToNewVertexMap.find(tt.ad().edge_handle())->second;

        // Add top.
        newCells.push_back(vh);
        newCells.push_back(pvhs[3]);
        newCells.push_back(pvhs[4]);
        newCells.push_back(pvhs[5]);

        uint32_t splits = prism.cuts.bitfield;
        for (int i = 0; i < 12; i++) {
            newCells.push_back(pvhs[PRISM_TO_TET_TABLE[splits][i]]);
        }
    }

    for (tetIdx = 0; tetIdx < numIncidentTets; tetIdx++) {
        delete tetTopologies[tetIdx];
    }
    delete[] tetTopologies;

    for (const auto& eh : edgesToDelete) {
        ovmMesh.delete_edge(eh);
    }
    ovmMesh.collect_garbage();

    // Add the new cells after the old ones have been deleted.
    for (size_t i = 0; i < newCells.size(); i += 4) {
        auto& p0Ovm = vertexPositionsOvm.at(newCells.at(i));
        auto& p1Ovm = vertexPositionsOvm.at(newCells.at(i + 1));
        auto& p2Ovm = vertexPositionsOvm.at(newCells.at(i + 2));
        auto& p3Ovm = vertexPositionsOvm.at(newCells.at(i + 3));
        glm::vec3 p0(p0Ovm[0], p0Ovm[1], p0Ovm[2]);
        glm::vec3 p1(p1Ovm[0], p1Ovm[1], p1Ovm[2]);
        glm::vec3 p2(p2Ovm[0], p2Ovm[1], p2Ovm[2]);
        glm::vec3 p3(p3Ovm[0], p3Ovm[1], p3Ovm[2]);
        float volumeSign = -glm::sign(glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0));
        assert(volumeSign > 0.0f && "Invalid winding");
        ovmMesh.add_cell(newCells.at(i), newCells.at(i + 1), newCells.at(i + 2), newCells.at(i + 3), true);
    }
#endif

#ifdef USE_OVM_SANITY_CHECK
    // Sanity check: Every tetrahedral cell may only have 4 vertices.
    for (OpenVolumeMesh::CellIter c_it = ovmMesh.cells_begin(); c_it != ovmMesh.cells_end(); c_it++) {
        auto ch = *c_it;
        int vidx = 0;
        for (auto cv_it = ovmMesh.cv_iter(ch); cv_it.valid(); cv_it++) {
            if (vidx >= 4) {
                int i = 0;
                for (auto cv_it = ovmMesh.cv_iter(ch); cv_it.valid(); cv_it++) {
                    std::cout << "v" << i << ": " << cv_it->uidx() << std::endl;
                    i++;
                }
            }
            assert(vidx < 4);
            vidx++;
        }
    }

    // Sanity check: Winding.
    for (OpenVolumeMesh::CellIter c_it = ovmMesh.cells_begin(); c_it != ovmMesh.cells_end(); c_it++) {
        auto ch = *c_it;
        auto cellVertices = ovmMesh.get_cell_vertices(ch);
        auto& p0Ovm = vertexPositionsOvm.at(cellVertices.at(0));
        auto& p1Ovm = vertexPositionsOvm.at(cellVertices.at(1));
        auto& p2Ovm = vertexPositionsOvm.at(cellVertices.at(2));
        auto& p3Ovm = vertexPositionsOvm.at(cellVertices.at(3));
        glm::vec3 p0(p0Ovm[0], p0Ovm[1], p0Ovm[2]);
        glm::vec3 p1(p1Ovm[0], p1Ovm[1], p1Ovm[2]);
        glm::vec3 p2(p2Ovm[0], p2Ovm[1], p2Ovm[2]);
        glm::vec3 p3(p3Ovm[0], p3Ovm[1], p3Ovm[2]);
        float volumeSign = -glm::sign(glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0));
        assert(volumeSign > 0.0f && "Invalid winding");
    }

    // Sanity check: Genus doesn't change from subdivision.
    int genus = ovmMesh.genus();
    if (genus != 0) {
        sgl::Logfile::get()->throwError("Error: genus (" + std::to_string(genus) + ") != 0 failed.");
    }
#endif

    verticesDirty = true;
    facesDirty = true;
    cellsDirty = true;
}

void TetMesh::updateVerticesIfNecessary() {
    if (useOvmRepresentation && verticesDirty) {
        // Update vertex data.
        auto& ovmMesh = ovmRepresentationData->ovmMesh;
        auto& vertexColorProp = ovmRepresentationData->vertexColorProp;
        vertexPositions.resize(ovmMesh.n_vertices());
        vertexColors.resize(ovmMesh.n_vertices());
        for (OpenVolumeMesh::VertexIter v_it = ovmMesh.vertices_begin(); v_it != ovmMesh.vertices_end(); v_it++) {
            auto vh = *v_it;
            const auto& vertexPosition = ovmMesh.vertex(vh);
            const auto& vertexColor = vertexColorProp[vh];
            vertexPositions.at(vh.idx()) = glm::vec3(vertexPosition[0], vertexPosition[1], vertexPosition[2]);
            vertexColors.at(vh.idx()) = glm::vec4(vertexColor[0], vertexColor[1], vertexColor[2], vertexColor[3]);
        }

        boundingBox = {};
        for (const glm::vec3& pt : vertexPositions) {
            boundingBox.combine(pt);
        }
        meshNumVertices = vertexPositions.size();
        verticesDirty = false;
    }
}

void TetMesh::updateFacesIfNecessary() {
    if (useOvmRepresentation && facesDirty) {
        // Update face data.
        auto& ovmMesh = ovmRepresentationData->ovmMesh;
        facesSlim.resize(ovmMesh.n_faces());
        facesBoundarySlim.resize(ovmMesh.n_faces());
        verticesBoundarySlim.clear();
        verticesBoundarySlim.resize(ovmMesh.n_vertices(), false);
        FaceSlim faceSlim{};
        for (OpenVolumeMesh::FaceIter f_it = ovmMesh.faces_begin(); f_it != ovmMesh.faces_end(); f_it++) {
            auto fh = *f_it;
            bool isBoundary =
                    !ovmMesh.incident_cell(fh.halfface_handle(0)).is_valid()
                    || !ovmMesh.incident_cell(fh.halfface_handle(1)).is_valid();
            facesBoundarySlim.at(fh.idx()) = isBoundary;
            int vidx = 0;
            for (auto fv_it = ovmMesh.fv_iter(fh); fv_it.valid(); fv_it++) {
                assert(vidx < 3);
                faceSlim.vs[vidx] = fv_it->idx();
                if (isBoundary) {
                    verticesBoundarySlim[fv_it->uidx()] = true;
                }
                vidx++;
            }
            auto faceCells = ovmMesh.face_cells(fh);
            faceSlim.tetId0 = faceCells.at(0).is_valid() ? faceCells.at(0).idx() : INVALID_TET;
            faceSlim.tetId1 = faceCells.at(1).is_valid() ? faceCells.at(1).idx() : INVALID_TET;
            facesSlim.at(fh.idx()) = faceSlim;
        }
        facesDirty = false;
    }
}

void TetMesh::updateCellIndicesIfNecessary() {
    if (useOvmRepresentation && cellsDirty) {
        // Update cell indices.
        auto& ovmMesh = ovmRepresentationData->ovmMesh;
        cellIndices.resize(4 * ovmMesh.n_cells());
        for (OpenVolumeMesh::CellIter c_it = ovmMesh.cells_begin(); c_it != ovmMesh.cells_end(); c_it++) {
            auto ch = *c_it;
            auto cellVertices = ovmMesh.get_cell_vertices(ch);
            assert(cellVertices.size() == 4);
            for (int vidx = 0; vidx < int(cellVertices.size()); vidx++) {
                cellIndices.at(ch.idx() * 4 + vidx) = cellVertices.at(vidx).uidx();
            }
            /*int vidx = 0;
            for (auto cv_it = ovmMesh.cv_iter(ch); cv_it.valid(); cv_it++) {
                assert(vidx < 4);
                cellIndices.at(ch.idx() * 4 + vidx) = cv_it->uidx();
                vidx++;
            }*/
        }
        meshNumCells = cellIndices.size() / 4;
        cellsDirty = false;
    }
}
#else
void TetMesh::updateVerticesIfNecessary() {
    ;
}

void TetMesh::updateFacesIfNecessary() {
    ;
}

void TetMesh::updateCellIndicesIfNecessary() {
    ;
}
#endif

void TetMesh::setForceUseOvmRepresentation() {
    useOvmRepresentation = true;
    forceUseOvmRepresentation = true;
}

void TetMesh::subdivideVertices(const std::vector<float>& gradientMagnitudes, uint32_t numSplits) {
#ifndef USE_OPEN_VOLUME_MESH
    sgl::Logfile::get()->throwError(
                "Error in TetMesh::subdivideVertices: Not built with OpenVolumeMesh support.");
#endif

    if (!useOvmRepresentation) {
        sgl::Logfile::get()->throwError(
                "Error in TetMesh::subdivideVertices: OpenVolumeMesh representation is not used.");
    }

    fetchVertexDataFromDeviceIfNecessary();

    std::vector<std::pair<float, uint32_t>> gradientMagnitudePairs;
    for (uint32_t i = 0; i < uint32_t(gradientMagnitudes.size()); i++) {
        gradientMagnitudePairs.emplace_back(gradientMagnitudes.at(i), i);
    }
    std::sort(gradientMagnitudePairs.rbegin(), gradientMagnitudePairs.rend());
    numSplits = std::min(numSplits, uint32_t(vertexPositions.size()));

#ifdef USE_OPEN_VOLUME_MESH
    for (uint32_t i = 0; i < numSplits; i++) {
        const float t = 0.5;
        uint32_t vertexIndex = gradientMagnitudePairs.at(i).second;
        subdivideAtVertex(vertexIndex, t);
    }
    uploadDataToDevice();
#endif
}

void TetMesh::splitByLargestGradientMagnitudes(SplitGradientType splitGradientType, float numSplitsRatio) {
    if (!vertexPositionGradientStagingBuffer
            || vertexPositionGradientStagingBuffer->getSizeInBytes() != vertexPositionGradientBuffer->getSizeInBytes()) {
        vertexPositionGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, vertexPositionGradientBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    }
    if (!vertexColorGradientStagingBuffer
            || vertexColorGradientStagingBuffer->getSizeInBytes() != vertexColorGradientBuffer->getSizeInBytes()) {
        vertexColorGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, vertexColorGradientBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    }
    vertexPositionGradientBuffer->copyDataTo(
            vertexPositionGradientStagingBuffer, 0, 0, vertexPositionGradientBuffer->getSizeInBytes(),
            renderer->getVkCommandBuffer());
    vertexColorGradientBuffer->copyDataTo(
            vertexColorGradientStagingBuffer, 0, 0, vertexColorGradientBuffer->getSizeInBytes(),
            renderer->getVkCommandBuffer());
    renderer->syncWithCpu();
    auto* vertexPositionGradients = reinterpret_cast<glm::vec3*>(vertexPositionGradientStagingBuffer->mapMemory());
    auto* vertexColorGradients = reinterpret_cast<glm::vec4*>(vertexColorGradientStagingBuffer->mapMemory());
    setVerticesChangedOnDevice(true);

    std::vector<float> gradientMagnitudes(getNumVertices());
    if (splitGradientType == SplitGradientType::POSITION
            || splitGradientType == SplitGradientType::ABS_POSITION) {
        computeVectorMagnitudeField(
                vertexPositionGradients, gradientMagnitudes.data(), int(getNumVertices()));
    } else {
        computeVectorMagnitudeField(
                vertexColorGradients, gradientMagnitudes.data(), int(getNumVertices()));
    }
    auto numSplits = uint32_t(std::ceil(double(numSplitsRatio) * double(getNumVertices())));
    subdivideVertices(gradientMagnitudes, numSplits);
    tetMeshVolumeRendererOpt->setTetMeshData(tetMeshOpt);

    vertexPositionGradientStagingBuffer->unmapMemory();
    vertexColorGradientStagingBuffer->unmapMemory();
}



const int hexFaceTable[6][4] = {
        // Use consistent winding for faces at the boundary (normals pointing out of the cell - no arbitrary decisions).
        { 0,1,2,3 },
        { 5,4,7,6 },
        { 4,5,1,0 },
        { 4,0,3,7 },
        { 6,7,3,2 },
        { 1,5,6,2 },
};
/*
 * Vertex and edge IDs:
 *
 *      3 +----------------+ 2
 *       /|               /|
 *      / |              / |
 *     /  |             /  |
 *    /   |            /   |
 * 7 +----------------+ 6  |
 *   |    |           |    |
 *   |    |           |    |
 *   |    |           |    |
 *   |  0 +-----------|----+ 1
 *   |   /            |   /
 *   |  /             |  /
 *   | /              | /
 *   |/               |/
 * 4 +----------------+ 5
 *
 * Tet mapping: https://cs.stackexchange.com/questions/89910/how-to-decompose-a-unit-cube-into-tetrahedra
 */
const int HEX_TO_TET_TABLE[6][4] = {
        // { 0, 4, 7, 6 }, -1
        // { 0, 4, 5, 6 }, +1
        // { 0, 3, 7, 6 }, +1
        // { 0, 3, 2, 6 }, -1
        // { 0, 1, 5, 6 }, -1
        // { 0, 1, 2, 6 }, +1
        { 0, 4, 7, 6 },
        { 0, 4, 6, 5 },
        { 0, 3, 6, 7 },
        { 0, 3, 2, 6 },
        { 0, 1, 5, 6 },
        { 0, 1, 6, 2 },
};

void convertHexToTetIndices(const std::vector<uint32_t>& hexIndices, std::vector<uint32_t>& tetIndices) {
    // Add all tet indices.
    uint32_t hex[8];
    for (size_t i = 0; i < hexIndices.size(); i += 8) {
        for (size_t j = 0; j < 8; j++) {
            hex[j] = hexIndices[i + j];
        }
        for (int tet = 0; tet < 6; tet++) {
            for (int idx = 0; idx < 4; idx++) {
                tetIndices.push_back(hex[HEX_TO_TET_TABLE[tet][idx]]);
            }
        }
    }
}

void TetMesh::setHexMeshConst(
        const sgl::AABB3& gridAabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor) {
    std::vector<uint32_t> hexIndices;
    cellIndices.clear();
    vertexPositions.clear();
    vertexColors.clear();

    // Add vertex positions & colors.
    for (int iz = 0; iz < int(zs); iz++) {
        for (int iy = 0; iy < int(ys); iy++) {
            for (int ix = 0; ix < int(xs); ix++) {
                glm::vec3 p;
                p.x = gridAabb.min.x + (float(ix) / float(xs - 1)) * (gridAabb.max.x - gridAabb.min.x);
                p.y = gridAabb.min.y + (float(iy) / float(ys - 1)) * (gridAabb.max.y - gridAabb.min.y);
                p.z = gridAabb.min.z + (float(iz) / float(zs - 1)) * (gridAabb.max.z - gridAabb.min.z);
                vertexPositions.emplace_back(p);
                vertexColors.emplace_back(constColor);
            }
        }
    }

    // Add all tet indices.
#define IDXS(x,y,z) ((z)*xs*ys + (y)*xs + (x))
    int hex[8];
    for (int iz = 0; iz < int(zs) - 1; iz++) {
        for (int iy = 0; iy < int(ys) - 1; iy++) {
            for (int ix = 0; ix < int(xs) - 1; ix++) {
                hex[0] = IDXS(ix,     iy,     iz    );
                hex[1] = IDXS(ix + 1, iy,     iz    );
                hex[2] = IDXS(ix + 1, iy + 1, iz    );
                hex[3] = IDXS(ix,     iy + 1, iz    );
                hex[4] = IDXS(ix,     iy,     iz + 1);
                hex[5] = IDXS(ix + 1, iy,     iz + 1);
                hex[6] = IDXS(ix + 1, iy + 1, iz + 1);
                hex[7] = IDXS(ix,     iy + 1, iz + 1);
                for (int tet = 0; tet < 6; tet++) {
                    for (int idx = 0; idx < 4; idx++) {
                        cellIndices.push_back(hex[HEX_TO_TET_TABLE[tet][idx]]);
                    }
                }
            }
        }
    }
#undef IDXS

    convertHexToTetIndices(hexIndices, cellIndices);
    setTetMeshDataInternal();
}

void TetMesh::unlinkTets() {
    updateCellIndicesIfNecessary();
    updateVerticesIfNecessary();
    if (!forceUseOvmRepresentation) {
        useOvmRepresentation = false;
    }
    std::vector<uint32_t> cellIndicesUnlinked(cellIndices.size());
    std::vector<glm::vec3> vertexPositionsUnlinked(cellIndices.size());
    std::vector<glm::vec4> vertexColorsUnlinked(cellIndices.size());
    const size_t numIndices = cellIndicesUnlinked.size();
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numIndices), [&](auto const& r) {
            for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
    #pragma omp parallel for shared(numIndices, cellIndices, cellIndicesUnlinked, vertexPositionsUnlinked, vertexColorsUnlinked) default(none)
#endif
    for (size_t i = 0; i < numIndices; i++) {
#endif
        size_t vidx = cellIndices.at(i);
        cellIndicesUnlinked.at(i) = i;
        vertexPositionsUnlinked.at(i) = vertexPositions.at(vidx);
        vertexColorsUnlinked.at(i) = vertexColors.at(vidx);
    }
#ifdef USE_TBB
    });
#endif
    setTetMeshData(cellIndicesUnlinked, vertexPositionsUnlinked, vertexColorsUnlinked);
}
