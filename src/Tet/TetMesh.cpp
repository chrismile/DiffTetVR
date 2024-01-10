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

#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>

#ifdef USE_OPEN_VOLUME_MESH
#include <OpenVolumeMesh/Geometry/VectorT.hh>
#include <OpenVolumeMesh/Mesh/TetrahedralMesh.hh>
#include "Loaders/OvmLoader.hpp"
#endif

#include "Loaders/BinTetLoader.hpp"
#include "Loaders/TxtTetLoader.hpp"
#include "TetQualityFunctions.hpp"
#include "TetMesh.hpp"

#ifdef USE_OPEN_VOLUME_MESH
struct OvmRepresentationData {
    OpenVolumeMesh::GeometricTetrahedralMeshV3f ovmMesh;
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

void TetMesh::setTetMeshData(
        const std::vector<uint32_t>& _cellIndices, const std::vector<glm::vec3>& _vertexPositions,
        const std::vector<glm::vec4>& _vertexColors) {
    cellIndices = _cellIndices;
    vertexPositions = _vertexPositions;
    vertexColors = _vertexColors;

    //for (auto& color : vertexColors) {
    //    color.a = std::max(color.a, 1e-3f);
    //}

    meshNumCells = _cellIndices.size() / 4;
    meshNumVertices = _vertexPositions.size();
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
            std::cout << "Invalid sign for tet " << (cellIndices.size() / 4) << std::endl;
        }
    }
}

void TetMesh::uploadDataToDevice() {
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
    faceToTetMapBuffer = {};
    tetQualityBuffer = {};

    cellIndicesBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * cellIndices.size(), cellIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    triangleIndexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * triangleIndices.size(), triangleIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * vertexPositions.size(), vertexPositions.data(),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexColorBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec4) * vertexColors.size(), vertexColors.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    faceBoundaryBitBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * facesBoundarySlim.size(), facesBoundarySlim.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    faceToTetMapBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::uvec2) * faceToTetMapArray.size(), faceToTetMapArray.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void TetMesh::setTetQualityMetric(TetQualityMetric _tetQualityMetric) {
    if (tetQualityArray.empty() || tetQualityMetric != _tetQualityMetric) {
        tetQualityMetric = _tetQualityMetric;
        isTetQualityDataDirty = true;
        getTetQualityBuffer();
    }
}

sgl::vk::BufferPtr TetMesh::getTetQualityBuffer() {
    if (verticesChangedOnDevice) {
        device->waitIdle();
        auto commandBuffer = device->beginSingleTimeCommands();
        sgl::vk::BufferPtr stagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, vertexPositionBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        vertexPositionBuffer->copyDataTo(stagingBuffer, commandBuffer);
        device->endSingleTimeCommands(commandBuffer);
        auto* dataPtr = reinterpret_cast<glm::vec3*>(stagingBuffer->mapMemory());
        for (size_t i = 0; i < vertexPositions.size(); i++) {
            vertexPositions.at(i) = dataPtr[i];
        }
        stagingBuffer->unmapMemory();
        verticesChangedOnDevice = false;
        isTetQualityDataDirty = true;
    }

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
            useOvmRepresentation = false;
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
        std::vector<FaceSlim>& facesSlim, std::vector<uint32_t>& isBoundaryFace) {
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
}

void TetMesh::rebuildInternalRepresentationIfNecessary_Slim() {
    buildFacesSlim(vertexPositions, cellIndices, facesSlim, facesBoundarySlim);
    isVisualRepresentationDirty = true;
}

#ifdef USE_OPEN_VOLUME_MESH
void TetMesh::rebuildInternalRepresentationIfNecessary_Ovm() {
    if (newData) {
        // https://www.graphics.rwth-aachen.de/media/openvolumemesh_static/Documentation/OpenVolumeMesh-Doc-Latest/ovm_tutorial_01.html
        OpenVolumeMesh::GeometricTetrahedralMeshV3f& ovmMesh = ovmRepresentationData->ovmMesh;
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

void TetMesh::updateVerticesIfNecessary() {
    if (useOvmRepresentation && verticesDirty) {
        // Update vertex data.
        OpenVolumeMesh::GeometricTetrahedralMeshV3f& ovmMesh = ovmRepresentationData->ovmMesh;
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
    }
}

void TetMesh::updateFacesIfNecessary() {
    if (useOvmRepresentation && facesDirty) {
        // Update face data.
        OpenVolumeMesh::GeometricTetrahedralMeshV3f& ovmMesh = ovmRepresentationData->ovmMesh;
        facesSlim.resize(ovmMesh.n_faces());
        facesBoundarySlim.resize(ovmMesh.n_faces());
        FaceSlim faceSlim{};
        for (OpenVolumeMesh::FaceIter f_it = ovmMesh.faces_begin(); f_it != ovmMesh.faces_end(); f_it++) {
            auto fh = *f_it;
            facesBoundarySlim.at(fh.idx()) = !fh.halfface_handle(0).is_valid() || !fh.halfface_handle(1).is_valid();
            int vidx = 0;
            for (auto fv_it = ovmMesh.fv_iter(fh); fv_it.valid(); fv_it++) {
                assert(vidx < 3);
                faceSlim.vs[vidx] = fv_it->idx();
                vidx++;
            }
            auto faceCells = ovmMesh.face_cells(fh);
            faceSlim.tetId0 = faceCells.at(0).is_valid() ? faceCells.at(0).idx() : INVALID_TET;
            faceSlim.tetId1 = faceCells.at(1).is_valid() ? faceCells.at(1).idx() : INVALID_TET;
            facesSlim.at(fh.idx()) = faceSlim;
        }
    }
}

void TetMesh::updateCellIndicesIfNecessary() {
    if (useOvmRepresentation && cellsDirty) {
        // Update cell indices.
        OpenVolumeMesh::GeometricTetrahedralMeshV3f& ovmMesh = ovmRepresentationData->ovmMesh;
        cellIndices.resize(4 * ovmMesh.n_cells());
        for (OpenVolumeMesh::CellIter c_it = ovmMesh.cells_begin(); c_it != ovmMesh.cells_end(); c_it++) {
            auto ch = *c_it;
            int vidx = 0;
            for (auto cv_it = ovmMesh.cv_iter(ch); cv_it.valid(); cv_it++) {
                assert(vidx < 4);
                cellIndices.at(ch.idx() * 4 + vidx) = cv_it->uidx();
                vidx++;
            }
        }
        meshNumCells = cellIndices.size() / 4;
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
