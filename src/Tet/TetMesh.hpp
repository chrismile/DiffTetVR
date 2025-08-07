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

#ifndef DIFFTETVR_TETMESH_HPP
#define DIFFTETVR_TETMESH_HPP

#include <vector>
#include <map>
#include <functional>
#include <string>
#include <memory>
#include <cstdint>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <Math/Geometry/AABB3.hpp>

#ifdef BUILD_PYTHON_MODULE
#include <torch/types.h>
#ifdef SUPPORT_COMPUTE_INTEROP
#include <Graphics/Vulkan/Utils/InteropCompute.hpp>
#endif
#endif

#include "../Renderer/OptimizerDefines.hpp"
#include "Meshing/TetMeshing.hpp"
#include "ColorStorage.hpp"
#include "TetQuality.hpp"

namespace sgl {
class TransferFunctionWindow;
}

namespace sgl { namespace vk {
class Device;
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
class Renderer;
}}

class TetMeshLoader;
class TetMeshWriter;

const uint32_t INVALID_TET = 0xFFFFFFFFu;

/**
 * Slim data representation is used only for rendering.
 */
struct FaceSlim {
    uint32_t vs[3]; ///< vertex indices
    uint32_t tetId0 = INVALID_TET, tetId1 = INVALID_TET;
};

enum class TestCase {
    SINGLE_TETRAHEDRON, CUBE_CENTRAL_GRADIENT
};

enum class TetMeshRepresentationType {
    SLIM, OPEN_VOLUME_MESH
};
struct OvmRepresentationData;

class TetMesh {
public:
    explicit TetMesh(sgl::vk::Device* device, sgl::TransferFunctionWindow* transferFunctionWindow);
    ~TetMesh();
    void setTetMeshData(
            const std::vector<uint32_t>& _cellIndices, const std::vector<glm::vec3>& _vertexPositions,
            const std::vector<glm::vec4>& _vertexColors);
    void setTetMeshDataCell(
            const std::vector<uint32_t>& _cellIndices, const std::vector<glm::vec3>& _vertexPositions,
            const std::vector<glm::vec4>& _cellColors);
    void loadTestData(TestCase testCase);
    bool loadFromFile(const std::string& filePath);
    bool saveToFile(const std::string& filePath);
    [[nodiscard]] inline bool isDirty() const { return isVisualRepresentationDirty; }
    inline void resetDirty() { isVisualRepresentationDirty = false; }
    [[nodiscard]] inline const sgl::AABB3& getBoundingBox() { return boundingBox; }
    void setVerticesChangedOnDevice(bool _verticesChanged);
    void onZeroGrad();
    void setTetQualityMetric(TetQualityMetric _tetQualityMetric);

    /// Returns whether any tetrahedral element is degenerate (i.e., has a volume <= 0).
    [[nodiscard]] bool checkIsAnyTetDegenerate();

    // Coarse to fine strategy.
    void setForceUseOvmRepresentation();
    void subdivideVertices(const std::vector<float>& gradientMagnitudes, uint32_t numSplits);
    void splitByLargestGradientMagnitudes(
            sgl::vk::Renderer* renderer, SplitGradientType splitGradientType, float splitsRatio);
    /// Initialize with tetrahedralized hex mesh with constant color.
    void setHexMeshConst(
            const sgl::AABB3& aabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor,
            ColorStorage ColorStorage);
    /// Initialize with tetrahedralized boundary mesh with constant color using an external application.
    bool setTetrahedralizedGridFTetWild(
            const sgl::AABB3& aabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor,
            ColorStorage ColorStorage, const FTetWildParams& params);
    bool setTetrahedralizedGridTetGen(
            const sgl::AABB3& aabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor,
            ColorStorage ColorStorage, const TetGenParams& params);

    [[nodiscard]] const std::vector<uint32_t>& getCellIndices() { return cellIndices; }
    [[nodiscard]] const std::vector<glm::vec3>& getVertexPositions() { fetchVertexDataFromDeviceIfNecessary(); return vertexPositions; }
    [[nodiscard]] const std::vector<glm::vec4>& getVertexColors() { fetchVertexDataFromDeviceIfNecessary(); return vertexColors; }
    [[nodiscard]] const std::vector<glm::vec4>& getCellColors() { fetchVertexDataFromDeviceIfNecessary(); return cellColors; }
    [[nodiscard]] bool getUseVertexColors() const { return !vertexColors.empty(); }
    [[nodiscard]] bool getUseCellColors() const { return !cellColors.empty(); }
    [[nodiscard]] ColorStorage getColorStorage() const { return !vertexColors.empty() ? ColorStorage::PER_VERTEX : ColorStorage::PER_CELL; }
    const sgl::vk::BufferPtr& getCellIndicesBuffer() { return cellIndicesBuffer; }
    const sgl::vk::BufferPtr& getTriangleIndexBuffer() { return triangleIndexBuffer; }
    const sgl::vk::BufferPtr& getVertexPositionBuffer() { return vertexPositionBuffer; }
    const sgl::vk::BufferPtr& getVertexColorBuffer() { return vertexColorBuffer; }
    const sgl::vk::BufferPtr& getCellColorBuffer() { return cellColorBuffer; }
    const sgl::vk::BufferPtr& getFaceBoundaryBitBuffer() { return faceBoundaryBitBuffer; }
    const sgl::vk::BufferPtr& getVertexBoundaryBitBuffer() { return vertexBoundaryBitBuffer; }
    // Buffers below are only used for tet quality renderer.
    const sgl::vk::BufferPtr& getFaceToTetMapBuffer() { return faceToTetMapBuffer; }
    const sgl::vk::BufferPtr& getTetQualityBuffer();

    // Gradient interface.
    void setUseGradients(bool _useGradient);
    [[nodiscard]] inline bool getUseGradients() const { return useGradients; }
    sgl::vk::BufferPtr getVertexPositionGradientBuffer() { return vertexPositionGradientBuffer; }
    sgl::vk::BufferPtr getVertexColorGradientBuffer() { return vertexColorGradientBuffer; }
    sgl::vk::BufferPtr getCellColorGradientBuffer() { return cellColorGradientBuffer; }

    // PyTorch buffer interface.
#ifdef BUILD_PYTHON_MODULE
    void setUseComputeInterop(bool _useComputeInterop);
    void setUsedDeviceType(torch::DeviceType _usedDeviceType);
    void copyGradientsToCpu(sgl::vk::Renderer* renderer);
    torch::Tensor getVertexPositionTensor();
    torch::Tensor getVertexColorTensor();
    torch::Tensor getCellColorTensor();
    torch::Tensor getVertexBoundaryBitTensor();
    // Experimental triangle mesh support.
    void setTriangleMeshData(
            torch::Tensor triangleIndicesTensor,
            torch::Tensor vertexPositionsTensor,
            torch::Tensor vertexColorsTensor);
    bool getHasTriangleMeshData();
#endif

    // Get mesh information.
    [[nodiscard]] inline size_t getNumCells() const { return meshNumCells; }
    [[nodiscard]] inline size_t getNumVertices() const { return meshNumVertices; }
    [[nodiscard]] inline bool getIsEmpty() const { return meshNumCells == 0 || meshNumVertices == 0; }

    // File loaders.
    TetMeshLoader* createTetMeshLoaderByExtension(const std::string& fileExtension);
    std::map<std::string, std::function<TetMeshLoader*()>> factoriesLoader;
    // Interface such that the next loader uses a constant vertex color instead of info from the file.
    void setNextLoaderUseConstColor(const glm::vec4& constColor, ColorStorage ColorStorage);

    // File writers.
    TetMeshWriter* createTetMeshWriterByExtension(const std::string& fileExtension);
    std::map<std::string, std::function<TetMeshWriter*()>> factoriesWriter;

    // Removes the links between all tets, i.e., a potentially used shared index representation is reversed.
    void unlinkTets();

private:
    sgl::vk::Device* device;

    // Mesh data.
    void setTetMeshDataInternal();
    size_t meshNumCells = 0;
    size_t meshNumVertices = 0;
    std::vector<uint32_t> cellIndices;
    std::vector<glm::vec3> vertexPositions;
    // Data sets can either provide vertex colors (for barycentric interpolation) or cell colors.
    std::vector<glm::vec4> vertexColors;
    std::vector<glm::vec4> cellColors;
    sgl::AABB3 boundingBox;
    bool newData = false;
    bool verticesDirty = false;
    bool facesDirty = false;
    bool cellsDirty = false;
    bool isVisualRepresentationDirty = false;
    bool useOvmRepresentation = false;
    bool forceUseOvmRepresentation = false;
    bool verticesChangedOnDevice = false;
    //TetMeshRepresentationType representationType = TetMeshRepresentationType::SLIM;
    // Tet quality.
    sgl::TransferFunctionWindow* transferFunctionWindow;
    std::vector<float> tetQualityArray;
    TetQualityMetric tetQualityMetric = DEFAULT_QUALITY_METRIC;
    bool isTetQualityDataDirty = true;

    bool nextLoaderUseConstColor = false;
    glm::vec4 constColorNext{};
    ColorStorage constColorNextColorStorage = ColorStorage::PER_VERTEX;

    void rebuildInternalRepresentationIfNecessary_Slim();
    std::vector<FaceSlim> facesSlim;
    std::vector<uint32_t> facesBoundarySlim;
    std::vector<uint32_t> verticesBoundarySlim;

#ifdef USE_OPEN_VOLUME_MESH
    void subdivideAtVertex(uint32_t vertexIndex, float t);
    void rebuildInternalRepresentationIfNecessary_Ovm();
    OvmRepresentationData* ovmRepresentationData = nullptr;
#endif

    void updateVerticesIfNecessary();
    void updateFacesIfNecessary();
    void updateCellIndicesIfNecessary();
    void fetchVertexDataFromDeviceIfNecessary();

    // GPU data.
    void uploadDataToDevice();
    sgl::vk::BufferPtr cellIndicesBuffer;
    sgl::vk::BufferPtr triangleIndexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexColorBuffer;
    sgl::vk::BufferPtr cellColorBuffer;
    sgl::vk::BufferPtr faceBoundaryBitBuffer;
    sgl::vk::BufferPtr vertexBoundaryBitBuffer;
    // Buffers below are only used for tet quality renderer.
    sgl::vk::BufferPtr faceToTetMapBuffer;
    sgl::vk::BufferPtr tetQualityBuffer;
    // Gradient data.
    bool useGradients = false;
    sgl::vk::BufferPtr vertexPositionGradientBuffer;
    sgl::vk::BufferPtr vertexColorGradientBuffer;
    sgl::vk::BufferPtr cellColorGradientBuffer;

    // CPU buffers.
    sgl::vk::BufferPtr vertexPositionBufferCpu;
    sgl::vk::BufferPtr vertexColorBufferCpu;
    sgl::vk::BufferPtr cellColorBufferCpu;
    sgl::vk::BufferPtr vertexPositionGradientBufferCpu;
    sgl::vk::BufferPtr vertexColorGradientBufferCpu;
    sgl::vk::BufferPtr cellColorGradientBufferCpu;
    void* vertexPositionBufferCpuPtr = nullptr;
    void* vertexColorBufferCpuPtr = nullptr;
    void* cellColorBufferCpuPtr = nullptr;
    void* vertexPositionGradientBufferCpuPtr = nullptr;
    void* vertexColorGradientBufferCpuPtr = nullptr;
    void* cellColorGradientBufferCpuPtr = nullptr;

#ifdef BUILD_PYTHON_MODULE
    bool useComputeInterop = false;
    torch::DeviceType usedDeviceType = torch::DeviceType::CPU;
#ifdef SUPPORT_COMPUTE_INTEROP
    sgl::vk::BufferVkComputeApiExternalMemoryPtr vertexPositionBufferCu;
    sgl::vk::BufferVkComputeApiExternalMemoryPtr vertexColorBufferCu;
    sgl::vk::BufferVkComputeApiExternalMemoryPtr cellColorBufferCu;
    sgl::vk::BufferVkComputeApiExternalMemoryPtr vertexBoundaryBitBufferCu;
    sgl::vk::BufferVkComputeApiExternalMemoryPtr vertexPositionGradientBufferCu;
    sgl::vk::BufferVkComputeApiExternalMemoryPtr vertexColorGradientBufferCu;
    sgl::vk::BufferVkComputeApiExternalMemoryPtr cellColorGradientBufferCu;
#endif
    bool hasTriangleMeshData = false;
#endif
};

typedef std::shared_ptr<TetMesh> TetMeshPtr;

#endif //DIFFTETVR_TETMESH_HPP
