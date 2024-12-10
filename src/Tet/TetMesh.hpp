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

#if defined(BUILD_PYTHON_MODULE) && defined(SUPPORT_CUDA_INTEROP)
#include <torch/types.h>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif

#include "../Renderer/OptimizerDefines.hpp"
#include "TetQuality.hpp"

namespace sgl {
class TransferFunctionWindow;
}

namespace sgl { namespace vk {
class Device;
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
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
    SINGLE_TETRAHEDRON
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
    void loadTestData(TestCase testCase);
    bool loadFromFile(const std::string& filePath);
    bool saveToFile(const std::string& filePath);
    [[nodiscard]] inline bool isDirty() const { return isVisualRepresentationDirty; }
    inline void resetDirty() { isVisualRepresentationDirty = false; }
    [[nodiscard]] inline const sgl::AABB3& getBoundingBox() { return boundingBox; }
    void setVerticesChangedOnDevice(bool _verticesChanged) { verticesChangedOnDevice = _verticesChanged; }
    void setTetQualityMetric(TetQualityMetric _tetQualityMetric);

    /// Returns whether any tetrahedral element is degenerate (i.e., has a volume <= 0).
    [[nodiscard]] bool checkIsAnyTetDegenerate();

    // Coarse to fine strategy.
    void setForceUseOvmRepresentation();
    void subdivideVertices(const std::vector<float>& gradientMagnitudes, uint32_t numSplits);
    void splitByLargestGradientMagnitudes(
            sgl::vk::Renderer* renderer, SplitGradientType splitGradientType, float splitsRatio);
    /// Initialize with tetrahedralized tet mesh with constant color.
    void setHexMeshConst(const sgl::AABB3& aabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor);

    [[nodiscard]] const std::vector<uint32_t>& getCellIndices() const { return cellIndices; }
    [[nodiscard]] const std::vector<glm::vec3>& getVertexPositions() const { return vertexPositions; }
    [[nodiscard]] const std::vector<glm::vec4>& getVertexColors() const { return vertexColors; }
    const sgl::vk::BufferPtr& getCellIndicesBuffer() { return cellIndicesBuffer; }
    const sgl::vk::BufferPtr& getTriangleIndexBuffer() { return triangleIndexBuffer; }
    const sgl::vk::BufferPtr& getVertexPositionBuffer() { return vertexPositionBuffer; }
    const sgl::vk::BufferPtr& getVertexColorBuffer() { return vertexColorBuffer; }
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

    // PyTorch buffer interface.
#if defined(BUILD_PYTHON_MODULE) && defined(SUPPORT_CUDA_INTEROP)
    torch::Tensor getVertexPositionTensor();
    torch::Tensor getVertexColorTensor();
    torch::Tensor getVertexBoundaryBitTensor();
#endif

    // Get mesh information.
    [[nodiscard]] inline size_t getNumCells() const { return meshNumCells; }
    [[nodiscard]] inline size_t getNumVertices() const { return meshNumVertices; }
    [[nodiscard]] inline bool getIsEmpty() const { return meshNumCells == 0 || meshNumVertices == 0; }

    // File loaders.
    TetMeshLoader* createTetMeshLoaderByExtension(const std::string& fileExtension);
    std::map<std::string, std::function<TetMeshLoader*()>> factoriesLoader;

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
    std::vector<glm::vec4> vertexColors;
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
    sgl::vk::BufferPtr faceBoundaryBitBuffer;
    sgl::vk::BufferPtr vertexBoundaryBitBuffer;
    // Buffers below are only used for tet quality renderer.
    sgl::vk::BufferPtr faceToTetMapBuffer;
    sgl::vk::BufferPtr tetQualityBuffer;
    // Gradient data.
    bool useGradients = false;
    sgl::vk::BufferPtr vertexPositionGradientBuffer;
    sgl::vk::BufferPtr vertexColorGradientBuffer;
    sgl::vk::BufferPtr vertexPositionGradientStagingBuffer;
    sgl::vk::BufferPtr vertexColorGradientStagingBuffer;
#if defined(BUILD_PYTHON_MODULE) && defined(SUPPORT_CUDA_INTEROP)
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr vertexPositionBufferCu;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr vertexColorBufferCu;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr vertexBoundaryBitBufferCu;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr vertexPositionGradientBufferCu;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr vertexColorGradientBufferCu;
#endif
};

typedef std::shared_ptr<TetMesh> TetMeshPtr;

#endif //DIFFTETVR_TETMESH_HPP
