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

namespace sgl { namespace vk {
class Device;
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
}}

class TetMeshLoader;
class TetMeshWriter;

/**
 * Slim data representation is used only for rendering.
 */
struct FaceSlim {
    uint32_t vs[3]; ///< vertex indices
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
    explicit TetMesh(sgl::vk::Device* device);
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

    sgl::vk::BufferPtr getTriangleIndexBuffer() { return triangleIndexBuffer; }
    sgl::vk::BufferPtr getVertexPositionBuffer() { return vertexPositionBuffer; }
    sgl::vk::BufferPtr getVertexColorBuffer() { return vertexColorBuffer; }
    sgl::vk::BufferPtr getFaceBoundaryBitBuffer() { return faceBoundaryBitBuffer; }

    // Get mesh information.
    [[nodiscard]] inline size_t getNumCells() const { return meshNumCells; }
    [[nodiscard]] inline size_t getNumVertices() const { return meshNumVertices; }

    // File loaders.
    TetMeshLoader* createTetMeshLoaderByExtension(const std::string& fileExtension);
    std::map<std::string, std::function<TetMeshLoader*()>> factoriesLoader;
    std::vector<TetMeshLoader*> tetMeshLoaders;

    // File writers.
    TetMeshWriter* createTetMeshWriterByExtension(const std::string& fileExtension);
    std::map<std::string, std::function<TetMeshWriter*()>> factoriesWriter;

private:
    sgl::vk::Device* device;

    // Mesh data.
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
    //TetMeshRepresentationType representationType = TetMeshRepresentationType::SLIM;

    void rebuildInternalRepresentationIfNecessary_Slim();
    std::vector<FaceSlim> facesSlim;
    std::vector<uint32_t> facesBoundarySlim;

#ifdef USE_OPEN_VOLUME_MESH
    void rebuildInternalRepresentationIfNecessary_Ovm();
    OvmRepresentationData* ovmRepresentationData = nullptr;
#endif

    void updateVerticesIfNecessary();
    void updateFacesIfNecessary();
    void updateCellIndicesIfNecessary();

    // GPU data.
    void uploadDataToDevice();
    sgl::vk::BufferPtr triangleIndexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexColorBuffer;
    sgl::vk::BufferPtr faceBoundaryBitBuffer;
};

#endif //DIFFTETVR_TETMESH_HPP
