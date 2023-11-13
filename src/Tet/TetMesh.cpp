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

#include <Graphics/Vulkan/Buffers/Buffer.hpp>

#include "TetMesh.hpp"

TetMesh::TetMesh(sgl::vk::Device* device) : device(device) {
}

void TetMesh::setTetMeshData(
        const std::vector<uint32_t>& _cellIndices, const std::vector<glm::vec3>& _vertexPositions,
        const std::vector<glm::vec4>& _vertexColors) {
    cellIndices = _cellIndices;
    vertexPositions = _vertexPositions;
    vertexColors = _vertexColors;
    rebuildInternalRepresentationIfNecessary_Slim();
    uploadDataToDevice();
}

void TetMesh::uploadDataToDevice() {
    std::vector<uint32_t> triangleIndices;
    triangleIndices.reserve(facesSlim.size() * 3);
    for (auto& f : facesSlim) {
        for (uint32_t v_idx : f.vs) {
            triangleIndices.push_back(v_idx);
        }
    }

    triangleIndexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * triangleIndices.size(), triangleIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * vertexPositions.size(), vertexPositions.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexColorBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec4) * vertexColors.size(), vertexColors.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    faceBoundaryBitBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * facesBoundarySlim.size(), facesBoundarySlim.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void TetMesh::loadTestData(TestCase testCase) {
    if (testCase == TestCase::SINGLE_TETRAHEDRON) {
        std::vector<uint32_t> _cellIndices = { 0, 1, 2, 3 };
        std::vector<glm::vec3> _vertexPositions = {
                {-0.1f, -0.1f,  0.1f},
                { 0.1f, -0.1f,  0.1f},
                { 0.0f,  0.1f,  0.0f},
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
        std::vector<glm::vec4> _vertexColors = {
                {1.0f, 1.0f, 1.0f, 0.1f},
                {1.0f, 1.0f, 1.0f, 0.1f},
                {0.0f, 1.0f, 0.0f, 0.1f},
                {1.0f, 1.0f, 1.0f, 0.8f},
        };
        setTetMeshData(_cellIndices, _vertexPositions, _vertexColors);
    }
}

void buildVerticesSlim(
        const std::vector<glm::vec3>& vertices, const std::vector<uint32_t>& cellIndices,
        std::vector<VertexSlim>& verticesSlim) {
    verticesSlim.resize(vertices.size());
    const uint32_t numTets = cellIndices.size() / 4;
    for (uint32_t h_id = 0; h_id < numTets; h_id++) {
        for (uint32_t v_internal_id = 0; v_internal_id < 4; v_internal_id++) {
            uint32_t v_id = cellIndices.at(h_id * 4 + v_internal_id);
            verticesSlim.at(v_id).hs.push_back(h_id);
        }
    }
}

const int tetFaceTable[4][3] = {
        // Use consistent winding for faces at the boundary (normals pointing out of the cell - no arbitrary decisions).
        { 0, 1, 2 },
        { 1, 0, 3 },
        { 0, 2, 3 },
        { 2, 1, 3 },
};

void buildFacesSlim(
        const std::vector<glm::vec3>& vertices, const std::vector<uint32_t>& cellIndices,
        std::vector<FaceSlim>& facesSlim, std::vector<uint32_t>& isBoundaryFace) {
    struct TempFace {
        uint32_t vertexId[3];
        uint32_t faceId;

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

    const uint32_t numTets = cellIndices.size() / 4;
    std::vector<FaceSlim> totalFaces(numTets * 4);
    std::vector<TempFace> tempFaces(numTets * 4);

    FaceSlim face{};
    for (uint32_t cellId = 0; cellId < numTets; ++cellId) {
        for (uint32_t faceIdx = 0; faceIdx < 4; faceIdx++){
            for (uint32_t vertexIdx = 0; vertexIdx < 3; vertexIdx++) {
                face.vs[vertexIdx] = cellIndices.at(cellId * 4 + tetFaceTable[faceIdx][vertexIdx]);
            }

            uint32_t faceId = 6 * cellId + faceIdx;
            totalFaces[faceId] = face;
            std::sort(face.vs, face.vs + 3);
            tempFaces[faceId] = TempFace{
                    { face.vs[0], face.vs[1], face.vs[2] }, faceId
            };
        }
    }
    std::sort(tempFaces.begin(), tempFaces.end());

    facesSlim.reserve(tempFaces.size() / 2);
    uint32_t numFaces = 0;
    for (uint32_t i = 0; i < tempFaces.size(); ++i) {
        if (i == 0 || tempFaces[i] != tempFaces[i - 1]) {
            face = totalFaces[tempFaces[i].faceId];
            facesSlim.push_back(face);
            isBoundaryFace.push_back(true);
            numFaces++;
        } else {
            isBoundaryFace[numFaces - 1] = false;
        }
    }
}

void TetMesh::rebuildInternalRepresentationIfNecessary_Slim() {
    if (verticesSlim.empty()) {
        buildVerticesSlim(vertexPositions, cellIndices, verticesSlim);
        buildFacesSlim(vertexPositions, cellIndices, facesSlim, facesBoundarySlim);
    }

    if (dirty) {
        //updateMeshTriangleIntersectionDataStructure_Slim();
        dirty = false;
    }
}
