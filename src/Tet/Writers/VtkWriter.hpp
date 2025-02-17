/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

#ifndef DIFFTETVR_VTKWRITER_HPP
#define DIFFTETVR_VTKWRITER_HPP

#include <string>
#include <vector>
#include <memory>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

class TetMesh;
typedef std::shared_ptr<TetMesh> TetMeshPtr;

namespace sgl { namespace vk {
class Renderer;
}}

class VtkWriter {
public:
    /**
     * @return The file ending of the format.
     */
    static std::string getOutputFormatEnding() { return ".vtk"; }

    /**
     * Intializes the file writer.
     * @param filename The file name of the file to write to.
     */
    void initializeWriter(const std::string& filename, bool isBinaryVtk = true);
    /**
     * Writes the data of the current time step to a file. Currently, only tetrahedral meshes are supported.
     */
    void writeNextTimeStep(
            const std::vector<uint32_t>& cellIndices,
            const glm::vec3* vertexPositions,
            const glm::vec4* vertexColors,
            const glm::vec4* cellColors,
            const glm::vec3* vertexPositionGradients,
            const glm::vec4* vertexColorGradients,
            const glm::vec4* cellColorGradients,
            int numPoints);

    /// Writes the time step using the data from a TetMesh object.
    void writeNextTimeStep(sgl::vk::Renderer* renderer, const TetMeshPtr& tetMesh);

private:
    void writeVtkHeader(FILE* file) const;
    void writePointCoordinates(FILE* file, const glm::vec3* vertexPositions, int numPoints) const;
    void writeCells(FILE* file, const std::vector<uint32_t>& cellIndices) const;
    void writeDataVector(
            FILE* file, const glm::vec3* vectorData, int numPoints, const std::string& vectorName) const;
    void writeDataScalar(
            FILE* file, const float* scalarData, int numPoints, const std::string& scalarName) const;
    void writeDataColor(
            FILE* file, const glm::vec4* colorDataVec4, int numPoints, const std::string& scalarName) const;

    std::string filename;
    bool isBinaryVtk;
    int timeStepNumber = 0;
};

#endif //DIFFTETVR_VTKWRITER_HPP
