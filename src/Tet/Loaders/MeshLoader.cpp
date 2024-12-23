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

#include <fstream>

#include <Utils/StringUtils.hpp>
#include <Utils/Events/Stream/Stream.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/LineReader.hpp>

#include "MeshLoader.hpp"

#pragma pack(1)
struct MeshFormatVertex {
    float x;
    float y;
    float z;
    int32_t label;
};

bool MeshLoader::parseVertices(
        const std::string& filePath, sgl::LineReader& lineReader,
        size_t numVertices, std::vector<glm::vec3>& vertexPositions) {
    MeshFormatVertex vertex{};
    for (size_t vertexIdx = 0; vertexIdx < numVertices; vertexIdx++) {
        lineReader.readStructLine(vertex.x, vertex.y, vertex.z, vertex.label);
        vertexPositions.emplace_back(vertex.x, vertex.y, vertex.z);
    }
    return true;
}

#pragma pack(1)
struct MeshFormatTet {
    uint32_t i0;
    uint32_t i1;
    uint32_t i2;
    uint32_t i3;
    int32_t label;
};

bool MeshLoader::parseTetrahedra(
        const std::string& filePath, sgl::LineReader& lineReader,
        size_t numTetrahedra, std::vector<uint32_t>& cellIndices) {
    MeshFormatTet tet{};
    for (size_t tetIdx = 0; tetIdx < numTetrahedra; tetIdx++) {
        lineReader.readStructLine(tet.i0, tet.i1, tet.i2, tet.i3, tet.label);
        // The default ordering in .mesh files is in different winding than what DiffTetVR expects.
        cellIndices.push_back(tet.i1 - 1);
        cellIndices.push_back(tet.i0 - 1);
        cellIndices.push_back(tet.i2 - 1);
        cellIndices.push_back(tet.i3 - 1);
    }
    return true;
}

#pragma pack(1)
struct MeshFormatTri {
    uint32_t i0;
    uint32_t i1;
    uint32_t i2;
    int32_t label;
};

bool MeshLoader::parseTriangles(
        const std::string& filePath, sgl::LineReader& lineReader, size_t numTriangles) {
    MeshFormatTri tri{};
    for (size_t triIdx = 0; triIdx < numTriangles; triIdx++) {
        lineReader.readStructLine(tri.i0, tri.i1, tri.i2, tri.label);
    }
    return true;
}

#pragma pack(1)
struct MeshFormatEdge {
    uint32_t i0;
    uint32_t i1;
    int32_t label;
};

bool MeshLoader::parseEdges(
        const std::string& filePath, sgl::LineReader& lineReader, size_t numTriangles) {
    MeshFormatEdge edge{};
    for (size_t triIdx = 0; triIdx < numTriangles; triIdx++) {
        lineReader.readStructLine(edge.i0, edge.i1, edge.label);
    }
    return true;
}

bool MeshLoader::loadFromFile(
        const std::string& filePath, std::vector<uint32_t>& cellIndices,
        std::vector<glm::vec3>& vertexPositions, std::vector<glm::vec4>& vertexColors) {
    uint8_t* buffer = nullptr;
    size_t length = 0;
    bool loaded = sgl::loadFileFromSource(filePath, buffer, length, true);
    if (!loaded) {
        return false;
    }
    sgl::LineReader lineReader(reinterpret_cast<const char*>(buffer), length);
    auto firstLine = sgl::stringTrimCopy(lineReader.readLine());
    if (firstLine != "MeshVersionFormatted 1") {
        sgl::Logfile::get()->writeErrorVar(
                "Error in MeshLoader::loadFromFile: Expected \"MeshVersionFormatted 1\" as first line in file \"",
                filePath, "\".");
        return false;
    }

    std::vector<std::string> sectionHeader;
    while (lineReader.isLineLeft()) {
        size_t numEntries = 0;
        sectionHeader.clear();
        lineReader.readVectorLine(sectionHeader);
        if (sectionHeader.at(0).front() == '#') {
            continue;
        } else if (sectionHeader.at(0) == "End") {
            break;
        } else if (sectionHeader.size() == 2) {
            numEntries = sgl::fromString<size_t>(sectionHeader.at(1));
        } else if (sectionHeader.size() == 1) {
            numEntries = lineReader.readScalarLine<size_t>();
        } else {
            sgl::Logfile::get()->writeErrorVar(
                    "Error in MeshLoader::loadFromFile: Invalid section header length in file \"", filePath, "\".");
            return false;
        }
        std::string sectionName = sectionHeader.at(0);
        if (sectionName == "Dimension") {
            if (numEntries != 3) {
                sgl::Logfile::get()->writeErrorVar(
                        "Error in MeshLoader::loadFromFile: Invalid number of dimensions in file \"", filePath, "\".");
                return false;
            }
        } else if (sectionName == "Vertices") {
            if (!parseVertices(filePath, lineReader, numEntries, vertexPositions)) {
                return false;
            }
        } else if (sectionName == "Tetrahedra") {
            if (!parseTetrahedra(filePath, lineReader, numEntries, cellIndices)) {
                return false;
            }
        } else if (sectionName == "Triangles") {
            if (!parseTriangles(filePath, lineReader, numEntries)) {
                return false;
            }
        } else if (sectionName == "Edges") {
            if (!parseEdges(filePath, lineReader, numEntries)) {
                return false;
            }
        } else {
            sgl::Logfile::get()->writeErrorVar(
                    "Error in MeshLoader::loadFromFile: Unsupported section \"",
                    sectionName, "\" in file \"", filePath, "\".");
            return false;
        }
    }
    // TODO: Support loading of color data.
    vertexColors.resize(vertexPositions.size(), glm::vec4(0.5f, 0.5f, 0.5f, 0.1f));

    delete[] buffer;
    return true;
}

bool MeshLoader::peekSizes(const std::string& filePath, size_t& numCells, size_t& numVertices) {
    return false;
}
