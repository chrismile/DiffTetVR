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

#include <fstream>

#include <Utils/Events/Stream/Stream.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/LineReader.hpp>

#include "TxtTetLoader.hpp"

bool TxtTetLoader::loadFromFile(
        const std::string& filePath, std::vector<uint32_t>& cellIndices,
        std::vector<glm::vec3>& vertexPositions, std::vector<glm::vec4>& vertexColors) {
    uint8_t* buffer = nullptr;
    size_t length = 0;
    bool loaded = sgl::loadFileFromSource(filePath, buffer, length, true);
    if (!loaded) {
        return false;
    }

    sgl::LineReader lineReader(reinterpret_cast<const char*>(buffer), length);
    std::vector<std::string> linesInfo;
    std::vector<uint32_t> cellIndicesVector;
    std::vector<float> vertexPositionsVector;
    std::vector<float> vertexColorsVector;
    while (lineReader.isLineLeft()) {
        lineReader.readVectorLine<std::string>(linesInfo);
        if (linesInfo.size() != 2) {
            sgl::Logfile::get()->writeError("Error in TxtTetLoader::loadFromFile: Invalid header line.");
            delete[] buffer;
            return false;
        }
        const std::string& key = linesInfo.at(0);
        const auto numEntries = sgl::fromString<uint32_t>(linesInfo.at(1));
        if (key == "cellIndices") {
            cellIndices.reserve(4 * numEntries);
            for (uint32_t i = 0; i < numEntries; i++) {
                lineReader.readVectorLine<uint32_t>(cellIndicesVector);
                if (cellIndicesVector.size() != 4) {
                    sgl::Logfile::get()->writeError("Error in TxtTetLoader::loadFromFile: Invalid number of indices.");
                    delete[] buffer;
                    return false;
                }
                for (auto entry : cellIndicesVector) {
                    cellIndices.push_back(entry);
                }
            }
        } else if (key == "vertexPositions") {
            vertexPositions.reserve(numEntries);
            for (uint32_t i = 0; i < numEntries; i++) {
                lineReader.readVectorLine<float>(vertexPositionsVector);
                if (vertexPositionsVector.size() != 3) {
                    sgl::Logfile::get()->writeError(
                            "Error in TxtTetLoader::loadFromFile: Invalid number of vertex position entries.");
                    delete[] buffer;
                    return false;
                }
                vertexPositions.emplace_back(
                        vertexPositionsVector.at(0), vertexPositionsVector.at(1), vertexPositionsVector.at(2));
            }
        } else if (key == "vertexColors") {
            vertexColors.reserve(numEntries);
            for (uint32_t i = 0; i < numEntries; i++) {
                lineReader.readVectorLine<float>(vertexColorsVector);
                if (vertexColorsVector.size() != 4) {
                    sgl::Logfile::get()->writeError(
                            "Error in TxtTetLoader::loadFromFile: Invalid number of vertex color entries.");
                    delete[] buffer;
                    return false;
                }
                vertexColors.emplace_back(
                        vertexColorsVector.at(0), vertexColorsVector.at(1),
                        vertexColorsVector.at(2), vertexColorsVector.at(3));
            }
        }
    }

    delete[] buffer;
    return true;
}

bool TxtTetLoader::peekSizes(const std::string& filePath, size_t& numCells, size_t& numVertices) {
    std::ifstream file;
    file.open(filePath);

    // Unfortunately, there's no good way to peek at the sizes for the text format, as we cannot skip the data.
    // Thus, check if the file is moderately large, and in that case load and parse the file completely.
    file.seekg (0, std::ios::end);
    auto length = file.tellg();
    file.seekg (0, std::ios::beg);
    file.close();

    // Maximum size: 1MiB.
    if (length > 1024 * 1024 * 1024) {
        return false;
    }

    std::vector<uint32_t> cellIndices;
    std::vector<glm::vec3> vertexPositions;
    std::vector<glm::vec4> vertexColors;
    loadFromFile(filePath, cellIndices, vertexPositions, vertexColors);

    numCells = cellIndices.size() / 4;
    numVertices = vertexPositions.size();
    return true;
}

bool TxtTetWriter::saveToFile(
        const std::string& filePath, const std::vector<uint32_t>& cellIndices,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec4>& vertexColors) {
    std::ofstream file(filePath.c_str(), std::ofstream::binary);
    if (!file.is_open()) {
        sgl::Logfile::get()->writeError(
                std::string() + "Error in TxtTetWriter::saveToFile: File \""
                + filePath + "\" could not be opened for writing.");
        return false;
    }

    file << "cellIndices " << (cellIndices.size() / 4) << "\n";
    for (size_t i = 0; i < cellIndices.size(); i += 4) {
        file << cellIndices.at(i) << " " << cellIndices.at(i + 1) << " "
                << cellIndices.at(i + 2) << " " << cellIndices.at(i + 3) << "\n";
    }

    file << "vertexPositions " << vertexPositions.size() << "\n";
    for (const auto& v : vertexPositions) {
        file << v.x << " " << v.y << " " << v.z << "\n";
    }

    file << "vertexColors " << vertexColors.size() << "\n";
    for (const auto& v : vertexColors) {
        file << v.x << " " << v.y << " " << v.z << " " << v.w << "\n";
    }

    file.close();
    return true;
}
