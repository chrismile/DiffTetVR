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

#include <Utils/StringUtils.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/Execute.hpp>

#include "../TetMesh.hpp"
#include "WriteSurfaceMesh.hpp"
#include "TetMeshing.hpp"

#ifdef BUILD_PYTHON_MODULE
std::string getLibraryPath(); // Defined in Module.cpp.
#endif

bool runTetrahedralizerApp(
        const std::string& appName, const std::string& exeName, const std::vector<std::string>& args) {
#ifdef BUILD_PYTHON_MODULE
    const std::string appPath = getLibraryPath() + "/" + exeName;
#else
    std::string appPath = sgl::FileUtils::get()->getExecutableDirectory() + exeName;
    if (!sgl::FileUtils::get()->exists(appPath)) {
        // For development builds, the fTetWild and TetGen executables may not be copied to the executable directory.
        if (appName == "fTetWild") {
            appPath = sgl::FileUtils::get()->joinPath(
                    sgl::FileUtils::get()->getParentFolderPath(sgl::AppSettings::get()->getDataDirectory()),
                    "third_party", appName, exeName);
        } else {
            std::string thirdPartyDir = sgl::FileUtils::get()->joinPath(
                    sgl::FileUtils::get()->getParentFolderPath(sgl::AppSettings::get()->getDataDirectory()),
                    "third_party");
            std::vector<std::string> thirdPartyDirChildren =
                    sgl::FileUtils::get()->getFilesInDirectoryVector(thirdPartyDir);
            std::string tetgenDir;
            for (const std::string& thirdPartyDirChild : thirdPartyDirChildren) {
                auto thirdPartyDirChildName = sgl::FileUtils::get()->getPureFilename(thirdPartyDirChild);
                if (sgl::startsWith(thirdPartyDirChildName, "tetgen")) {
                    tetgenDir = thirdPartyDirChild;
                }
            }
            appPath = sgl::FileUtils::get()->joinPath(tetgenDir, "bin", exeName);
        }
    }
#endif
    if (!sgl::FileUtils::get()->fileExists(appPath)) {
        sgl::Logfile::get()->writeErrorVar(
                "Error in generateTetMeshFromGrid: Could not find ", appName, " executable.");
        return false;
    }
    auto returnCode = sgl::executeProgram(appPath.c_str(), args);
    if (returnCode != 0) {
        return false;
    }
    return true;
}

void createGridSurfaceMesh(
        const sgl::AABB3& gridAabb, uint32_t xs, uint32_t ys, uint32_t zs,
        std::vector<uint32_t>& quadIndices, std::vector<glm::vec3>& vertexPositions) {
    // Create the surface mesh.
    std::vector<std::vector<uint32_t>> faceVertexIndices(6); // 0,1: xy planes, 2,3: xz planes, 4,5: yz planes
    faceVertexIndices.at(0).reserve(xs * ys);
    faceVertexIndices.at(1).reserve(xs * ys);
    faceVertexIndices.at(2).reserve(xs * zs);
    faceVertexIndices.at(3).reserve(xs * zs);
    faceVertexIndices.at(4).reserve(ys * zs);
    faceVertexIndices.at(5).reserve(ys * zs);
    for (uint32_t iz = 0; iz < zs; iz++) {
        for (uint32_t iy = 0; iy < ys; iy++) {
            for (uint32_t ix = 0; ix < xs; ix++) {
                if (ix != 0 && ix != xs - 1 && iy != 0 && iy != ys - 1 && iz != 0 && iz != zs - 1) {
                    continue;
                }
                if (iz == 0) {
                    faceVertexIndices.at(0).push_back(uint32_t(vertexPositions.size()));
                }
                if (iz == zs - 1) {
                    faceVertexIndices.at(1).push_back(uint32_t(vertexPositions.size()));
                }
                if (iy == 0) {
                    faceVertexIndices.at(2).push_back(uint32_t(vertexPositions.size()));
                }
                if (iy == ys - 1) {
                    faceVertexIndices.at(3).push_back(uint32_t(vertexPositions.size()));
                }
                if (ix == 0) {
                    faceVertexIndices.at(4).push_back(uint32_t(vertexPositions.size()));
                }
                if (ix == xs - 1) {
                    faceVertexIndices.at(5).push_back(uint32_t(vertexPositions.size()));
                }
                glm::vec3 p;
                p.x = gridAabb.min.x + (float(ix) / float(xs - 1)) * (gridAabb.max.x - gridAabb.min.x);
                p.y = gridAabb.min.y + (float(iy) / float(ys - 1)) * (gridAabb.max.y - gridAabb.min.y);
                p.z = gridAabb.min.z + (float(iz) / float(zs - 1)) * (gridAabb.max.z - gridAabb.min.z);
                vertexPositions.emplace_back(p);
            }
        }
    }
    // xy planes indices
    for (uint32_t i = 0; i <= 1; i++) {
        for (uint32_t iy = 0; iy < ys - 1; iy++) {
            for (uint32_t ix = 0; ix < xs - 1; ix++) {
                uint32_t i0 = faceVertexIndices.at(i).at((ix) + (iy) * xs);
                uint32_t i1 = faceVertexIndices.at(i).at((ix + 1) + (iy) * xs);
                uint32_t i2 = faceVertexIndices.at(i).at((ix + 1) + (iy + 1) * xs);
                uint32_t i3 = faceVertexIndices.at(i).at((ix) + (iy + 1) * xs);
                quadIndices.push_back(i0);
                quadIndices.push_back(i1);
                quadIndices.push_back(i2);
                quadIndices.push_back(i3);
            }
        }
    }
    // xz planes indices
    for (int i = 2; i <= 3; i++) {
        for (uint32_t iz = 0; iz < zs - 1; iz++) {
            for (uint32_t ix = 0; ix < xs - 1; ix++) {
                uint32_t i0 = faceVertexIndices.at(i).at((ix) + (iz) * xs);
                uint32_t i1 = faceVertexIndices.at(i).at((ix + 1) + (iz) * xs);
                uint32_t i2 = faceVertexIndices.at(i).at((ix + 1) + (iz + 1) * xs);
                uint32_t i3 = faceVertexIndices.at(i).at((ix) + (iz + 1) * xs);
                quadIndices.push_back(i0);
                quadIndices.push_back(i1);
                quadIndices.push_back(i2);
                quadIndices.push_back(i3);
            }
        }
    }
    // yz planes indices
    for (int i = 4; i <= 5; i++) {
        for (uint32_t iz = 0; iz < zs - 1; iz++) {
            for (uint32_t iy = 0; iy < ys - 1; iy++) {
                uint32_t i0 = faceVertexIndices.at(i).at((iy) + (iz) * ys);
                uint32_t i1 = faceVertexIndices.at(i).at((iy + 1) + (iz) * ys);
                uint32_t i2 = faceVertexIndices.at(i).at((iy + 1) + (iz + 1) * ys);
                uint32_t i3 = faceVertexIndices.at(i).at((iy) + (iz + 1) * ys);
                quadIndices.push_back(i0);
                quadIndices.push_back(i1);
                quadIndices.push_back(i2);
                quadIndices.push_back(i3);
            }
        }
    }
}

bool tetrahedralizeGridFTetWild(
        TetMesh* tetMesh,
        const sgl::AABB3& gridAabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor,
        ColorStorage ColorStorage, const FTetWildParams& params) {
    if (xs < 1 || ys < 1 || zs < 1) {
        return false;
    }
    std::vector<uint32_t> quadIndices;
    std::vector<glm::vec3> vertexPositions;
    createGridSurfaceMesh(gridAabb, xs, ys, zs, quadIndices, vertexPositions);

    // Create a temporary directory.
    std::string tmpDirectory = sgl::FileUtils::get()->joinPath(sgl::AppSettings::get()->getDataDirectory(), "tmp");
    sgl::FileUtils::get()->ensureDirectoryExists(tmpDirectory);

    std::string surfaceMeshFile = sgl::FileUtils::get()->joinPath(tmpDirectory, "tmp.obj");
    std::string tetMeshFile = sgl::FileUtils::get()->joinPath(tmpDirectory, "tmp.msh");

    // Write the surface mesh to a temporary file.
    saveQuadMeshObj(surfaceMeshFile, quadIndices, vertexPositions);
    surfaceMeshFile = sgl::FileUtils::get()->getPathAbsolute(surfaceMeshFile);
    tetMeshFile = sgl::FileUtils::get()->getPathAbsolute(tetMeshFile);

    // Now, run the fTetWild executable.
#if defined(_WIN32)
    const std::string exeName = "FloatTetwild_bin.exe";
#else
    const std::string exeName = "FloatTetwild_bin";
#endif
    std::vector<std::string> args = {
            exeName, "-i", surfaceMeshFile, "-o", tetMeshFile
    };
    if (params.relativeIdealEdgeLength != 0.05) {
        args.emplace_back("-l");
        args.emplace_back(std::to_string(params.relativeIdealEdgeLength));
    }
    if (params.epsilon != 1e-3) {
        args.emplace_back("-e");
        args.emplace_back(std::to_string(params.epsilon));
    }
    if (params.skipSimplify) {
        args.emplace_back("--skip-simlify");
    }
    if (params.coarsen) {
        args.emplace_back("--coarsen");
    }
    if (!runTetrahedralizerApp("fTetWild", exeName, args)) {
        return false;
    }

    // Load the generated tet mesh from disk.
    tetMesh->setNextLoaderUseConstColor(constColor, ColorStorage);
    bool loadingSucceeded = tetMesh->loadFromFile(tetMeshFile);
    if (!loadingSucceeded) {
        return false;
    }

    // Remove the temporary files.
    auto tmpFilePaths = sgl::FileUtils::get()->getFilesInDirectoryVector(tmpDirectory);
    for (const auto& filePath : tmpFilePaths) {
        if (!sgl::FileUtils::get()->isDirectory(filePath)) {
            sgl::FileUtils::get()->removeFile(filePath);
        }
    }

    return true;
}

bool tetrahedralizeGridTetGen(
        TetMesh* tetMesh,
        const sgl::AABB3& gridAabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor,
        ColorStorage ColorStorage, const TetGenParams& params) {
    if (xs < 1 || ys < 1 || zs < 1) {
        return false;
    }
    std::vector<uint32_t> quadIndices;
    std::vector<glm::vec3> vertexPositions;
    createGridSurfaceMesh(gridAabb, xs, ys, zs, quadIndices, vertexPositions);

    // Create a temporary directory.
    std::string tmpDirectory = sgl::FileUtils::get()->joinPath(sgl::AppSettings::get()->getDataDirectory(), "tmp");
    sgl::FileUtils::get()->ensureDirectoryExists(tmpDirectory);

    std::string surfaceMeshFile = sgl::FileUtils::get()->joinPath(tmpDirectory, "tmp.off");
    std::string tetMeshFile = sgl::FileUtils::get()->joinPath(tmpDirectory, "tmp.1.mesh");

    // Write the surface mesh to a temporary file.
    saveQuadMeshOff(surfaceMeshFile, quadIndices, vertexPositions);
    surfaceMeshFile = sgl::FileUtils::get()->getPathAbsolute(surfaceMeshFile);
    tetMeshFile = sgl::FileUtils::get()->getPathAbsolute(tetMeshFile);

    // Now, run the fTetWild executable.
#if defined(_WIN32)
    const std::string exeName = "tetgen.exe";
#else
    const std::string exeName = "tetgen";
#endif
    std::string commandLineSwitch = "-pg"; // g is used for outputting .mesh file.
    if (params.useSteinerPoints) {
        commandLineSwitch += "q";
        if (params.useRadiusEdgeRatioBound) {
            commandLineSwitch += std::to_string(params.radiusEdgeRatioBound);
        }
        if (params.useMaximumVolumeConstraint) {
            commandLineSwitch += "a";
            commandLineSwitch += std::to_string(params.maximumTetrahedronVolume);
        }
    }
    if (params.coarsen) {
        commandLineSwitch += "R";
    }
    if (params.maximumDihedralAngle != 0.0f) {
        commandLineSwitch += "-o/" + std::to_string(params.maximumDihedralAngle);
    }
    if (params.meshOptimizationLevel != 2
            || !params.useEdgeAndFaceFlips
            || !params.useVertexSmoothing
            || !params.useVertexInsertionAndDeletion) {
        int bitFlags = 0;
        if (params.useEdgeAndFaceFlips) {
            bitFlags |= 1;
        }
        if (params.useVertexSmoothing) {
            bitFlags |= 2;
        }
        if (params.useVertexInsertionAndDeletion) {
            bitFlags |= 4;
        }
        commandLineSwitch += "-O" + std::to_string(params.meshOptimizationLevel) + "/" + std::to_string(bitFlags);
    }

    std::vector<std::string> args = {
            exeName, commandLineSwitch, surfaceMeshFile
    };
    if (!runTetrahedralizerApp("TetGen", exeName, args)) {
        return false;
    }

    // Load the generated tet mesh from disk.
    tetMesh->setNextLoaderUseConstColor(constColor, ColorStorage);
    bool loadingSucceeded = tetMesh->loadFromFile(tetMeshFile);
    if (!loadingSucceeded) {
        return false;
    }

    // Remove the temporary files.
    auto tmpFilePaths = sgl::FileUtils::get()->getFilesInDirectoryVector(tmpDirectory);
    for (const auto& filePath : tmpFilePaths) {
        if (!sgl::FileUtils::get()->isDirectory(filePath)) {
            sgl::FileUtils::get()->removeFile(filePath);
        }
    }

    return true;
}
