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

#ifndef DIFFTETVR_TETMESHING_HPP
#define DIFFTETVR_TETMESHING_HPP

#include <Math/Geometry/AABB3.hpp>
#include "../ColorStorage.hpp"

class TetMesh;

/**
 * TetMeshingApp specifies the external application to use for tetrahedralization.
 * - fTetWild: https://github.com/wildmeshing/fTetWild
 * - TetGen: https://wias-berlin.de/software/index.jsp?id=TetGen&lang=1
 * DiffTetVR only invokes the executables of these applications via the command-line interface.
 * It is thus not considered a derived work, and its source code is not affected by the licenses of the applications.
 */
enum class TetMeshingApp {
    FTETWILD, TETGEN
};

/// https://github.com/wildmeshing/fTetWild?tab=readme-ov-file#command-line-switches
struct FTetWildParams {
    double relativeIdealEdgeLength = 0.05; // -l
    double epsilon = 1e-3; // -e
    bool skipSimplify = false; // --skip-simlify
    bool coarsen = false; // --coarsen
};

/// See:
/// - https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html
/// - https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual004.html#sec30
struct TetGenParams {
    bool useSteinerPoints = true; // -q; to remove badly-shaped tetrahedra
    bool useRadiusEdgeRatioBound = false;
    double radiusEdgeRatioBound = 1.2; // radius-edge ratio bound
    bool useMaximumVolumeConstraint = false; // -a
    double maximumTetrahedronVolume = 1.0;
    bool coarsen = false; // -R
    double maximumDihedralAngle = 165.0; // -o/
    // -O; mesh optimization settings
    int meshOptimizationLevel = 2; // Between 0 and 10.
    bool useEdgeAndFaceFlips = true;
    bool useVertexSmoothing = true;
    bool useVertexInsertionAndDeletion = true;
};

/**
 * Tetrahedralizes a grid using the external application TetGen and stores it in a passed tet mesh object.
 * @param tetMesh The tet mesh to store the tetrahedralized data in.
 * @param gridAabb The AABB of the grid in world space.
 * @param xs The number of vertices in x direction.
 * @param ys The number of vertices in x direction.
 * @param zs The number of vertices in x direction.
 * @param constColor The constant color at all vertices.
 * @param colorStorage Whether to use per-vertex or per-cell colors.
 * @param params The meshing parameters.
 * @return Whether generating the mesh was successful.
 */
bool tetrahedralizeGridFTetWild(
        TetMesh* tetMesh,
        const sgl::AABB3& gridAabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor,
        ColorStorage colorStorage, const FTetWildParams& params);

/**
 * Tetrahedralizes a grid using the external application TetGen and stores it in a passed tet mesh object.
 * @param tetMesh The tet mesh to store the tetrahedralized data in.
 * @param gridAabb The AABB of the grid in world space.
 * @param xs The number of vertices in x direction.
 * @param ys The number of vertices in x direction.
 * @param zs The number of vertices in x direction.
 * @param constColor The constant color at all vertices.
 * @param colorStorage Whether to use per-vertex or per-cell colors.
 * @param params The meshing parameters.
 * @return Whether generating the mesh was successful.
 */
bool tetrahedralizeGridTetGen(
        TetMesh* tetMesh,
        const sgl::AABB3& gridAabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor,
        ColorStorage colorStorage, const TetGenParams& params);

#endif //DIFFTETVR_TETMESHING_HPP
