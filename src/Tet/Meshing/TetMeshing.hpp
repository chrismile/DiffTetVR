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

class TetMesh;

/**
 * TetMeshingApp specifies the external application to use for tetrahedralization.
 * - fTetWild: https://github.com/wildmeshing/fTetWild
 * - TetGen: https://wias-berlin.de/software/index.jsp?id=TetGen&lang=1
 */
/// The external application to use for tetrahedralization.
enum class TetMeshingApp {
    FTETWILD, TETGEN
};

/**
 * Tetrahedralizes a grid using an external application and stores it in a passed tet mesh object.
 * @param tetMesh The tet mesh to store the tetrahedralized data in.
 * @param tetMeshingApp The external application that should be used for tetrahedralization.
 * @param gridAabb The AABB of the grid in world space.
 * @param xs The number of vertices in x direction.
 * @param ys The number of vertices in x direction.
 * @param zs The number of vertices in x direction.
 * @param constColor The constant color at all vertices.
 * @return Whether generating the mesh was successful.
 */
bool tetrahedralizeGrid(
        TetMesh* tetMesh, TetMeshingApp tetMeshingApp,
        const sgl::AABB3& gridAabb, uint32_t xs, uint32_t ys, uint32_t zs, const glm::vec4& constColor);

#endif //DIFFTETVR_TETMESHING_HPP
