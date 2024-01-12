/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#include <cmath>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Utils/File/Logfile.hpp>

#include "LoadersUtil.hpp"

void swapEndianness(uint8_t* byteArray, size_t sizeInBytes, size_t bytesPerEntry) {
    /*
     * Variable length arrays (VLAs) are a C99-only feature, and supported by GCC and Clang merely as C++ extensions.
     * Visual Studio reports the following error when attempting to use VLAs:
     * "error C3863: array type 'uint8_t [this->8<L_ALIGN>8]' is not assignable".
     * Consequently, we will assume 'sizeInBytes' is less than or equal to 8 (i.e., 64 bit values).
     */
    if (bytesPerEntry > 8) {
        sgl::Logfile::get()->throwError("Error in swapEndianness: sizeInBytes is larger than 8.");
        return;
    }
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, sizeInBytes, bytesPerEntry), [&byteArray, bytesPerEntry](auto const& r) {
        uint8_t swappedEntry[8];
        for (auto i = r.begin(); i != r.end(); i += r.grainsize()) {
#else
    uint8_t swappedEntry[8];
#pragma omp parallel for shared(byteArray, sizeInBytes, bytesPerEntry) private(swappedEntry) default(none)
    for (size_t i = 0; i < sizeInBytes; i += bytesPerEntry) {
#endif
        for (size_t j = 0; j < bytesPerEntry; j++) {
            swappedEntry[j] = byteArray[i + bytesPerEntry - j - 1];
        }
        for (size_t j = 0; j < bytesPerEntry; j++) {
            byteArray[i + j] = swappedEntry[j];
        }
    }
#ifdef USE_TBB
    });
#endif
}
