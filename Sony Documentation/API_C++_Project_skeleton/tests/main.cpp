// COPYRIGHT AND CONFIDENTIALITY NOTICE
// SONY DEPTHSENSING SOLUTIONS CONFIDENTIAL INFORMATION
//
// All rights reserved to Sony Depthsensing Solutions SA/NV, a
// company incorporated and existing under the laws of Belgium, with
// its principal place of business at Boulevard de la Plainelaan 11,
// 1050 Brussels (Belgium), registered with the Crossroads bank for
// enterprises under company number 0811 784 189
//
// This file is part of the vitalslib, which is proprietary
// and confidential information of Sony Depthsensing Solutions SA/NV.
//
// Copyright (c) 2021 Sony Depthsensing Solutions SA/NV

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>

#pragma warning(push)
#pragma warning(disable : 4702)
#endif

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"


#ifdef __ORBIS__

#include <kernel.h>

// Set the heap size to "unlimited"
size_t sceLibcHeapSize = SCE_LIBC_HEAP_SIZE_EXTENDED_ALLOC_NO_LIMIT;
unsigned int sceLibcHeapExtendedAlloc = 1;
#endif

int main(int argc, char * const argv[])
{

#if defined(_WIN32) || defined(_WIN64)
    // Disables "XXX.exe has stopped working" windows which occur when a test segfaults.
    // This stalls the execution of the testsuite for no good reason.
    SetErrorMode( SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX );
#endif

    return Catch::Session().run( argc, argv );
}

#if defined(_WIN32) || defined(_WIN64)
#pragma warning(pop)
#endif