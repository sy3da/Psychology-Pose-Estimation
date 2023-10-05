/****************************************************************************************/
// COPYRIGHT AND CONFIDENTIALITY NOTICE
// SONY DEPTHSENSING SOLUTIONS CONFIDENTIAL INFORMATION
//
// All rights reserved to Sony Depthsensing Solutions SA/NV, a
// company incorporated and existing under the laws of Belgium, with
// its principal place of business at Boulevard de la Plainelaan 11,
// 1050 Brussels (Belgium), registered with the Crossroads bank for
// enterprises under company number 0811 784 189
//
// This file is part of vitalslib Library, which is proprietary
// and confidential information of Sony Depthsensing Solutions SA/NV.


// Copyright (c) 2020 Sony Depthsensing Solutions SA/NV
/****************************************************************************************/

#pragma once

#ifndef vitalslib_PLATFORM_INCLUDED
#define vitalslib_PLATFORM_INCLUDED

// Supported compiler
#define vitalslib_COMPILER_MSVC 1
#define vitalslib_COMPILER_GCC 2
#define vitalslib_COMPILER_CLANG 3
#define vitalslib_COMPILER_INTEL 4

// Supported platform
#define vitalslib_PLATFORM_WIN32 1
#define vitalslib_PLATFORM_WIN64 2
#define vitalslib_PLATFORM_LINUX_x86 3
#define vitalslib_PLATFORM_LINUX_x64 4
#define vitalslib_PLATFORM_LINUX_ARM_x64 5

// Finds the compiler type and version.
#if defined(_MSC_VER)
#define vitalslib_COMPILER vitalslib_COMPILER_MSVC
#define vitalslib_COMPILER_VERSION _MSC_VER
#elif defined(__INTEL_COMPILER)
#define vitalslib_COMPILER vitalslib_COMPILER_INTEL
#define vitalslib_COMPILER_VERSION __INTEL_COMPILER
#elif defined(__clang__)
#define vitalslib_GEN_VERSION(major, minor, patch) (((major)*100) + ((minor)*10) + (patch))
#define vitalslib_COMPILER_VERSION                                                                         \
    vitalslib_GEN_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#define vitalslib_COMPILER vitalslib_COMPILER_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
#define vitalslib_GEN_VERSION(major, minor, patch) (((major)*100) + ((minor)*10) + (patch))
#define vitalslib_COMPILER_VERSION vitalslib_GEN_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#define vitalslib_COMPILER vitalslib_COMPILER_GCC
#else
#error "Compilation error: Unsupported compiler."
#endif

/* Finds the platform - architecture
*/
#if defined(__WIN64__) || defined(_WIN64)
#define vitalslib_PLATFORM vitalslib_PLATFORM_WIN64
#elif defined(__WIN32__) || defined(_WIN32)
#define vitalslib_PLATFORM vitalslib_PLATFORM_WIN32
#elif defined(__linux__) || defined(__LINUX__)
#define vitalslib_PLATFORM_LINUX
#if defined(__x86) || defined(__x86__) || defined(__i386__) || defined(__i486__) ||                     \
    defined(__i586__) || defined(__i686__)
#define vitalslib_PLATFORM vitalslib_PLATFORM_LINUX_x86
#elif defined(__x86_64) || defined(__amd64) || defined(__amd64__)
#define vitalslib_PLATFORM vitalslib_PLATFORM_LINUX_x64
#elif defined(__aarch64__)
#define vitalslib_PLATFORM vitalslib_PLATFORM_LINUX_ARM_x64
#else
#error "Compilation error: Unsupported version of Linux platform."
#endif
#else
#error "Compilation error: Unsupported platform."
#endif

// Windows 32 or 64 bits defined.
#if (vitalslib_PLATFORM == vitalslib_PLATFORM_WIN32 || vitalslib_PLATFORM == vitalslib_PLATFORM_WIN64)
#define vitalslib_PLATFORM_WINDOWS
#endif

// Calling conventions
#if (vitalslib_PLATFORM == vitalslib_PLATFORM_WIN32 || vitalslib_PLATFORM == vitalslib_PLATFORM_WIN64) &&           \
    !defined(__MINGW32__) && !defined(__CYGWIN__)
#define vitalslib_DECL __stdcall
#else
#define vitalslib_DECL
#endif

// Shared library export definitions
#if (vitalslib_COMPILER == vitalslib_COMPILER_MSVC)
#define vitalslib_DLL_EXPORT __declspec(dllexport)
#define vitalslib_DLL_IMPORT __declspec(dllimport)
#else
#define vitalslib_DLL_EXPORT
#define vitalslib_DLL_IMPORT
#endif

// Import/export macros
#ifdef VITALSLIB_EXPORTS
#define vitalslib_API vitalslib_DLL_EXPORT
#else
#define vitalslib_API vitalslib_DLL_IMPORT
#endif

// Enable/disable warnings; use sparingly and with caution
#if (vitalslib_COMPILER == vitalslib_COMPILER_MSVC)

#define vitalslib_DISABLE_WARNING_NOINLINE
#define vitalslib_DISABLE_WARNING_INT_IN_BOOL_CONTEXT
#define vitalslib_DISABLE_WARNING_MAYBE_UNINITIALIZED
#define vitalslib_DISABLE_WARNING_STRICT_ALIASING

#define vitalslib_DISABLE_WARNING_TERMINATE __pragma(warning(disable : 4297))

#define vitalslib_DISABLE_WARNING_UNUSED_FUNC

#define vitalslib_DISABLE_WARNING_ASSIGN_COND_EXPR __pragma(warning(disable : 4706))

#define vitalslib_DISABLE_WARNING_DEPRECATED_DECLARATIONS __pragma(warning(disable : 4996))

#define vitalslib_PUSH_AND_DISABLE_WARNING_ALL __pragma(warning(push, 0))

#define vitalslib_WARNING_PUSH __pragma(warning(push))

#define vitalslib_WARNING_POP __pragma(warning(pop))
#elif (vitalslib_COMPILER == vitalslib_COMPILER_GCC)

#define vitalslib_DISABLE_WARNING_NOINLINE

#define vitalslib_DISABLE_WARNING_STRICT_ALIASING _Pragma("GCC diagnostic ignored \"-Wstrict-aliasing\"")

#define vitalslib_DISABLE_WARNING_TERMINATE _Pragma("GCC diagnostic ignored \"-Wterminate\"")

#define vitalslib_DISABLE_WARNING_UNUSED_FUNC _Pragma("GCC diagnostic ignored \"-Wunused-function\"")

#define vitalslib_DISABLE_WARNING_MAYBE_UNINITIALIZED                                                      \
    _Pragma("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")

#define vitalslib_DISABLE_WARNING_INT_IN_BOOL_CONTEXT                                                      \
    _Pragma("GCC diagnostic ignored \"-Wint-in-bool-context\"")

#define vitalslib_DISABLE_WARNING_ASSIGN_COND_EXPR

#define vitalslib_DISABLE_WARNING_DEPRECATED_DECLARATIONS                                                  \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")

#define vitalslib_PUSH_AND_DISABLE_WARNING_ALL                                                             \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wall\"")                          \
        _Pragma("GCC diagnostic ignored \"-Wignored-attributes\"")                                      \
            _Pragma("GCC diagnostic ignored \"-Wmisleading-indentation\"")                              \
                _Pragma("GCC diagnostic ignored \"-Wunused-local-typedefs\"")                           \
                    _Pragma("GCC diagnostic ignored \"-Wsign-compare\"")                            \

#define vitalslib_WARNING_PUSH _Pragma("GCC diagnostic push")

#define vitalslib_WARNING_POP _Pragma("GCC diagnostic pop")

#elif (vitalslib_COMPILER == vitalslib_COMPILER_CLANG)

#define vitalslib_DISABLE_WARNING_NOINLINE

#define vitalslib_DISABLE_WARNING_STRICT_ALIASING _Pragma("clang diagnostic ignored \"-Wstrict-aliasing\"")

#define vitalslib_DISABLE_WARNING_INT_IN_BOOL_CONTEXT
#define vitalslib_DISABLE_WARNING_MAYBE_UNINITIALIZED
#define vitalslib_DISABLE_WARNING_TERMINATE

#define vitalslib_DISABLE_WARNING_UNUSED_FUNC

#define vitalslib_DISABLE_WARNING_ASSIGN_COND_EXPR

#define vitalslib_DISABLE_WARNING_DEPRECATED_DECLARATIONS                                                  \
    _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")

#define vitalslib_PUSH_AND_DISABLE_WARNING_ALL                                                             \
    _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wall\"")                      \
        _Pragma("clang diagnostic ignored \"-Wextra\"")                                                 \
            _Pragma("clang diagnostic ignored \"-Wexceptions\"")


#define vitalslib_WARNING_PUSH _Pragma("clang diagnostic push")

#define vitalslib_WARNING_POP _Pragma("clang diagnostic pop")

#elif (vitalslib_COMPILER == vitalslib_COMPILER_INTEL)

#define vitalslib_DISABLE_WARNING_NOINLINE __pragma(warning(disable : 2196))
#define vitalslib_DISABLE_WARNING_INT_IN_BOOL_CONTEXT
#define vitalslib_DISABLE_WARNING_STRICT_ALIASING
#define vitalslib_DISABLE_WARNING_TERMINATE
#define vitalslib_DISABLE_WARNING_UNUSED_FUNC
#define vitalslib_DISABLE_WARNING_MAYBE_UNINITIALIZED
#define vitalslib_DISABLE_WARNING_ASSIGN_COND_EXPR
#define vitalslib_DISABLE_WARNING_DEPRECATED_DECLARATIONS
#define vitalslib_PUSH_AND_DISABLE_WARNING_ALL __pragma(warning(push))
#define vitalslib_WARNING_PUSH __pragma(warning(push))
#define vitalslib_WARNING_POP __pragma(warning(pop))
#else

#define vitalslib_DISABLE_WARNING_NOINLINE
#define vitalslib_DISABLE_WARNING_INT_IN_BOOL_CONTEXT
#define vitalslib_DISABLE_WARNING_STRICT_ALIASING
#define vitalslib_DISABLE_WARNING_TERMINATE
#define vitalslib_DISABLE_WARNING_UNUSED_FUNC
#define vitalslib_DISABLE_WARNING_MAYBE_UNINITIALIZED
#define vitalslib_DISABLE_WARNING_ASSIGN_COND_EXPR
#define vitalslib_DISABLE_WARNING_DEPRECATED_DECLARATIONS
#define vitalslib_PUSH_AND_DISABLE_WARNING_ALL
#define vitalslib_WARNING_PUSH
#define vitalslib_WARNING_POP

#endif
// enable/disable warnings

#endif