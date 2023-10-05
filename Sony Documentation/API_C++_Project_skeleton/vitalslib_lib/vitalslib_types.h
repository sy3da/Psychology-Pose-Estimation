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

/** \file vitalslib_types.h
*  Public types for the vitalslib API
*/

#pragma once

#ifndef vitalslib_TYPES_INCLUDED
#define vitalslib_TYPES_INCLUDED

#ifdef CUSTOM_STDINT
#include CUSTOM_STDINT
#else
#if (defined(_MSC_VER) && _MSC_VER > 1600) || !defined(_MSC_VER)
#ifdef __cplusplus
#include <cstdint>
#include <cstdlib>
#include <cstring>
#else
#include <cstdlib.h>
#include <cstring.h>
#include <stdint.h>
#endif
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

	/************************************************************************
	*                               General                                 *
	************************************************************************/

	/**
	* \brief Handle to a vitalslib instance
	*/
	typedef struct vitalslib_handle vitalslib_handle;

	/**
	*  \brief struct describing an Infrared image.
	*
	*/
	struct vitalslib_image
	{
		/// The image width, in pixels
		unsigned int width;
		/// The image height, in pixels
		unsigned int height;
		/// Array of pixel values
		int16_t* ptr;
	};

	/**
	* \brief Frame used as vitalslib input.
	*/
	struct vitalslib_frame
	{
		/// Depth image (must be properly allocated by the caller)
		vitalslib_image depth;
		/// Confidence image (must be properly allocated by the caller)
		vitalslib_image confidence;
		/// Timestamp of the frame
		unsigned long long timestamp;
	};

	/**
	* \brief Describes all the constant parameters for a given camera
	*/
	struct vitalslib_camera_parameters
	{
		/// Unique identifier of the camera
		unsigned int camera_id;
		/// the width of the confidence / depth map
		int width;
		/// the height of the confidence / depth map
		int height;
		/// the focal length along the x axis, expressed in pixel units
		float focal_x;
		/// the focal length along the y axis, expressed in pixel units
		float focal_y;
		/// the central point along the x axis, expressed in pixel units
		float central_x;
		/// the central point along the y axis, expressed in pixel units
		float central_y;
		/// the first radial distortion coefficient
		float k1;
		/// the second radial distortion coefficient
		float k2;
		/// the third radial distortion coefficient
		float k3;
		/// the fourth radial distortion coefficient
		float k4;
		/// the first tangential distortion coefficient
		float p1;
		/// the second tangential distortion coefficient
		float p2;
	};



	/************************************************************************
	*                               ERRORS                                  *
	************************************************************************/

	/**
	* \brief Return codes for vitalslib functions
	*/
	enum vitalslib_error_code
	{
		/// Everything went fine
		vitalslib_success = 0,
		/// Generic code for unforeseen errors
		vitalslib_other_failure,
		/// The function received a null handle
		vitalslib_null_handle,
		/// The function received a non-null handle that does not correspond to an allocated instance
		vitalslib_invalid_handle,
		/// Failed to initialize
		vitalslib_failed_to_initialize,
		/// Failed to shutdown
		vitalslib_failed_to_shutdown,
		/// Failed to compute
		vitalslib_failed_to_compute,
		/// At least one pointer for data passed to the function was null
		vitalslib_null_data_pointer,
		/// The number of error codes
		vitalslib_error_code_count
	};

	/**
	* \brief Describes an API call error.
	*/
	struct vitalslib_error_details
	{
		/// The error code. See vitalslib_errorcodes.h for the complete list
		vitalslib_error_code code;
		/// Human-readable message
		const char* message;
	};

	static const char* vitalslib_error_descriptions[vitalslib_error_code_count] = {
		/* vitalslib_success=0                   */ "success",
		/* vitalslib_other_failure               */ "undocumented error",
		/* vitalslib_null_handle                 */ "the given handle parameter is a null pointer",
		/* vitalslib_invalid_handle              */ "the given handle parameter is not a pointer that has been created by vitalslib_create()",
		/* vitalslib_failed_to_initialize        */ "initialization failed",
		/* vitalslib_failed_to_shutdown          */ "shutdown failed",
		/* vitalslib_failed_to_compute           */ "Failed to compute vitalslib",
		/* vitalslib_null_data_pointer           */ "pointer to user space data is a null pointer"
	};


#define vitalslib_IN
#define vitalslib_OUT
#define vitalslib_IN_OUT

#ifdef __cplusplus
}
#endif

#endif
