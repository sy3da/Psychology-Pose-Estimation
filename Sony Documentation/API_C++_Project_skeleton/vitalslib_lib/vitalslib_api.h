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

/**
 * \file vitalslib_api.h
 *  Main file for the vitalslib API
 */

#pragma once

#ifndef vitalslib_API_INCLUDED
#define vitalslib_API_INCLUDED

#include "vitalslib_types.h"
#include "vitalslib_platform.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

	/**
	 * \brief  Return the version number of vitalslib.
	 *
	 * \param[out]  version             A pointer to a char array containing the version.
	 *                                  This will contain the version after this call.
	 * \param[out]  error_details       Pointer to an allocated vitalslib_error_details struct that will
	 * contain the error code and a human readable error message. Use NULL to ignore errors.
	 *
	 * \returns     true if the call succeeded
	 */
	vitalslib_API bool vitalslib_DECL vitalslib_get_version(
		vitalslib_OUT const char** version,
		vitalslib_OUT vitalslib_error_details* error_details);


	/**
	 * \brief  Create an instance of vitalslib.
	 *
	 * \param[in]	camera_parameters	Array of allocated vitalslib_camera_parameters structs,
	 *									contains the camera parameters to use. The size must be equal to
	 *									camera_count.
	 * \param[in]	config_path			Relative vitalslib configuration path, by default if field empty read "./vitalslib.config.xml path"
	 * \param[out]	handle				Pointer to a handle pointer.
	 * \param[out]	error_details		Pointer to an allocated vitalslib_error_details struct that will
	 *contain the error code and a human readable error message. Use NULL to ignore errors.
	 *
	 * \returns		true if the call succeeded
	 *
	 * \post
	 *	- if successful, handle is a valid vitalslib handle which can be used by the other API functions
	 *	- if not handle is a NULL
	 */
	vitalslib_API bool vitalslib_DECL vitalslib_init(
		vitalslib_IN const vitalslib_camera_parameters* const camera_parameters,
		vitalslib_IN const char* config_path,
		vitalslib_OUT vitalslib_handle** handle,
		vitalslib_OUT vitalslib_error_details* error_details);


	/**
	 * \brief  Release an instance of vitalslib.
	 *
	 * \param[in,out]	handle			Pointer to a valid handle pointer.
	 * \param[out]		error_details	Pointer to an allocated vitalslib_error_details struct that will
	 *contain the error code and a human readable error message. Use NULL to ignore errors.
	 *
	 * \returns		true if the call succeeded
	 *
	 * \post
	 *	- handle set to NULL if the destruction was successful
	 */
	vitalslib_API bool vitalslib_DECL vitalslib_release(
		vitalslib_IN_OUT vitalslib_handle** handle,
		vitalslib_OUT vitalslib_error_details* error_details);

	/**
	 * \brief Process frames
	 *
	 * \param[in,out]	handle			handle created by vitalslib_create().
	 * \param[in]		frames			Array of allocated vitalslib_frame structs, contains the
	 *									newest frames to feed to the pipeline.
	 * \param[out]		error_details	Pointer to an allocated vitalslib_error_details struct that will
	 *contain the error code and a human readable error message. Use NULL to ignore errors.
	 *
	 * \returns		true if the call succeeded
	 *
	 */
	vitalslib_API bool vitalslib_DECL vitalslib_compute(
		vitalslib_IN_OUT vitalslib_handle* handle,
		vitalslib_IN const vitalslib_frame* const frames,
		vitalslib_OUT vitalslib_error_details* error_details);


	/** EXAMPLE
	* \brief Get the carlib dataset
	*
	* \param[in]	handle				handle created by vitalslib_create().
	* \param[out]	heart_rate	        heart_rate
	* \param[out]	error_details		Pointer to an allocated vitalslib_error_details struct that will contain
	*the error code and a human readable error message. Use NULL to ignore errors.
	*
	* \returns		true if the call succeeded
	*
	*/
	vitalslib_API bool vitalslib_DECL vitalslib_get_heart_rate(
		vitalslib_IN const vitalslib_handle* const handle,
		vitalslib_OUT float* heart_rate,
		vitalslib_OUT vitalslib_error_details* error_details);

	/** EXAMPLE
	* \brief Get the icm dataset
	*
	* \param[in]	handle				handle created by vitalslib_create().
	* \param[out]	respiratory_rate	respiratory_rate
	* \param[out]	error_details		Pointer to an allocated vitalslib_error_details struct that will contain
	*the error code and a human readable error message. Use NULL to ignore errors.
	*
	* \returns		true if the call succeeded
	*
	*/
	vitalslib_API bool vitalslib_DECL vitalslib_get_respiratory_rate(
		vitalslib_IN const vitalslib_handle* const handle,
		vitalslib_OUT float* respiratory_rate,
		vitalslib_OUT vitalslib_error_details* error_details);


#ifdef __cplusplus
}
#endif

#endif
