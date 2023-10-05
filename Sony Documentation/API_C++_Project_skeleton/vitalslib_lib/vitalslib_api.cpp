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

#include "vitalslib_api.h"
#include "vitalslib_types.h"
#include "version.h"
#include "core/pipeline.h"

#include <iostream>



#include <experimental/filesystem>

namespace
{
static const std::string vitalslib_version_str = std::to_string(vitalslib_LIBRARY_VERSION_MAJOR) + "." +
											  std::to_string(vitalslib_LIBRARY_VERSION_MINOR) + "." +
											  std::to_string(vitalslib_LIBRARY_VERSION_PATCH) + "." +
											  std::to_string(vitalslib_LIBRARY_VERSION_STAGE);

void fill_error_struct(vitalslib_error_details* error_details, const vitalslib_error_code code)
{
	if (error_details)
	{
		error_details->code = code;
		error_details->message = vitalslib_error_descriptions[code];
	}
}

bool is_valid_handle(const vitalslib_handle* const handle, vitalslib_error_details* error_details)
{
	if (handle)
	{
		if (reinterpret_cast<const depthsense::vitalslib::pipeline* const>(handle)->is_valid())
		{
			fill_error_struct(error_details, vitalslib_success);
			return true;
		}
		else
		{
			fill_error_struct(error_details, vitalslib_invalid_handle);
			return false;
		}
	}
	fill_error_struct(error_details, vitalslib_null_handle);
	return false;
}

} // namespace

vitalslib_API bool vitalslib_DECL vitalslib_get_version(
	vitalslib_OUT const char** version,
	vitalslib_OUT vitalslib_error_details* error_details)
{
	if (!version)
	{
		fill_error_struct(error_details, vitalslib_null_data_pointer);
		return false;
	}

	*version = vitalslib_version_str.c_str();

	fill_error_struct(error_details, vitalslib_success);
	return true;
}


vitalslib_API bool vitalslib_DECL vitalslib_init(
	vitalslib_IN const vitalslib_camera_parameters* const camera_parameters,
	vitalslib_IN const char* config_path,
	vitalslib_OUT vitalslib_handle** handle,
	vitalslib_OUT vitalslib_error_details* error_details)
{
	try
	{
		// TODO : empty mock handler type for vitalslib
		auto pipeline = new depthsense::vitalslib::pipeline();
		(*handle) = reinterpret_cast<vitalslib_handle*>(pipeline);

		fill_error_struct(error_details, vitalslib_success);
		return true;
	}
	catch (...)
	{
		(*handle) = nullptr;
		fill_error_struct(error_details, vitalslib_failed_to_initialize);
		return false;
	}
}



vitalslib_API bool vitalslib_DECL vitalslib_release(
	vitalslib_IN_OUT vitalslib_handle** handle,
	vitalslib_OUT vitalslib_error_details* error_details)
{
	if (!is_valid_handle(*handle, error_details))
	{
		if (!handle)
		{
			return false;
		}
	}

	auto pipeline = reinterpret_cast<depthsense::vitalslib::pipeline*>(*handle);
	delete pipeline;
	(*handle) = nullptr;
	fill_error_struct(error_details, vitalslib_success);
	return true;
}


vitalslib_API bool vitalslib_DECL vitalslib_compute(
	vitalslib_IN_OUT vitalslib_handle* handle,
	vitalslib_IN const vitalslib_frame* const frames,
	vitalslib_OUT vitalslib_error_details* error_details)
{
	if (!is_valid_handle(handle, error_details))
	{
		return false;
	}

	auto pipeline = reinterpret_cast<depthsense::vitalslib::pipeline*>(handle);

	try
	{
		const vitalslib_frame internal_frame = *frames;
		pipeline->process_frame(internal_frame);
	}
	catch (...)
	{
		fill_error_struct(error_details, vitalslib_other_failure);
		return false;
	}


	fill_error_struct(error_details, vitalslib_success);
	return true;
}



vitalslib_API bool vitalslib_DECL vitalslib_get_heart_rate(
	vitalslib_IN const vitalslib_handle* const handle,
	vitalslib_OUT float* heart_rate,
	vitalslib_OUT vitalslib_error_details* error_details)
{
	if (!is_valid_handle(handle, error_details))
	{
		return false;
	}

	auto pipeline = reinterpret_cast<const depthsense::vitalslib::pipeline* const>(handle);

	try
	{
		*heart_rate = pipeline->get_heart_rate(); //<- EXAMPLE

	}
	catch (...)
	{
		fill_error_struct(error_details, vitalslib_other_failure);
		return false;
	}

	fill_error_struct(error_details, vitalslib_success);
	return true;
}


vitalslib_API bool vitalslib_DECL vitalslib_get_respiratory_rate(
	vitalslib_IN const vitalslib_handle* const handle,
	vitalslib_OUT float* respiratory_rate,
	vitalslib_OUT vitalslib_error_details* error_details)
{
	if (!is_valid_handle(handle, error_details))
	{
		return false;
	}

	auto pipeline = reinterpret_cast<const depthsense::vitalslib::pipeline* const>(handle);

	try
	{

		*respiratory_rate = pipeline->get_respiratory_rate(); //<- EXAMPLE

	}
	catch (...)
	{
		fill_error_struct(error_details, vitalslib_other_failure);
		return false;
	}


	fill_error_struct(error_details, vitalslib_success);
	return true;
}

