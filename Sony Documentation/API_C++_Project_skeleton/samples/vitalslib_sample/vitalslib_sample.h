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

#pragma once
#ifndef DEPTHSENSE_vitalslib_SAMPLE
#define DEPTHSENSE_vitalslib_SAMPLE

#include "vitalslib_types.h"

namespace depthsense
{
	/*!
	* \brief						This is the main function to process "things" with vitalslibLib
	*
	* \param[in] vitalslib_arg		Example of argument 
	*
	*/
	int vitalslib_run(const char* vitalslib_arg);
	
	//Example
	int other_function(const unsigned long long arg1, const unsigned long long arg2);
}
#endif
