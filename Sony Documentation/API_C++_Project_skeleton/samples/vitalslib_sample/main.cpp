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

#include <iostream>

#include "vitalslib_sample.h"

int main(int argc, char* argv[])
{
	// command line argument check
	if (argc != 2)
	{
		std::cout << "Usage: vitalslib_sample <argument>" << std::endl;
		return 1;
	}

	const char* vitalslib_arg = argv[1];
	return depthsense::vitalslib_run(vitalslib_arg);
}