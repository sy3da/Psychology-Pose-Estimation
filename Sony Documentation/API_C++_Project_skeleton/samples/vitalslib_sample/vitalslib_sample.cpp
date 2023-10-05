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

#include "vitalslib_sample.h"
#include "vitalslib_api.h"

#include <iostream>

namespace depthsense
{
	int vitalslib_run(const char* vitalslib_arg)
	{
		std::cout << "Argument input : " << vitalslib_arg << std::endl;

		//#### RUN SUB-ROUTINE START####

		//Get Version (0)
		vitalslib_error_details error_details;
		const char* version;
		vitalslib_get_version(&version, &error_details);
		std::cout << "vitalslib version : " << version << std::endl;

		//start ui456 module

		//Intialise (1)
		/*vitalslib_init()*/
		

		//while(){
			//Processing (2)
			/*vitalslib_compute()*/

			//Get data (3)
			/*vitalslib_get_heart_rate()*/
			/*vitalslib_get_respiratory_rate()*/
	    //}

		//Release (4)
		/*vitalslib_release()*/

		auto result = other_function(0, 0);
		std::cout << "Result output : " << result << std::endl;

		//#### RUN SUB-ROUTINE STOP####

		return result;
	}

// Example
	int other_function(const unsigned long long arg1, const unsigned long long arg2)
	{
		// Function that can be use in the SUB-ROUTINE
		
		return 0;
	}

} // namespace depthsense
