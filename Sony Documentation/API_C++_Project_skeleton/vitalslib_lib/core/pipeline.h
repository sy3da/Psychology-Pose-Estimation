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

#include "vitalslib_types.h"

namespace depthsense
{
namespace vitalslib
{
class pipeline
{

public:
	bool is_valid() const { return internal_state == state::good; }

	void process_frame(const vitalslib_frame& frames)
	{
		// TODO : this is only for mock data, this should be handled properly by the lib
		// if something is wrong , then internal state = invalid

		last_processed_timestamp = frames.timestamp;
	}

	unsigned long long get_processed_timestamp() const { return last_processed_timestamp; }

	float get_heart_rate() const
	{
		// TODO : filling mock data
		// fill mock data
		return 0.f;
	}

	float get_respiratory_rate() const
	{
		// TODO : filling mock data
		// fill mock data
		return 0.f;
	}


private:

	enum class state
	{
		good = 0,
		invalid = 1,
	};

	mutable state internal_state = state::good;

	// TODO : this is only for mock data, this should be handled properly by the lib
	unsigned long long last_processed_timestamp = 0;
};

}
}