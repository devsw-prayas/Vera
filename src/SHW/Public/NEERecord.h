#pragma once
#include <cstdint>

namespace Vera::SHW {
	struct NEERecord {
		uint32_t m_LightId;
		float    m_LambdaPrime;  // sampled emission/excitation wavelength
		float    m_Weight;       // f * G * V / pdf
	};
}
