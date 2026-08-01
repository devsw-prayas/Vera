#pragma once
#include <cstdint>
#include <vector_types.h>

namespace BSPT::Spectral::HWSS {
	static constexpr uint8_t BSDF_REFLECT  = 0x01;
	static constexpr uint8_t BSDF_TRANSMIT = 0x02;
	static constexpr uint8_t BSDF_DELTA    = 0x04;

	// Scalar (single-wavelength) BSDF result — HWSS evaluates one f/pdf per lane
	// rather than one RGB triple, since reflectance/IOR are wavelength-dependent.
	struct BSDFSample final {
		float3  wi     = {};
		float   weight = 0.f; // f * cosTheta / pdf, at the sampling (hero) wavelength
		float   pdf    = 0.f;
		uint8_t flags  = 0;
	};

	struct BSDFEval final {
		float f   = 0.f;
		float pdf = 0.f;
	};
}
