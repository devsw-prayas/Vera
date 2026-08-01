#pragma once

namespace BSPT::Spectral::HWSS {
	// Visible-range sampling bounds shared by wavelength sampling and material evaluation.
	static constexpr float LAMBDA_MIN = 400.f;
	static constexpr float LAMBDA_MAX = 700.f;
	static constexpr float LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;
}
