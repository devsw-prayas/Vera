#pragma once

namespace Vera::Spectral::HWSS {
	static constexpr float PI = 3.14159265358979323846f;

	// Spectral sampling bounds shared by wavelength sampling and material evaluation.
	// LAMBDA_MIN reaches into the near-UV (below the ~400nm visible edge) so
	// fluorescent excitation, typically absorbed below 400nm, is actually sampled.
	// The CIE colour-matching functions are ~0 there, so near-UV samples deposit
	// energy into fluorophores but contribute nothing directly to the image.
	static constexpr float LAMBDA_MIN = 360.f;
	static constexpr float LAMBDA_MAX = 700.f;
	static constexpr float LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;
}
