#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cmath>

#include "RGB2Spec.h"
#include "SpectralConstants.h"

namespace Vera::Spectral::HWSS {

	// Environment light. Two modes, selected by `tex`:
	//
	//  * Constant (tex == 0): a single uniform sky colour, upsampled once on the
	//    host the same way an emissive Material is. No NEE — a constant field
	//    under a cosine/GGX lobe is already low-variance and BSDF sampling picks
	//    it up on ray miss.
	//
	//  * Textured (tex != 0): an equirectangular (lat-long) HDRI. Per texel we
	//    store the Jakob-Hanika sigmoid-polynomial coefficients of the *unit-max
	//    normalised* colour in .xyz and the scalar radiance magnitude in .w
	//    (sigmoid upsampling is only defined on [0,1], so chroma and magnitude
	//    are separated). A PBRT-style 2D piecewise-constant distribution
	//    (per-row conditional CDFs + a marginal over rows, each row weighted by
	//    sin(theta)) drives importance-sampled NEE, MIS-combined with BSDF
	//    sampling on the miss path.
	//
	// Direction <-> uv convention (world space, +Y up):
	//   theta = acos(dir.y)          v = theta / PI           (v=0 at +Y pole)
	//   phi   = atan2(dir.z, dir.x)  u = phi / (2 PI) + 0.5
	struct EnvMapHWSS final {
		RGBSigmoidPolynomial colorSpec;         // constant-mode colour
		float                intensity   = 0.f; // radiance multiplier (both modes)

		cudaTextureObject_t  tex         = 0;   // float4 lat-long texels; 0 => constant mode
		int                  width       = 0;
		int                  height      = 0;
		const float*         func        = nullptr; // device: width*height, sin(theta)-weighted luminance
		const float*         condCdf     = nullptr; // device: height*(width+1), per-row normalised CDF
		const float*         marginalCdf = nullptr; // device: height+1, normalised CDF over rows
		float                funcInt     = 0.f;     // mean of `func`; pdf(u,v) = func[u,v] / funcInt
	};

	// ── uv <-> direction ─────────────────────────────────────────────────────
	__host__ __device__ inline void EnvDirToUV(float3 dir, float& u, float& v, float& sinTheta) {
		float d = 1.f / sqrtf(fmaxf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z, 1e-20f));
		float x = dir.x*d, y = dir.y*d, z = dir.z*d;
		float cosTheta = fminf(fmaxf(y, -1.f), 1.f);
		float theta = acosf(cosTheta);
		float phi   = atan2f(z, x);
		u = phi * (0.5f / PI) + 0.5f;
		v = theta * (1.f / PI);
		sinTheta = sqrtf(fmaxf(1.f - cosTheta * cosTheta, 0.f));
	}

	__host__ __device__ inline float3 EnvUVToDir(float u, float v, float& sinTheta) {
		float theta = v * PI;
		float phi   = (u - 0.5f) * (2.f * PI);
		float st = sinf(theta), ct = cosf(theta);
		sinTheta = st;
		return make_float3(st * cosf(phi), ct, st * sinf(phi));
	}

	// ── constant-mode host factory (unchanged behaviour) ─────────────────────
	inline EnvMapHWSS MakeConstEnvMap(const RGB2SpecTable& table, float3 colorRGB, float intensity) {
		EnvMapHWSS env{};
		env.colorSpec = RGB2SpecLookup(table, colorRGB);
		env.intensity = intensity;
		return env;
	}

	// Textured-mode build/free live in EnvMapHWSS.cu (CUDA allocation + the
	// Distribution2D construction).
	struct HdrEnvSource { const float3* pixels; int width; int height; };
	EnvMapHWSS MakeImageEnvMap(const RGB2SpecTable& table, const HdrEnvSource& src, float intensity);
	void       FreeImageEnvMap(EnvMapHWSS& env);

	// ── device evaluation ───────────────────────────────────────────────────
	// Per-lane radiance from direction `dir` at wavelength `lambda`.
	__device__ inline float EvalEnvMapDir(const EnvMapHWSS& env, float3 dir, float lambda) {
		if (env.tex == 0) return env.colorSpec(lambda) * env.intensity;
		float u, v, sinTheta;
		EnvDirToUV(dir, u, v, sinTheta);
		float4 t = tex2D<float4>(env.tex, u, v);
		RGBSigmoidPolynomial poly{ t.x, t.y, t.z };
		return poly(lambda) * t.w * env.intensity;
	}

	// Smallest i in [0, n-1] with cdf[i+1] > u, plus the in-bin fraction.
	// cdf has n+1 entries, cdf[0] == 0, cdf[n] == 1.
	__device__ inline int EnvSampleCdf(const float* cdf, int n, float u, float& frac) {
		int lo = 0, hi = n;
		while (lo < hi) {
			int mid = (lo + hi) >> 1;
			if (cdf[mid + 1] <= u) lo = mid + 1; else hi = mid;
		}
		int i = lo < n ? lo : n - 1;
		float lo_c = cdf[i], hi_c = cdf[i + 1];
		frac = (hi_c > lo_c) ? (u - lo_c) / (hi_c - lo_c) : 0.5f;
		return i;
	}

	// Importance-sample a direction from the lat-long luminance distribution.
	// Returns solid-angle pdf in `pdf` (0 if degenerate or constant-mode).
	__device__ inline float3 SampleEnvMap(const EnvMapHWSS& env, float2 xi, float& pdf) {
		if (env.tex == 0) { pdf = 0.f; return make_float3(0.f, 1.f, 0.f); }
		float fv, fu;
		int iv = EnvSampleCdf(env.marginalCdf, env.height, xi.y, fv);
		int iu = EnvSampleCdf(env.condCdf + (size_t)iv * (env.width + 1), env.width, xi.x, fu);

		float u = (iu + fu) / env.width;
		float v = (iv + fv) / env.height;
		float sinTheta;
		float3 dir = EnvUVToDir(u, v, sinTheta);
		if (sinTheta <= 0.f) { pdf = 0.f; return dir; }

		float pdfUV = env.func[(size_t)iv * env.width + iu] / fmaxf(env.funcInt, 1e-20f);
		pdf = pdfUV / (2.f * PI * PI * sinTheta);
		return dir;
	}

	// Solid-angle pdf the sampler above would assign to `dir` (for MIS).
	__device__ inline float EnvMapPdf(const EnvMapHWSS& env, float3 dir) {
		if (env.tex == 0) return 0.f;
		float u, v, sinTheta;
		EnvDirToUV(dir, u, v, sinTheta);
		if (sinTheta <= 0.f) return 0.f;
		int iu = min(max(int(u * env.width), 0), env.width - 1);
		int iv = min(max(int(v * env.height), 0), env.height - 1);
		float pdfUV = env.func[(size_t)iv * env.width + iu] / fmaxf(env.funcInt, 1e-20f);
		return pdfUV / (2.f * PI * PI * sinTheta);
	}
}
