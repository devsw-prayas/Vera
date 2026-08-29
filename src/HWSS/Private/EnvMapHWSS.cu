#include "EnvMapHWSS.h"

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>

namespace Vera::Spectral::HWSS {

	// Build the lat-long importance-sampling data + upload the spectral texel
	// texture. `func` is the sin(theta)-weighted per-texel luminance PBRT's
	// InfiniteAreaLight uses as its 2D distribution: the sin(theta) row weight
	// makes uniform-ish skies sample uniformly over the sphere and cancels
	// against the equirect Jacobian (2 PI^2 sin theta) in the pdf.
	EnvMapHWSS MakeImageEnvMap(const RGB2SpecTable& table, const HdrEnvSource& src, float intensity) {
		EnvMapHWSS env{};
		env.intensity = intensity;

		const int w = src.width, h = src.height;
		if (!src.pixels || w <= 0 || h <= 0) {
			fprintf(stderr, "MakeImageEnvMap: empty source — falling back to black constant env.\n");
			return env; // tex == 0, intensity carried but colorSpec is black
		}

		env.width  = w;
		env.height = h;

		std::vector<float4> texels((size_t)w * h);
		std::vector<float>  func((size_t)w * h);
		std::vector<float>  condCdf((size_t)h * (w + 1));
		std::vector<float>  marginalCdf(h + 1);
		std::vector<float>  rowInt(h);

		for (int y = 0; y < h; ++y) {
			// Texel centre latitude -> sin(theta) row weight.
			float theta = PI * (y + 0.5f) / h;
			float sinTheta = sinf(theta);

			float* rowCdf = &condCdf[(size_t)y * (w + 1)];
			rowCdf[0] = 0.f;

			for (int x = 0; x < w; ++x) {
				float3 c = src.pixels[(size_t)y * w + x];
				float r = fmaxf(c.x, 0.f), g = fmaxf(c.y, 0.f), b = fmaxf(c.z, 0.f);

				float m = fmaxf(r, fmaxf(g, b));               // radiance magnitude
				RGBSigmoidPolynomial poly{ 0.f, 0.f, 0.f };
				if (m > 1e-8f)
					poly = RGB2SpecLookup(table, make_float3(r / m, g / m, b / m)); // unit-max chroma
				else
					m = 0.f;
				texels[(size_t)y * w + x] = make_float4(poly.c0, poly.c1, poly.c2, m);

				float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b; // linear luminance
				float weight = Y * sinTheta;
				func[(size_t)y * w + x] = weight;
				rowCdf[x + 1] = rowCdf[x] + weight / w;            // unnormalised running integral
			}

			rowInt[y] = rowCdf[w];
			if (rowInt[y] > 0.f) {
				float inv = 1.f / rowInt[y];
				for (int x = 1; x <= w; ++x) rowCdf[x] *= inv;
			} else {
				for (int x = 1; x <= w; ++x) rowCdf[x] = (float)x / w; // uniform fallback for a black row
			}
			rowCdf[w] = 1.f;
		}

		marginalCdf[0] = 0.f;
		for (int y = 0; y < h; ++y) marginalCdf[y + 1] = marginalCdf[y] + rowInt[y] / h;
		float funcInt = marginalCdf[h];
		if (funcInt > 0.f) {
			float inv = 1.f / funcInt;
			for (int y = 1; y <= h; ++y) marginalCdf[y] *= inv;
		} else {
			for (int y = 1; y <= h; ++y) marginalCdf[y] = (float)y / h;
		}
		marginalCdf[h] = 1.f;
		env.funcInt = (funcInt > 0.f) ? funcInt : 1.f;

		// ── texel texture: wrap in u (phi), clamp in v (theta) ──
		cudaArray_t array = nullptr;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
		cudaMallocArray(&array, &desc, w, h);
		cudaMemcpy2DToArray(array, 0, 0, texels.data(), (size_t)w * sizeof(float4),
			(size_t)w * sizeof(float4), h, cudaMemcpyHostToDevice);

		cudaResourceDesc resDesc{};
		resDesc.resType         = cudaResourceTypeArray;
		resDesc.res.array.array = array;

		cudaTextureDesc texDesc{};
		texDesc.addressMode[0]   = cudaAddressModeWrap;
		texDesc.addressMode[1]   = cudaAddressModeClamp;
		texDesc.filterMode       = cudaFilterModeLinear;
		texDesc.readMode         = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;
		cudaCreateTextureObject(&env.tex, &resDesc, &texDesc, nullptr);

		// ── distribution buffers ──
		float *dFunc = nullptr, *dCond = nullptr, *dMarg = nullptr;
		cudaMalloc(&dFunc, func.size()       * sizeof(float));
		cudaMalloc(&dCond, condCdf.size()    * sizeof(float));
		cudaMalloc(&dMarg, marginalCdf.size()* sizeof(float));
		cudaMemcpy(dFunc, func.data(),        func.size()        * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dCond, condCdf.data(),     condCdf.size()     * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dMarg, marginalCdf.data(), marginalCdf.size() * sizeof(float), cudaMemcpyHostToDevice);
		env.func        = dFunc;
		env.condCdf     = dCond;
		env.marginalCdf = dMarg;

		return env;
	}

	void FreeImageEnvMap(EnvMapHWSS& env) {
		if (env.tex) {
			cudaResourceDesc rd{};
			cudaGetTextureObjectResourceDesc(&rd, env.tex);
			cudaDestroyTextureObject(env.tex);
			if (rd.resType == cudaResourceTypeArray && rd.res.array.array)
				cudaFreeArray(rd.res.array.array);
		}
		if (env.func)        cudaFree((void*)env.func);
		if (env.condCdf)     cudaFree((void*)env.condCdf);
		if (env.marginalCdf) cudaFree((void*)env.marginalCdf);
		env = EnvMapHWSS{};
	}
}
