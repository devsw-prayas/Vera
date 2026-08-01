#include "RGB2Spec.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Binary layout matches mitsuba-renderer/rgb2spec's .coeff format (baked via
// its rgb2spec_opt tool), so files produced by that tool load directly:
//   char     magic[4]  = "SPEC"
//   uint32_t res
//   float    scale[res]
//   float    data[3 * res * res * res * 3]
//   ... (trailing forward-model block, unused here, not read)

namespace BSPT::Spectral::HWSS {
	static constexpr char kMagic[4] = { 'S','P','E','C' };

	RGB2SpecTable RGB2SpecTable::Load(const char* path) {
		RGB2SpecTable table{};

		FILE* f = fopen(path, "rb");
		if (!f) {
			fprintf(stderr, "RGB2SpecTable::Load: failed to open '%s'\n", path);
			return table;
		}

		char magic[4];
		if (fread(magic, 1, 4, f) != 4 || memcmp(magic, kMagic, 4) != 0) {
			fprintf(stderr, "RGB2SpecTable::Load: bad magic in '%s'\n", path);
			fclose(f);
			return table;
		}

		int32_t res = 0;
		fread(&res, sizeof(int32_t), 1, f);
		if (res <= 1) {
			fprintf(stderr, "RGB2SpecTable::Load: invalid resolution in '%s'\n", path);
			fclose(f);
			return table;
		}

		size_t dataCount = (size_t)3 * res * res * res * 3;
		table.res   = res;
		table.scale = new float[res];
		table.data  = new float[dataCount];
		fread(table.scale, sizeof(float), res, f);
		fread(table.data,  sizeof(float), dataCount, f);
		fclose(f);
		return table;
	}

	void RGB2SpecTable::Free(RGB2SpecTable& table) {
		delete[] table.scale;
		delete[] table.data;
		table = RGB2SpecTable{};
	}

	static int UpperBound(const float* arr, int n, float v) {
		int lo = 0, hi = n;
		while (lo < hi) {
			int mid = (lo + hi) >> 1;
			if (arr[mid] <= v) lo = mid + 1; else hi = mid;
		}
		return lo;
	}

	RGBSigmoidPolynomial RGB2SpecLookup(const RGB2SpecTable& table, float3 rgb) {
		const int res = table.res;

		rgb.x = std::min(std::max(rgb.x, 0.f), 1.f);
		rgb.y = std::min(std::max(rgb.y, 0.f), 1.f);
		rgb.z = std::min(std::max(rgb.z, 0.f), 1.f);

		if (rgb.x == rgb.y && rgb.y == rgb.z) {
			RGBSigmoidPolynomial poly{};
			float v = rgb.x;
			float denom = sqrtf(std::max(v * (1.f - v), 1e-6f));
			poly.c0 = 0.f;
			poly.c1 = 0.f;
			poly.c2 = (v - 0.5f) / denom;
			return poly;
		}

		float comp[3] = { rgb.x, rgb.y, rgb.z };
		int maxc = (comp[0] > comp[1]) ? (comp[0] > comp[2] ? 0 : 2) : (comp[1] > comp[2] ? 1 : 2);
		float z = comp[maxc];
		float x = comp[(maxc + 1) % 3] * (res - 1) / z;
		float y = comp[(maxc + 2) % 3] * (res - 1) / z;

		int zi = std::min(std::max(UpperBound(table.scale, res, z) - 1, 0), res - 2);
		float dz = (z - table.scale[zi]) / (table.scale[zi + 1] - table.scale[zi]);

		int xi = std::min(std::max((int)x, 0), res - 2);
		float dx = x - xi;
		int yi = std::min(std::max((int)y, 0), res - 2);
		float dy = y - yi;

		auto idx = [&](int zc, int yc, int xc, int c) -> size_t {
			return (((size_t)maxc * res + zc) * res + yc) * res * 3 + (size_t)xc * 3 + c;
		};

		RGBSigmoidPolynomial poly{};
		float* out[3] = { &poly.c0, &poly.c1, &poly.c2 };
		for (int c = 0; c < 3; ++c) {
			float c000 = table.data[idx(zi,   yi,   xi,   c)];
			float c100 = table.data[idx(zi,   yi,   xi+1, c)];
			float c010 = table.data[idx(zi,   yi+1, xi,   c)];
			float c110 = table.data[idx(zi,   yi+1, xi+1, c)];
			float c001 = table.data[idx(zi+1, yi,   xi,   c)];
			float c101 = table.data[idx(zi+1, yi,   xi+1, c)];
			float c011 = table.data[idx(zi+1, yi+1, xi,   c)];
			float c111 = table.data[idx(zi+1, yi+1, xi+1, c)];

			float c00 = c000 * (1.f - dx) + c100 * dx;
			float c10 = c010 * (1.f - dx) + c110 * dx;
			float c01 = c001 * (1.f - dx) + c101 * dx;
			float c11 = c011 * (1.f - dx) + c111 * dx;

			float c0 = c00 * (1.f - dy) + c10 * dy;
			float c1 = c01 * (1.f - dy) + c11 * dy;

			*out[c] = c0 * (1.f - dz) + c1 * dz;
		}
		return poly;
	}
}
