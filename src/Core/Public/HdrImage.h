#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

// Radiance RGBE (.hdr) loader for equirectangular env maps.
// Handles new-style per-channel RLE and flat fallback. Output is linear.
namespace Vera::Core {
	struct HdrImage final {
		int                 width = 0;
		int                 height = 0;
		std::vector<float3>  pixels;              // row-major, top row first
		bool ok() const { return width > 0 && height > 0 && (int)pixels.size() == width * height; }
	};

	namespace detail {
		inline float3 RGBEToFloat(const unsigned char rgbe[4]) {
			if (rgbe[3] == 0) return make_float3(0.f, 0.f, 0.f);
			// stb_image convention: f = 2^(e - 128 - 8), no +0.5 mantissa bias.
			float f = ldexpf(1.0f, (int)rgbe[3] - (128 + 8));
			return make_float3(rgbe[0] * f, rgbe[1] * f, rgbe[2] * f);
		}
	}

	// Returns an empty (ok()==false) image on any parse failure.
	inline HdrImage LoadHdr(const char* path) {
		HdrImage img;
		FILE* f = fopen(path, "rb");
		if (!f) { fprintf(stderr, "LoadHdr: cannot open '%s'\n", path); return img; }

		// -- ASCII header ------------------------------------------------------
		char line[512];
		bool sawMagic = false;
		while (fgets(line, sizeof(line), f)) {
			if (line[0] == '\n') break;                       // blank line ends the header
			if (strncmp(line, "#?", 2) == 0) sawMagic = true; // #?RADIANCE / #?RGBE
			// FORMAT / EXPOSURE / GAMMA lines are ignored - we only decode 32-bit_rle_rgbe.
		}
		if (!sawMagic) { fprintf(stderr, "LoadHdr: '%s' is not a Radiance file\n", path); fclose(f); return img; }

		// -- Resolution line: expect "-Y <h> +X <w>" (the standard orientation) --
		if (!fgets(line, sizeof(line), f)) { fclose(f); return img; }
		int w = 0, h = 0;
		if (sscanf(line, "-Y %d +X %d", &h, &w) != 2 || w <= 0 || h <= 0) {
			fprintf(stderr, "LoadHdr: unsupported resolution line '%s'\n", line);
			fclose(f); return img;
		}

		img.width = w;
		img.height = h;
		img.pixels.resize((size_t)w * h);

		std::vector<unsigned char> scan((size_t)w * 4);

		for (int y = 0; y < h; ++y) {
			unsigned char hdr[4];
			if (fread(hdr, 1, 4, f) != 4) { fprintf(stderr, "LoadHdr: truncated at row %d\n", y); img = HdrImage{}; fclose(f); return img; }

			bool newRLE = (hdr[0] == 2 && hdr[1] == 2 && ((hdr[2] << 8) | hdr[3]) == w && w >= 8 && w < 0x8000);
			if (!newRLE) {
				// Flat scanline: hdr holds the first pixel, read w-1 more RGBE quads.
				img.pixels[(size_t)y * w + 0] = detail::RGBEToFloat(hdr);
				for (int x = 1; x < w; ++x) {
					unsigned char px[4];
					if (fread(px, 1, 4, f) != 4) { img = HdrImage{}; fclose(f); return img; }
					img.pixels[(size_t)y * w + x] = detail::RGBEToFloat(px);
				}
				continue;
			}

			// New-RLE: four separately RLE-encoded channel planes (R,G,B,E).
			for (int c = 0; c < 4; ++c) {
				int x = 0;
				while (x < w) {
					int count = fgetc(f);
					if (count == EOF) { img = HdrImage{}; fclose(f); return img; }
					if (count > 128) {
						int run = count - 128;
						int val = fgetc(f);
						if (val == EOF || x + run > w) { img = HdrImage{}; fclose(f); return img; }
						for (int i = 0; i < run; ++i) scan[(size_t)(x++) * 4 + c] = (unsigned char)val;
					} else {
						int run = count == 0 ? 256 : count; // 0 is not legal, treat defensively
						if (x + run > w) { img = HdrImage{}; fclose(f); return img; }
						for (int i = 0; i < run; ++i) {
							int v = fgetc(f);
							if (v == EOF) { img = HdrImage{}; fclose(f); return img; }
							scan[(size_t)(x++) * 4 + c] = (unsigned char)v;
						}
					}
				}
			}
			for (int x = 0; x < w; ++x)
				img.pixels[(size_t)y * w + x] = detail::RGBEToFloat(&scan[(size_t)x * 4]);
		}

		fclose(f);
		return img;
	}
}
