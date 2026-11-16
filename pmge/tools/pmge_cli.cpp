// pmge_cli: materialise a heightmap of one request through the public C ABI.
//   pmge_cli --mode circle|box|line --size 512 --extent 60000 --seed 7 --peaks 5
//            --max 3000 --types sharp=1,flat=0 --erode thermal|fluvial|none --out out
//            --style auto|alpine|himalayan|... --complexity 1 --age 0 --asymmetry 0 --plateau 0
// Writes <out>.pgm (16-bit, big-endian, 0 = min_m, 65535 = max_m) and <out>.json (the plan).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pmge/pmge.h"

int main(int argc, char** argv) {
  pmge_request r;
  pmge_request_init(&r);
  std::string mode = "circle", out = "pmge_out", erode = "none";
  int size = 512;
  double extent = 60000;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--mode") mode = v;
    else if (k == "--size") size = std::atoi(v.c_str());
    else if (k == "--extent") extent = std::atof(v.c_str());
    else if (k == "--seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
    else if (k == "--peaks") r.peak_count = std::atoi(v.c_str());
    else if (k == "--max") r.max_m = std::atof(v.c_str());
    else if (k == "--maturity") r.maturity = std::atof(v.c_str());
    else if (k == "--age") r.age_ma = std::atof(v.c_str());
    else if (k == "--complexity") r.complexity = std::atof(v.c_str());
    else if (k == "--asymmetry") r.asymmetry = std::atof(v.c_str());
    else if (k == "--plateau") r.plateau = std::atof(v.c_str());
    else if (k == "--style") {
      static const char* styles[] = {"auto", "volcanic_arc", "alpine", "himalayan", "andean",
                                     "fold_belt", "old_massif", "broad_upland", "mixed", "block"};
      r.style = -1;
      for (int s = 0; s < PMGE_STYLES; ++s)
        if (v == styles[s]) r.style = s;
      if (r.style < 0) {
        std::fprintf(stderr, "unknown style %s\n", v.c_str());
        return 2;
      }
    }
    else if (k == "--erode") erode = v;
    else if (k == "--out") out = v;
    else if (k == "--types") {
      static const char* names[] = {"sharp", "jagged", "rounded", "dome",
                                    "flat", "needle", "volcanic", "truncated"};
      for (double& w : r.peak_type_weights) w = 0;
      size_t pos = 0;
      while (pos < v.size()) {
        size_t end = v.find(',', pos);
        std::string item = v.substr(pos, end == std::string::npos ? end : end - pos);
        size_t eq = item.find('=');
        for (int t = 0; t < PMGE_PEAK_TYPES; ++t)
          if (item.substr(0, eq) == names[t])
            r.peak_type_weights[t] = eq == std::string::npos ? 1.0 : std::atof(item.c_str() + eq + 1);
        if (end == std::string::npos) break;
        pos = end + 1;
      }
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  double scale = extent / 60000.0;
  r.ridge_half_width_m *= scale;
  r.spur_spacing_m *= scale;
  r.spur_length_m *= scale;
  r.spur_half_width_m *= scale;
  r.foothill_width_m *= scale;
  r.peak_spacing_m *= scale;
  std::vector<double> line;
  if (mode == "circle") {
    r.footprint_type = PMGE_FOOTPRINT_CIRCLE;
    r.center_x = r.center_y = extent / 2;
    r.radius_m = 0.42 * extent;
  } else if (mode == "box") {
    r.footprint_type = PMGE_FOOTPRINT_BOX;
    r.box_min_x = 0.05 * extent;
    r.box_max_x = 0.95 * extent;
    r.box_min_y = 0.25 * extent;
    r.box_max_y = 0.75 * extent;
  } else if (mode == "line") {
    r.footprint_type = PMGE_FOOTPRINT_LINE;
    line = {0.1 * extent, 0.7 * extent, 0.4 * extent, 0.45 * extent,
            0.65 * extent, 0.55 * extent, 0.9 * extent, 0.25 * extent};
  } else {
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
  }
  char msg[256];
  if (pmge_validate(&r, msg, sizeof msg) != PMGE_OK) {
    std::fprintf(stderr, "invalid request: %s\n", msg);
    return 1;
  }
  pmge_plan* plan = pmge_plan_create(&r, line.empty() ? nullptr : line.data(), int32_t(line.size() / 2));
  if (!plan) {
    std::fprintf(stderr, "plan failed: %s\n", pmge_last_error());
    return 1;
  }
  std::vector<float> h(size_t(size) * size);
  double cell = extent / size;
  pmge_generate_heightmap(plan, 0.5 * cell, 0.5 * cell, cell, size, size, h.data());
  if (erode == "thermal") pmge_erode_thermal(h.data(), size, size, cell, 32.0, 40, 0.5);
  if (erode == "fluvial") pmge_erode_fluvial(h.data(), size, size, cell, 0.004, 12);
  FILE* f = std::fopen((out + ".pgm").c_str(), "wb");
  if (!f) return 1;
  std::fprintf(f, "P5\n%d %d\n65535\n", size, size);
  for (float v : h) {
    double t = (v - r.min_m) / (r.max_m - r.min_m);
    unsigned u = unsigned(std::lround(65535.0 * (t < 0 ? 0 : t > 1 ? 1 : t)));
    std::fputc(int(u >> 8), f);
    std::fputc(int(u & 255), f);
  }
  std::fclose(f);
  char* json = pmge_plan_json(plan);
  f = std::fopen((out + ".json").c_str(), "wb");
  if (f) {
    std::fputs(json, f);
    std::fclose(f);
  }
  pmge_free(json);
  int32_t nr, np, npass, nv;
  pmge_plan_counts(plan, &nr, &np, &npass, &nv);
  std::printf("%s: %dx%d, cell %.1f m, ridges %d, peaks %d, passes %d, valleys %d\n", out.c_str(),
              size, size, cell, nr, np, npass, nv);
  pmge_plan_destroy(plan);
  return 0;
}
