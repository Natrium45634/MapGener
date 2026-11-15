// PMGE tests through the public C ABI only (what an embedding program sees).
// No external test framework: the library must build with a bare C++20 toolchain.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "pmge/pmge.h"

#if defined(_WIN32)
static void set_threads(const char* v) { _putenv_s("PMGE_THREADS", v ? v : ""); }
#else
static void set_threads(const char* v) {
  if (v)
    setenv("PMGE_THREADS", v, 1);
  else
    unsetenv("PMGE_THREADS");
}
#endif

static int g_failed = 0, g_checks = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    ++g_checks;                                                                  \
    if (!(cond)) {                                                               \
      ++g_failed;                                                                \
      std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
    }                                                                            \
  } while (0)

static pmge_request base_request() {
  pmge_request r;
  pmge_request_init(&r);
  r.seed = 42;
  r.center_x = 30000;
  r.center_y = 30000;
  r.radius_m = 24000;
  r.base_m = 300;
  r.min_m = 300;
  r.max_m = 2800;
  return r;
}

struct PlanHolder {
  pmge_plan* p;
  explicit PlanHolder(pmge_plan* plan) : p(plan) {}
  ~PlanHolder() { pmge_plan_destroy(p); }
};

static void test_abi_and_validation() {
  CHECK(pmge_abi_version() == PMGE_ABI_VERSION);
  CHECK(std::strstr(pmge_version(), "pmge") != nullptr);
  pmge_request r = base_request();
  char msg[200];
  CHECK(pmge_validate(&r, msg, sizeof msg) == PMGE_OK);
  r.max_m = 100;  // below base
  CHECK(pmge_validate(&r, msg, sizeof msg) == PMGE_ERR_ARGUMENT);
  CHECK(std::strlen(msg) > 0);
  CHECK(pmge_plan_create(&r, nullptr, 0) == nullptr);
  CHECK(std::strlen(pmge_last_error()) > 0);
  r = base_request();
  r.maturity = 2;
  CHECK(pmge_validate(&r, msg, sizeof msg) == PMGE_ERR_ARGUMENT);
  r = base_request();
  for (double& w : r.peak_type_weights) w = 0;
  CHECK(pmge_validate(&r, msg, sizeof msg) == PMGE_ERR_ARGUMENT);
  r = base_request();
  r.footprint_type = PMGE_FOOTPRINT_LINE;
  CHECK(pmge_plan_create(&r, nullptr, 0) == nullptr);  // a line needs points
  CHECK(pmge_sample_batch(nullptr, nullptr, nullptr, 1, nullptr) == PMGE_ERR_ARGUMENT);
}

static void test_heights_and_limits() {
  pmge_request r = base_request();
  PlanHolder h(pmge_plan_create(&r, nullptr, 0));
  CHECK(h.p != nullptr);
  int n = 160;
  std::vector<float> map(size_t(n) * n);
  CHECK(pmge_generate_heightmap(h.p, 0, 0, 375.0, n, n, map.data()) == PMGE_OK);
  float lo = 1e30f, hi = -1e30f;
  for (float v : map) {
    CHECK(std::isfinite(v));
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  CHECK(lo >= r.min_m && hi <= r.max_m);  // hard limits, never violated
  CHECK(hi > r.base_m + 0.5 * (r.max_m - r.base_m));  // and actually mountainous
  // the corner of the raster lies outside the footprint: plain base level
  CHECK(std::abs(map[0] - r.base_m) < 1e-3);
}

static void test_point_queries_without_a_heightmap() {
  pmge_request r = base_request();
  PlanHolder h(pmge_plan_create(&r, nullptr, 0));
  double a = pmge_sample_height(h.p, 30000, 30000);
  std::vector<double> x{30000, 1e7}, y{30000, -1e7};
  pmge_sample s[2];
  CHECK(pmge_sample_batch(h.p, x.data(), y.data(), 2, s) == PMGE_OK);
  CHECK(s[0].height == a);  // point and batch queries agree exactly
  CHECK(s[1].height == r.base_m && s[1].system == -1);  // far away: nothing
  // repeated plan with the same seed and request is bit-identical
  PlanHolder h2(pmge_plan_create(&r, nullptr, 0));
  char* j1 = pmge_plan_json(h.p);
  char* j2 = pmge_plan_json(h2.p);
  CHECK(std::strcmp(j1, j2) == 0);
  pmge_free(j1);
  pmge_free(j2);
  CHECK(pmge_sample_height(h2.p, 31234.5, 28765.25) == pmge_sample_height(h.p, 31234.5, 28765.25));
  pmge_request other = r;
  other.seed = 43;
  PlanHolder h3(pmge_plan_create(&other, nullptr, 0));
  char* j3 = pmge_plan_json(h3.p);
  char* j1b = pmge_plan_json(h.p);
  CHECK(std::strcmp(j1b, j3) != 0);
  pmge_free(j3);
  pmge_free(j1b);
}

static void test_ridge_graph_is_consistent() {
  pmge_request r = base_request();
  r.peak_count = 6;
  PlanHolder h(pmge_plan_create(&r, nullptr, 0));
  int32_t ridges, peaks, passes, valleys;
  CHECK(pmge_plan_counts(h.p, &ridges, &peaks, &passes, &valleys) == PMGE_OK);
  CHECK(peaks == 6 && passes == 5);  // a pass between every pair of neighbouring peaks
  CHECK(ridges >= 3 && valleys >= 1);
  char* json = pmge_plan_json(h.p);
  std::string js(json);
  pmge_free(json);
  CHECK(js.find("\"between_peaks\":[0,1]") != std::string::npos);
  CHECK(js.find("\"rank\":\"spur\"") != std::string::npos);
  // a pass is a saddle: lower than the peaks it joins, higher than the flanks beside it
  double px[2], py[2];
  {
    size_t a = js.find("\"passes\":[{");
    size_t xs = js.find("\"x_m\":", a), ys = js.find("\"y_m\":", a);
    px[0] = std::atof(js.c_str() + xs + 6);
    py[0] = std::atof(js.c_str() + ys + 6);
  }
  {
    size_t a = js.find("\"peaks\":[{");
    size_t xs = js.find("\"x_m\":", a), ys = js.find("\"y_m\":", a);
    px[1] = std::atof(js.c_str() + xs + 6);
    py[1] = std::atof(js.c_str() + ys + 6);
  }
  double pass_h = pmge_sample_height(h.p, px[0], py[0]);
  double peak_h = pmge_sample_height(h.p, px[1], py[1]);
  CHECK(peak_h > pass_h + 50.0);
  // the pass is a maximum across the ridge: step sideways on both sides and go down
  double dx = px[1] - px[0], dy = py[1] - py[0], l = std::hypot(dx, dy);
  double nx = -dy / l, ny = dx / l, off = 0.6 * r.ridge_half_width_m;
  CHECK(pmge_sample_height(h.p, px[0] + nx * off, py[0] + ny * off) < pass_h);
  CHECK(pmge_sample_height(h.p, px[0] - nx * off, py[0] - ny * off) < pass_h);
}

static void test_peak_types_differ() {
  // one mountain per type, same seed: the summit shapes must be distinguishable
  const int types[] = {PMGE_PEAK_SHARP, PMGE_PEAK_ROUNDED, PMGE_PEAK_FLAT, PMGE_PEAK_VOLCANIC,
                       PMGE_PEAK_NEEDLE, PMGE_PEAK_DOME};
  std::vector<std::vector<double>> profiles;
  for (int t : types) {
    pmge_request r = base_request();
    r.radius_m = 12000;
    r.peak_count = 1;
    r.spur_length_m = 0;
    r.detail = 0;
    for (double& w : r.peak_type_weights) w = 0;
    r.peak_type_weights[t] = 1;
    PlanHolder h(pmge_plan_create(&r, nullptr, 0));
    CHECK(h.p != nullptr);
    std::vector<double> prof;
    for (int k = 0; k <= 40; ++k) prof.push_back(pmge_sample_height(h.p, 30000 + 75.0 * k, 30000));
    profiles.push_back(prof);
  }
  // flat top: the first 1.5 km hardly change; sharp: the summit falls off fast
  auto drop = [](const std::vector<double>& p, int k) { return p[0] - p[k]; };
  CHECK(drop(profiles[2], 20) < 0.25 * drop(profiles[0], 20) + 1.0);
  // volcanic: a crater, so the very centre is lower than a point just off it
  CHECK(profiles[3][0] < profiles[3][3]);
  // needle rises above the sharp peak right at the summit
  CHECK(profiles[4][0] > profiles[0][0]);
  int distinct = 0;
  for (size_t a = 0; a < profiles.size(); ++a)
    for (size_t b = a + 1; b < profiles.size(); ++b) {
      double diff = 0;
      for (size_t k = 0; k < profiles[a].size(); ++k)
        diff = std::max(diff, std::abs(profiles[a][k] - profiles[b][k]));
      distinct += diff > 40.0;
    }
  CHECK(distinct == 15);  // every pair of types differs by more than 40 m somewhere
}

static void test_patch_is_local_and_policies() {
  pmge_request r = base_request();
  r.radius_m = 8000;
  PlanHolder h(pmge_plan_create(&r, nullptr, 0));
  int n = 200;
  double cell = 300.0;
  std::vector<float> terrain(size_t(n) * n), delta(size_t(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) terrain[size_t(j) * n + i] = float(250 + 0.01 * i * cell);
  CHECK(pmge_generate_patch(h.p, 0, 0, cell, n, n, terrain.data(), PMGE_BLEND_ADD, delta.data()) ==
        PMGE_OK);
  int touched_outside = 0, inside_nonzero = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      double d = std::hypot(i * cell - 30000, j * cell - 30000);
      float v = delta[size_t(j) * n + i];
      if (d > r.radius_m && v != 0.0f) ++touched_outside;
      if (d < 0.5 * r.radius_m && v > 0.0f) ++inside_nonzero;
    }
  CHECK(touched_outside == 0);  // exactly zero outside the footprint
  CHECK(inside_nonzero > 100);
  // constrained fit never leaves the absolute limits
  CHECK(pmge_generate_patch(h.p, 0, 0, cell, n, n, terrain.data(), PMGE_BLEND_CONSTRAINED_FIT,
                            delta.data()) == PMGE_OK);
  // inside the footprint the result respects the limits; outside nothing is touched
  // (here the existing terrain there is even below min_m, and it stays as it was)
  bool within = true, untouched = true;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      size_t k = size_t(j) * n + i;
      double d = std::hypot(i * cell - 30000, j * cell - 30000);
      double v = terrain[k] + delta[k];
      if (d < 0.8 * r.radius_m) within &= v >= r.min_m - 1e-3 && v <= r.max_m + 1e-3;
      if (d > r.radius_m) untouched &= delta[k] == 0.0f;
    }
  CHECK(within && untouched);
  // max: never lowers the existing terrain
  CHECK(pmge_generate_patch(h.p, 0, 0, cell, n, n, terrain.data(), PMGE_BLEND_MAX, delta.data()) ==
        PMGE_OK);
  bool never_lower = true;
  for (float v : delta) never_lower &= v >= 0.0f;
  CHECK(never_lower);
  CHECK(pmge_generate_patch(h.p, 0, 0, cell, n, n, terrain.data(), 99, delta.data()) ==
        PMGE_ERR_ARGUMENT);
}

static void test_chunks_have_no_seams() {
  pmge_request r = base_request();
  r.footprint_type = PMGE_FOOTPRINT_LINE;
  double line[] = {0, 20000, 30000, 26000, 60000, 18000, 90000, 30000};
  PlanHolder h(pmge_plan_create(&r, line, 4));
  // the same world rectangle as one raster and as two chunks with different origins
  int n = 120;
  double cell = 500.0;
  std::vector<float> whole(size_t(2 * n) * n), left(size_t(n) * n), right(size_t(n) * n);
  CHECK(pmge_generate_heightmap(h.p, 0, 0, cell, 2 * n, n, whole.data()) == PMGE_OK);
  CHECK(pmge_generate_heightmap(h.p, 0, 0, cell, n, n, left.data()) == PMGE_OK);
  CHECK(pmge_generate_heightmap(h.p, n * cell, 0, cell, n, n, right.data()) == PMGE_OK);
  bool same = true;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      same &= whole[size_t(j) * 2 * n + i] == left[size_t(j) * n + i];
      same &= whole[size_t(j) * 2 * n + n + i] == right[size_t(j) * n + i];
    }
  CHECK(same);  // bit-identical, so no seam can exist
  // a coarse raster (different LOD) agrees with the fine one at shared points
  std::vector<float> coarse(size_t(n) * n / 2);
  CHECK(pmge_generate_heightmap(h.p, 0, 0, 2 * cell, n, n / 2, coarse.data()) == PMGE_OK);
  CHECK(coarse[size_t(10) * n + 30] == whole[size_t(20) * 2 * n + 60]);
  // the range really crosses the chunk border: high ground on both sides of it
  float lmax = 0, rmax = 0;
  for (int j = 0; j < n; ++j) {
    lmax = std::max(lmax, left[size_t(j) * n + n - 1]);
    rmax = std::max(rmax, right[size_t(j) * n]);
  }
  CHECK(lmax > r.base_m + 500 && rmax > r.base_m + 500);
}

static void test_thread_count_does_not_change_results() {
  pmge_request r = base_request();
  PlanHolder h(pmge_plan_create(&r, nullptr, 0));
  int n = 300;
  std::vector<float> one(size_t(n) * n), many(size_t(n) * n);
  set_threads("1");
  CHECK(pmge_generate_heightmap(h.p, 0, 0, 200.0, n, n, one.data()) == PMGE_OK);
  set_threads("7");
  CHECK(pmge_generate_heightmap(h.p, 0, 0, 200.0, n, n, many.data()) == PMGE_OK);
  set_threads(nullptr);
  CHECK(std::memcmp(one.data(), many.data(), one.size() * sizeof(float)) == 0);
}

static void test_hex_sampling() {
  pmge_request r = base_request();
  PlanHolder h(pmge_plan_create(&r, nullptr, 0));
  std::vector<double> cx, cy;
  for (int k = 0; k < 50; ++k) {
    cx.push_back(10000 + 800.0 * k);
    cy.push_back(30000);
  }
  std::vector<pmge_hex_sample> out(cx.size());
  for (int orientation : {PMGE_HEX_FLAT_TOP, PMGE_HEX_POINTY_TOP}) {
    CHECK(pmge_sample_hexes(h.p, cx.data(), cy.data(), int64_t(cx.size()), 1500.0, orientation,
                            PMGE_HEX_UNIFORM, out.data()) == PMGE_OK);
    for (size_t k = 0; k < out.size(); ++k) {
      const pmge_hex_sample& s = out[k];
      CHECK(s.min <= s.center + 1e-9 && s.center <= s.max + 1e-9);
      CHECK(s.min <= s.mean + 1e-9 && s.mean <= s.max + 1e-9);
      CHECK(s.variance >= 0 && s.coverage >= 0 && s.coverage <= 1);
      CHECK(s.center == pmge_sample_height(h.p, cx[k], cy[k]));  // same continuous surface
    }
  }
  bool any_relief = false, any_class = false;
  for (const auto& s : out) {
    any_relief |= s.max - s.min > 100;
    any_class |= s.class_id >= PMGE_CLASS_CREST;
  }
  CHECK(any_relief && any_class);
  // the centre-only strategy is cheaper and still returns the centre height
  CHECK(pmge_sample_hexes(h.p, cx.data(), cy.data(), 1, 1500.0, PMGE_HEX_FLAT_TOP, PMGE_HEX_CENTER,
                          out.data()) == PMGE_OK);
  CHECK(out[0].min == out[0].max);
  CHECK(pmge_sample_hexes(h.p, cx.data(), cy.data(), 1, -1.0, 0, 0, out.data()) ==
        PMGE_ERR_ARGUMENT);
}

static void test_erosion_models() {
  pmge_request r = base_request();
  int n = 128;
  double cell = 400.0;
  PlanHolder h(pmge_plan_create(&r, nullptr, 0));
  std::vector<float> map(size_t(n) * n);
  pmge_generate_heightmap(h.p, 4600, 4600, cell, n, n, map.data());
  auto stats = [&](const std::vector<float>& m, double& sum, double& steep) {
    sum = 0;
    steep = 0;
    for (int j = 0; j + 1 < n; ++j)
      for (int i = 0; i + 1 < n; ++i) {
        size_t k = size_t(j) * n + i;
        sum += m[k];
        steep = std::max(steep, double(std::abs(m[k] - m[k + 1])));
      }
  };
  double s0, steep0, s1, steep1;
  stats(map, s0, steep0);
  std::vector<float> thermal = map;
  CHECK(pmge_erode_thermal(thermal.data(), n, n, cell, 20.0, 60, 0.5) == PMGE_OK);
  stats(thermal, s1, steep1);
  double total0 = 0, total1 = 0;
  for (size_t k = 0; k < map.size(); ++k) {
    total0 += map[k];
    total1 += thermal[k];
  }
  CHECK(std::abs(total1 - total0) < 1e-4 * std::abs(total0));  // thermal conserves mass
  CHECK(steep1 < steep0);  // and lowers the steepest steps
  std::vector<float> fluvial = map;
  CHECK(pmge_erode_fluvial(fluvial.data(), n, n, cell, 0.002, 8) == PMGE_OK);
  bool never_up = true;
  double removed = 0;
  for (size_t k = 0; k < map.size(); ++k) {
    never_up &= fluvial[k] <= map[k] + 1e-3;
    removed += map[k] - fluvial[k];
  }
  CHECK(never_up && removed > 0);  // incision only removes material
  CHECK(pmge_erode_thermal(thermal.data(), n, n, cell, 95.0, 1, 0.5) == PMGE_ERR_ARGUMENT);
}

static void test_external_plan() {
  pmge_request r;
  pmge_request_init(&r);
  r.detail = 1.0;
  double xy[] = {0, 0, 10000, 0, 20000, 0};
  double crest[] = {0.5, 1.0, 0.6};
  int32_t offs[] = {0, 3};
  double hw[] = {4000};
  float amp[] = {2000, 2000, 2000, 2000};
  PlanHolder h(pmge_plan_create_external(&r, 1, offs, xy, crest, hw, nullptr, nullptr, nullptr, 0,
                                         nullptr, nullptr, nullptr, nullptr, -50000, -50000,
                                         100000, 2, 2, amp, nullptr, 0.0));
  CHECK(h.p != nullptr);
  double on = pmge_sample_height(h.p, 10000, 0), off = pmge_sample_height(h.p, 10000, 9000);
  CHECK(on > 1500 && std::abs(off) < 1e-9);  // a delta: zero away from the ridges
  CHECK(pmge_plan_create_external(&r, 1, offs, xy, crest, hw, nullptr, nullptr, nullptr, 0,
                                  nullptr, nullptr, nullptr, nullptr, 0, 0, 1, 1, 1, amp, nullptr,
                                  0.0) == nullptr);  // grid too small
}

int main() {
  struct {
    const char* name;
    void (*fn)();
  } tests[] = {{"abi_and_validation", test_abi_and_validation},
               {"heights_and_limits", test_heights_and_limits},
               {"point_queries_without_a_heightmap", test_point_queries_without_a_heightmap},
               {"ridge_graph_is_consistent", test_ridge_graph_is_consistent},
               {"peak_types_differ", test_peak_types_differ},
               {"patch_is_local_and_policies", test_patch_is_local_and_policies},
               {"chunks_have_no_seams", test_chunks_have_no_seams},
               {"hex_sampling", test_hex_sampling},
               {"thread_count_does_not_change_results", test_thread_count_does_not_change_results},
               {"erosion_models", test_erosion_models},
               {"external_plan", test_external_plan}};
  for (auto& t : tests) {
    int before = g_failed;
    t.fn();
    std::printf("%-40s %s\n", t.name, g_failed == before ? "ok" : "FAILED");
  }
  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed ? 1 : 0;
}
