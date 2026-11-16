// C ABI: opaque handles, explicit ownership, no exceptions across the boundary.
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "core.hpp"
#include "pmge/pmge.hpp"

struct pmge_plan {
  pmge::Plan plan;
};

namespace {
thread_local std::string g_error;

template <class F>
int guarded(F&& f) {
  try {
    f();
    g_error.clear();
    return PMGE_OK;
  } catch (const std::invalid_argument& e) {
    g_error = e.what();
    return PMGE_ERR_ARGUMENT;
  } catch (const std::exception& e) {
    g_error = e.what();
    return PMGE_ERR_INTERNAL;
  } catch (...) {
    g_error = "unknown error";
    return PMGE_ERR_INTERNAL;
  }
}
void need(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(what);
}
}  // namespace

extern "C" {

int pmge_abi_version(void) { return PMGE_ABI_VERSION; }
const char* pmge_version(void) { return "pmge 0.1.0 (cpu reference core)"; }
const char* pmge_last_error(void) { return g_error.c_str(); }
void pmge_free(void* p) { std::free(p); }

void pmge_request_init(pmge_request* r) {
  if (!r) return;
  std::memset(r, 0, sizeof *r);
  r->seed = 1;
  r->footprint_type = PMGE_FOOTPRINT_CIRCLE;
  r->radius_m = 6000;
  r->base_m = 0;
  r->min_m = 0;
  r->max_m = 2500;
  r->peak_spacing_m = 9000;
  r->peak_type_weights[PMGE_PEAK_SHARP] = 0.6;
  r->peak_type_weights[PMGE_PEAK_ROUNDED] = 0.4;
  r->ridge_half_width_m = 3500;
  r->spur_spacing_m = 7000;
  r->spur_length_m = 6000;
  r->spur_half_width_m = 2000;
  r->foothill_width_m = 1500;
  r->foothill_profile = PMGE_FOOTHILL_SMOOTH;
  r->maturity = 0.2;
  r->valley_depth = 0.6;
  r->detail = 1.0;
  r->style = PMGE_STYLE_AUTO;
  r->complexity = 1.0;
  r->age_ma = 0.0;
  r->asymmetry = 0.0;
  r->plateau = 0.0;
}

int pmge_validate(const pmge_request* request, char* message, size_t size) {
  int rc = guarded([&] {
    need(request != nullptr, "request is NULL");
    pmge::validate(*request);
  });
  if (message && size) {
    std::strncpy(message, rc == PMGE_OK ? "" : g_error.c_str(), size - 1);
    message[size - 1] = '\0';
  }
  return rc;
}

pmge_plan* pmge_plan_create(const pmge_request* request, const double* line_xy, int32_t points) {
  pmge_plan* out = nullptr;
  guarded([&] {
    need(request != nullptr, "request is NULL");
    std::vector<pmge::Point> line;
    if (request->footprint_type == PMGE_FOOTPRINT_LINE) {
      need(line_xy != nullptr && points >= 2, "a line footprint needs line_xy with >= 2 points");
      for (int32_t i = 0; i < points; ++i) line.push_back({line_xy[2 * i], line_xy[2 * i + 1]});
    }
    out = new pmge_plan{pmge::make_plan(*request, line)};
  });
  return out;
}

pmge_plan* pmge_plan_create_external(
    const pmge_request* request, int32_t ridge_count, const int32_t* ridge_offsets,
    const double* ridge_xy, const double* ridge_crest, const double* ridge_half_width,
    const int32_t* ridge_rank, const int32_t* ridge_system, const double* ridge_sharpness,
    const double* ridge_plateau, const double* ridge_asymmetry, int32_t valley_count,
    const int32_t* valley_offsets, const double* valley_xy, const double* valley_depth,
    const double* valley_half_width, double grid_x0, double grid_y0, double grid_cell,
    int32_t grid_nx, int32_t grid_ny, const float* amplitude, const float* gain,
    double mean_offset) {
  pmge_plan* out = nullptr;
  guarded([&] {
    need(request != nullptr, "request is NULL");
    need(ridge_count >= 0 && valley_count >= 0, "negative counts");
    need(ridge_count == 0 || (ridge_offsets && ridge_xy && ridge_crest && ridge_half_width),
         "ridge arrays are NULL");
    need(valley_count == 0 || (valley_offsets && valley_xy && valley_depth && valley_half_width),
         "valley arrays are NULL");
    need(amplitude != nullptr && grid_nx >= 2 && grid_ny >= 2 && grid_cell > 0,
         "an amplitude grid of at least 2 x 2 cells is required");
    need(request->detail >= 0 && request->detail <= 2, "detail must be within [0, 2]");
    auto plan = std::make_unique<pmge_plan>();
    pmge::Plan& p = plan->plan;
    p.request = *request;
    p.external = true;
    p.mean_offset = mean_offset;
    for (int32_t i = 0; i < ridge_count; ++i) {
      int32_t a = ridge_offsets[i], b = ridge_offsets[i + 1];
      need(b - a >= 2, "every ridge needs at least two points");
      need(ridge_half_width[i] > 0, "ridge half width must be positive");
      pmge::Ridge r;
      r.id = i;
      r.rank = ridge_rank ? ridge_rank[i] : 0;
      r.system = ridge_system ? ridge_system[i] : 0;
      r.half_width = ridge_half_width[i];
      r.sharpness = ridge_sharpness ? ridge_sharpness[i] : 1.2;
      r.plateau = ridge_plateau ? ridge_plateau[i] : 0.0;
      r.asymmetry = ridge_asymmetry ? ridge_asymmetry[i] : 0.0;
      need(r.sharpness > 0 && r.plateau >= 0 && r.plateau < 1 && std::abs(r.asymmetry) < 1,
           "ridge sharpness must be > 0, plateau in [0, 1), |asymmetry| < 1");
      for (int32_t k = a; k < b; ++k) {
        r.points.push_back({ridge_xy[2 * k], ridge_xy[2 * k + 1]});
        r.crest.push_back(ridge_crest[k]);
      }
      p.ridges.push_back(std::move(r));
    }
    for (int32_t i = 0; i < valley_count; ++i) {
      int32_t a = valley_offsets[i], b = valley_offsets[i + 1];
      need(b - a >= 2, "every valley needs at least two points");
      need(valley_half_width[i] > 0, "valley half width must be positive");
      pmge::Valley v;
      v.id = i;
      v.half_width = valley_half_width[i];
      for (int32_t k = a; k < b; ++k) {
        v.points.push_back({valley_xy[2 * k], valley_xy[2 * k + 1]});
        v.depth.push_back(valley_depth[k]);
      }
      p.valleys.push_back(std::move(v));
    }
    size_t cells = size_t(grid_nx) * grid_ny;
    p.amplitude = {grid_x0, grid_y0, grid_cell, grid_nx, grid_ny,
                   std::vector<float>(amplitude, amplitude + cells)};
    if (gain)
      p.gain = {grid_x0, grid_y0, grid_cell, grid_nx, grid_ny, std::vector<float>(gain, gain + cells)};
    p.finalize();
    out = plan.release();
  });
  return out;
}

void pmge_plan_destroy(pmge_plan* plan) { delete plan; }

int pmge_plan_counts(const pmge_plan* plan, int32_t* ridges, int32_t* peaks, int32_t* passes,
                     int32_t* valleys) {
  return guarded([&] {
    need(plan != nullptr, "plan is NULL");
    if (ridges) *ridges = int32_t(plan->plan.ridges.size());
    if (peaks) *peaks = int32_t(plan->plan.peaks.size());
    if (passes) *passes = int32_t(plan->plan.passes.size());
    if (valleys) *valleys = int32_t(plan->plan.valleys.size());
  });
}

char* pmge_plan_json(const pmge_plan* plan) {
  char* out = nullptr;
  guarded([&] {
    need(plan != nullptr, "plan is NULL");
    std::string s = plan->plan.to_json();
    out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) throw std::runtime_error("out of memory");
    std::memcpy(out, s.c_str(), s.size() + 1);
  });
  return out;
}

double pmge_sample_height(const pmge_plan* plan, double x, double y) {
  double h = 0;
  guarded([&] {
    need(plan != nullptr, "plan is NULL");
    h = plan->plan.sample(x, y).height;
  });
  return h;
}

int pmge_sample_batch(const pmge_plan* plan, const double* x, const double* y, int64_t n,
                      pmge_sample* out) {
  return guarded([&] {
    need(plan && (n == 0 || (x && y && out)) && n >= 0, "NULL argument or negative count");
    pmge::parallel_for(n, [&](int64_t a, int64_t b) {
      for (int64_t i = a; i < b; ++i) {
        pmge::Sample s = plan->plan.sample(x[i], y[i]);
        out[i] = {s.height, s.relief, s.structure, s.valley, s.ridge_distance, s.system,
                  s.class_id};
      }
    });
  });
}

int pmge_sample_hexes(const pmge_plan* plan, const double* cx, const double* cy, int64_t n,
                      double radius, int32_t orientation, int32_t strategy, pmge_hex_sample* out) {
  return guarded([&] {
    need(plan && (n == 0 || (cx && cy && out)) && n >= 0, "NULL argument or negative count");
    pmge::sample_hexes(plan->plan, cx, cy, n, radius, orientation, strategy, out);
  });
}

int pmge_generate_heightmap(const pmge_plan* plan, double x0, double y0, double cell, int32_t nx,
                            int32_t ny, float* out) {
  return guarded([&] {
    need(plan && out, "NULL argument");
    pmge::generate_heightmap(plan->plan, x0, y0, cell, nx, ny, out);
  });
}

int pmge_generate_patch(const pmge_plan* plan, double x0, double y0, double cell, int32_t nx,
                        int32_t ny, const float* existing, int32_t policy, float* delta) {
  return guarded([&] {
    need(plan && delta, "NULL argument");
    pmge::generate_patch(plan->plan, x0, y0, cell, nx, ny, existing, policy, delta);
  });
}

int pmge_erode_thermal(float* h, int32_t nx, int32_t ny, double cell, double talus_deg,
                       int32_t iterations, double strength) {
  return guarded([&] {
    need(h != nullptr && iterations >= 0, "NULL heightmap or negative iterations");
    pmge::erode_thermal(h, nx, ny, cell, talus_deg, iterations, strength);
  });
}

int pmge_erode_fluvial(float* h, int32_t nx, int32_t ny, double cell, double erodibility,
                       int32_t iterations) {
  return guarded([&] {
    need(h != nullptr && iterations >= 0, "NULL heightmap or negative iterations");
    pmge::erode_fluvial(h, nx, ny, cell, erodibility, iterations);
  });
}

}  // extern "C"
