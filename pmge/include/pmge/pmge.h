/* PMGE - Procedural Mountain Generation Engine. Stable C ABI.
 *
 * Units: metres for coordinates, heights and lengths; degrees for angles.
 * World axes are whatever the caller uses (PMGE never assumes a hex grid).
 * All functions are deterministic for the same request, seed and library version.
 * Thread safety: a pmge_plan is immutable after creation; any number of threads
 * may sample one plan concurrently. pmge_last_error() is thread-local.
 * Ownership: every pmge_plan* must be released with pmge_plan_destroy();
 * every char* returned by the library with pmge_free().
 */
#ifndef PMGE_H
#define PMGE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(PMGE_BUILD)
#    define PMGE_API __declspec(dllexport)
#  else
#    define PMGE_API __declspec(dllimport)
#  endif
#else
#  define PMGE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PMGE_ABI_VERSION 1

enum { PMGE_OK = 0, PMGE_ERR_ARGUMENT = 1, PMGE_ERR_CONSTRAINT = 2, PMGE_ERR_INTERNAL = 3 };
enum { PMGE_FOOTPRINT_CIRCLE = 0, PMGE_FOOTPRINT_BOX = 1, PMGE_FOOTPRINT_LINE = 2 };
enum {
  PMGE_PEAK_SHARP = 0, PMGE_PEAK_JAGGED = 1, PMGE_PEAK_ROUNDED = 2, PMGE_PEAK_DOME = 3,
  PMGE_PEAK_FLAT = 4, PMGE_PEAK_NEEDLE = 5, PMGE_PEAK_VOLCANIC = 6, PMGE_PEAK_TRUNCATED = 7,
  PMGE_PEAK_TYPES = 8
};
enum { PMGE_FOOTHILL_SMOOTH = 0, PMGE_FOOTHILL_ROLLING = 1, PMGE_FOOTHILL_ABRUPT = 2,
       PMGE_FOOTHILL_TERRACED = 3 };
enum { PMGE_BLEND_REPLACE = 0, PMGE_BLEND_ADD = 1, PMGE_BLEND_MAX = 2, PMGE_BLEND_MASKED = 3,
       PMGE_BLEND_CONSTRAINED_FIT = 4 };
enum { PMGE_HEX_FLAT_TOP = 0, PMGE_HEX_POINTY_TOP = 1 };
enum { PMGE_HEX_CENTER = 0, PMGE_HEX_VERTICES = 1, PMGE_HEX_UNIFORM = 2 };
enum { PMGE_CLASS_PLAIN = 0, PMGE_CLASS_FOOTHILL = 1, PMGE_CLASS_SLOPE = 2, PMGE_CLASS_CREST = 3,
       PMGE_CLASS_PEAK = 4, PMGE_CLASS_PASS = 5, PMGE_CLASS_VALLEY = 6 };

typedef struct pmge_plan pmge_plan;

typedef struct pmge_request {
  uint64_t seed;
  int32_t footprint_type;        /* PMGE_FOOTPRINT_* */
  double center_x, center_y;     /* circle */
  double radius_m;               /* circle */
  double box_min_x, box_min_y, box_max_x, box_max_y;
  double base_m;                 /* ground level the mountains rise from */
  double min_m, max_m;           /* hard absolute limits of the result */
  int32_t peak_count;            /* 0 = derive from peak_spacing_m */
  double peak_spacing_m;
  double peak_type_weights[PMGE_PEAK_TYPES];
  double ridge_half_width_m;
  double spur_spacing_m, spur_length_m, spur_half_width_m;
  double foothill_width_m;
  int32_t foothill_profile;      /* PMGE_FOOTHILL_* */
  double maturity;               /* 0 young and sharp .. 1 old and worn */
  double valley_depth;           /* 0 none .. 1 deep */
  double detail;                 /* multi-scale detail, 0 off .. 2 */
} pmge_request;

typedef struct pmge_sample {
  double height;          /* absolute height (standalone plan) or relief delta (external plan) */
  double relief;          /* height above base_m (standalone) or the delta itself */
  double structure;       /* 0..1: 1 on a crest at a peak, 0 away from every ridge */
  double valley;          /* 0..1 valley strength */
  double ridge_distance;  /* m to the nearest ridge axis; < 0 if none within reach */
  int32_t system;         /* mountain system id, -1 = none */
  int32_t class_id;       /* PMGE_CLASS_* */
} pmge_sample;

typedef struct pmge_hex_sample {
  double center, mean, min, max, variance;
  double slope_max, slope_mean;  /* m per m */
  double aspect_deg;             /* direction of steepest descent, atan2(dy, dx) */
  double coverage;               /* share of the hex that is mountain surface */
  double ridge_distance;         /* from the centre, m; < 0 if none */
  int32_t nearest_pass;          /* pass id, -1 = none within the hex neighbourhood */
  int32_t class_id;              /* PMGE_CLASS_* */
} pmge_hex_sample;

PMGE_API int pmge_abi_version(void);
PMGE_API const char* pmge_version(void);
PMGE_API const char* pmge_last_error(void);
PMGE_API void pmge_free(void* p);

PMGE_API void pmge_request_init(pmge_request* request);
/* Returns PMGE_OK or an error; a human-readable reason goes to message (may be NULL). */
PMGE_API int pmge_validate(const pmge_request* request, char* message, size_t message_size);

/* Standalone plan: PMGE designs the ridge graph inside the footprint.
 * line_xy = x0,y0,x1,y1,... is required for PMGE_FOOTPRINT_LINE, ignored otherwise. */
PMGE_API pmge_plan* pmge_plan_create(const pmge_request* request, const double* line_xy,
                                     int32_t line_points);

/* External plan: the caller supplies ridge axes and valleys (e.g. from a geological
 * model). Ridge i owns points ridge_offsets[i] .. ridge_offsets[i+1]-1 of xy/crest.
 * amplitude (nx*ny, row-major, metres) is the relief the structure is scaled by; gain may be
 * NULL (= 1). The sample is a delta: amplitude * gain * detail * (pattern - mean_offset). */
PMGE_API pmge_plan* pmge_plan_create_external(
    const pmge_request* request, int32_t ridge_count, const int32_t* ridge_offsets,
    const double* ridge_xy, const double* ridge_crest, const double* ridge_half_width,
    const int32_t* ridge_rank, const int32_t* ridge_system, const double* ridge_sharpness,
    int32_t valley_count, const int32_t* valley_offsets, const double* valley_xy,
    const double* valley_depth, const double* valley_half_width, double grid_x0, double grid_y0,
    double grid_cell, int32_t grid_nx, int32_t grid_ny, const float* amplitude, const float* gain,
    double mean_offset);

PMGE_API void pmge_plan_destroy(pmge_plan* plan);
PMGE_API int pmge_plan_counts(const pmge_plan* plan, int32_t* ridges, int32_t* peaks,
                              int32_t* passes, int32_t* valleys);
/* JSON description of the plan (ridges, peaks, passes, valleys, stable ids). */
PMGE_API char* pmge_plan_json(const pmge_plan* plan);

PMGE_API double pmge_sample_height(const pmge_plan* plan, double x, double y);
PMGE_API int pmge_sample_batch(const pmge_plan* plan, const double* x, const double* y, int64_t n,
                               pmge_sample* out);
PMGE_API int pmge_sample_hexes(const pmge_plan* plan, const double* center_x,
                               const double* center_y, int64_t n, double hex_radius_m,
                               int32_t orientation, int32_t strategy, pmge_hex_sample* out);
/* Row-major grid: out[j * nx + i] is the height at (x0 + i * cell, y0 + j * cell). */
PMGE_API int pmge_generate_heightmap(const pmge_plan* plan, double x0, double y0, double cell,
                                     int32_t nx, int32_t ny, float* out);
/* Writes the delta to apply to `existing`; it is exactly 0 outside the plan's footprint.
 * The caller applies it: PMGE never overwrites a terrain it was not given to own. */
PMGE_API int pmge_generate_patch(const pmge_plan* plan, double x0, double y0, double cell,
                                 int32_t nx, int32_t ny, const float* existing, int32_t policy,
                                 float* delta);

/* Raster erosion (optional post-process of a heightmap; not local: results near the edge of
 * the raster depend on the raster's extent, so give it a halo). */
PMGE_API int pmge_erode_thermal(float* height, int32_t nx, int32_t ny, double cell,
                                double talus_deg, int32_t iterations, double strength);
PMGE_API int pmge_erode_fluvial(float* height, int32_t nx, int32_t ny, double cell,
                                double erodibility, int32_t iterations);

#ifdef __cplusplus
}
#endif
#endif /* PMGE_H */
