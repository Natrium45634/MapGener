/* PMGE integration scenarios through the plain C ABI (compiles as C99).
 *   point-query        - one height, no raster anywhere
 *   hex-only           - statistics for hex cells, no heightmap
 *   local-patch        - a mountain inside an existing terrain, returned as a delta
 *   range-from-spline  - a range along a caller's polyline
 *   chunk-continuation - two neighbouring chunks of the same range join without a seam */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pmge/pmge.h"

static int fail(const char* what) {
  fprintf(stderr, "%s failed: %s\n", what, pmge_last_error());
  return 1;
}

int main(void) {
  pmge_request r;
  pmge_request_init(&r);
  r.seed = 2026;
  r.center_x = 120000;
  r.center_y = 45000;
  r.radius_m = 6000;
  r.base_m = 300;
  r.min_m = 300;
  r.max_m = 2800;
  r.peak_count = 5;
  r.ridge_half_width_m = 1200;
  r.spur_spacing_m = 1800;
  r.spur_length_m = 1800;
  r.spur_half_width_m = 700;
  r.peak_spacing_m = 2000;
  r.foothill_width_m = 1200;

  /* point-query */
  pmge_plan* plan = pmge_plan_create(&r, NULL, 0);
  if (!plan) return fail("plan");
  printf("point-query: H(120000, 45000) = %.1f m\n", pmge_sample_height(plan, 120000, 45000));

  /* hex-only: 7 flat-top hexes, circumradius 1 km */
  double hx[7] = {120000, 121500, 121500, 120000, 118500, 118500, 120000};
  double hy[7] = {45000, 45866, 44134, 43268, 44134, 45866, 46732};
  pmge_hex_sample hs[7];
  if (pmge_sample_hexes(plan, hx, hy, 7, 1000.0, PMGE_HEX_FLAT_TOP, PMGE_HEX_UNIFORM, hs))
    return fail("hexes");
  for (int i = 0; i < 7; ++i)
    printf("hex-only: hex %d centre %.0f m, min %.0f, max %.0f, slope %.3f, class %d\n", i,
           hs[i].center, hs[i].min, hs[i].max, hs[i].slope_mean, hs[i].class_id);

  /* local-patch: existing 64 x 64 terrain at 250 m cells; the delta is 0 outside */
  enum { N = 64 };
  static float terrain[N * N], delta[N * N];
  for (int k = 0; k < N * N; ++k) terrain[k] = 320.0f;
  double x0 = 120000 - 32 * 250.0, y0 = 45000 - 32 * 250.0;
  if (pmge_generate_patch(plan, x0, y0, 250.0, N, N, terrain, PMGE_BLEND_CONSTRAINED_FIT, delta))
    return fail("patch");
  int changed = 0;
  for (int k = 0; k < N * N; ++k) changed += delta[k] != 0.0f;
  printf("local-patch: %d of %d cells changed\n", changed, N * N);
  pmge_plan_destroy(plan);

  /* range-from-spline */
  r.footprint_type = PMGE_FOOTPRINT_LINE;
  r.peak_count = 0;
  r.peak_spacing_m = 9000;
  r.ridge_half_width_m = 3500;
  r.spur_spacing_m = 7000;
  r.spur_length_m = 6000;
  r.spur_half_width_m = 2000;
  double line[] = {0, 0, 60000, 15000, 120000, 5000, 200000, 30000};
  plan = pmge_plan_create(&r, line, 4);
  if (!plan) return fail("range");
  int32_t ridges, peaks, passes, valleys;
  pmge_plan_counts(plan, &ridges, &peaks, &passes, &valleys);
  printf("range-from-spline: 200 km range, %d ridges, %d peaks, %d passes, %d valleys\n", ridges,
         peaks, passes, valleys);

  /* chunk-continuation: two 100 x 100 chunks sharing an edge column */
  static float a[100 * 100], b[100 * 100];
  pmge_generate_heightmap(plan, 0, -25000, 500, 100, 100, a);      /* x 0 .. 49.5 km */
  pmge_generate_heightmap(plan, 49500, -25000, 500, 100, 100, b);  /* x 49.5 .. 99 km */
  double worst = 0;
  for (int j = 0; j < 100; ++j) worst = fmax(worst, fabs(a[j * 100 + 99] - b[j * 100]));
  printf("chunk-continuation: largest difference on the shared column %.6f m\n", worst);
  pmge_plan_destroy(plan);
  return worst == 0.0 && changed > 0 ? 0 : 1;
}
