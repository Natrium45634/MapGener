// Hex integration: the hex grid is an external consumer. PMGE only needs hex centres, the
// circumradius and the orientation; it never stores or assumes a grid.
#include <stdexcept>

#include "core.hpp"
#include "pmge/pmge.hpp"

namespace pmge {

void sample_hexes(const Plan& plan, const double* cx, const double* cy, int64_t n, double radius,
                  int orientation, int strategy, pmge_hex_sample* out) {
  if (!(radius > 0)) throw std::invalid_argument("hex_radius_m must be positive");
  if (orientation != PMGE_HEX_FLAT_TOP && orientation != PMGE_HEX_POINTY_TOP)
    throw std::invalid_argument("orientation is not a PMGE_HEX_* value");
  if (strategy < PMGE_HEX_CENTER || strategy > PMGE_HEX_UNIFORM)
    throw std::invalid_argument("strategy is not a PMGE_HEX_* sampling strategy");
  // sample offsets inside the hexagon (unit circumradius), centre first
  std::vector<Vec2> offs{{0, 0}};
  const double a0 = orientation == PMGE_HEX_FLAT_TOP ? 0.0 : 3.141592653589793 / 6.0;
  if (strategy >= PMGE_HEX_VERTICES)
    for (int k = 0; k < 6; ++k) {
      double a = a0 + k * 3.141592653589793 / 3.0;
      offs.push_back({0.96 * std::cos(a), 0.96 * std::sin(a)});
    }
  if (strategy == PMGE_HEX_UNIFORM) {
    for (int k = 0; k < 6; ++k) {
      double a = a0 + (k + 0.5) * 3.141592653589793 / 3.0;
      offs.push_back({0.82 * std::cos(a), 0.82 * std::sin(a)});  // edge midpoints
      offs.push_back({0.45 * std::cos(a), 0.45 * std::sin(a)});
    }
    for (int k = 0; k < 6; ++k) {
      double a = a0 + k * 3.141592653589793 / 3.0;
      offs.push_back({0.5 * std::cos(a), 0.5 * std::sin(a)});
    }
  }
  parallel_for(n, [&](int64_t i0, int64_t i1) {
  for (int64_t i = i0; i < i1; ++i) {
    pmge_hex_sample h{};
    Sample c = plan.sample(cx[i], cy[i]);
    h.center = c.height;
    h.ridge_distance = c.ridge_distance;
    h.class_id = c.class_id;
    h.nearest_pass = plan.nearest_pass(cx[i], cy[i], 1.5 * radius);
    double sum = c.height, sum2 = c.height * c.height, lo = c.height, hi = c.height;
    double covered = c.structure > 0.1 ? 1 : 0;
    // least-squares plane through the samples gives slope and aspect
    double sxx = 0, syy = 0, sxy = 0, sxh = 0, syh = 0, slope_max = 0;
    for (size_t k = 1; k < offs.size(); ++k) {
      double dx = offs[k].x * radius, dy = offs[k].y * radius;
      Sample s = plan.sample(cx[i] + dx, cy[i] + dy);
      sum += s.height;
      sum2 += s.height * s.height;
      lo = std::min(lo, s.height);
      hi = std::max(hi, s.height);
      covered += s.structure > 0.1 ? 1 : 0;
      double dh = s.height - c.height;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
      sxh += dx * dh;
      syh += dy * dh;
      slope_max = std::max(slope_max, std::abs(dh) / std::hypot(dx, dy));
    }
    double m = double(offs.size());
    h.mean = sum / m;
    h.min = lo;
    h.max = hi;
    h.variance = std::max(0.0, sum2 / m - h.mean * h.mean);
    h.coverage = covered / m;
    h.slope_max = slope_max;
    double det = sxx * syy - sxy * sxy;
    if (det > 1e-9) {
      double gx = (sxh * syy - syh * sxy) / det, gy = (syh * sxx - sxh * sxy) / det;
      h.slope_mean = std::hypot(gx, gy);
      h.aspect_deg = std::atan2(-gy, -gx) * 57.29577951308232;
    }
    out[i] = h;
  }
  });
}

}  // namespace pmge
