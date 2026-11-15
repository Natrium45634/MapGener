// Two raster erosion models (optional post-process of a materialised heightmap).
// They are NOT local: flow crosses the raster, so results near its edge depend on the
// raster's extent. Callers must give a halo and must not expect chunk-independent output.
#include <numeric>
#include <stdexcept>

#include "core.hpp"
#include "pmge/pmge.hpp"

namespace pmge {

// Thermal (talus) erosion: material moves downhill wherever the slope exceeds the angle of
// repose. Mass is conserved exactly.
void erode_thermal(float* h, int nx, int ny, double cell, double talus_deg, int iterations,
                   double strength) {
  if (nx < 2 || ny < 2 || !(cell > 0)) throw std::invalid_argument("bad raster geometry");
  if (!(talus_deg > 0 && talus_deg < 89) || strength < 0 || strength > 1)
    throw std::invalid_argument("talus_deg must be in (0, 89), strength in [0, 1]");
  const double limit = std::tan(talus_deg * 0.017453292519943295) * cell;
  std::vector<double> delta(size_t(nx) * ny);
  static const int ox[4] = {1, -1, 0, 0}, oy[4] = {0, 0, 1, -1};
  for (int it = 0; it < iterations; ++it) {
    std::fill(delta.begin(), delta.end(), 0.0);
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        size_t k = size_t(j) * nx + i;
        double excess[4], total = 0, worst = 0;
        for (int d = 0; d < 4; ++d) {
          int a = i + ox[d], b = j + oy[d];
          excess[d] = 0;
          if (a < 0 || b < 0 || a >= nx || b >= ny) continue;
          double drop = h[k] - h[size_t(b) * nx + a] - limit;
          if (drop > 0) {
            excess[d] = drop;
            total += drop;
            worst = std::max(worst, drop);
          }
        }
        if (total <= 0) continue;
        double move = 0.5 * strength * worst;
        delta[k] -= move;
        for (int d = 0; d < 4; ++d)
          if (excess[d] > 0)
            delta[size_t(j + oy[d]) * nx + (i + ox[d])] += move * excess[d] / total;
      }
    }
    for (size_t k = 0; k < delta.size(); ++k) h[k] = float(h[k] + delta[k]);
  }
}

// Flow-based (stream power) erosion: drainage area by steepest descent in order of
// decreasing height, incision ~ erodibility * sqrt(area) * slope, never below the receiver.
void erode_fluvial(float* h, int nx, int ny, double cell, double erodibility, int iterations) {
  if (nx < 2 || ny < 2 || !(cell > 0)) throw std::invalid_argument("bad raster geometry");
  if (erodibility < 0) throw std::invalid_argument("erodibility must not be negative");
  const size_t n = size_t(nx) * ny;
  std::vector<uint32_t> order(n), recv(n);
  std::vector<double> area(n), slope(n);
  for (int it = 0; it < iterations; ++it) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        size_t k = size_t(j) * nx + i;
        double best = 0;
        uint32_t to = uint32_t(k);
        for (int b = -1; b <= 1; ++b)
          for (int a = -1; a <= 1; ++a) {
            if ((!a && !b) || i + a < 0 || j + b < 0 || i + a >= nx || j + b >= ny) continue;
            size_t m = size_t(j + b) * nx + (i + a);
            double s = (h[k] - h[m]) / (cell * std::hypot(double(a), double(b)));
            if (s > best) {
              best = s;
              to = uint32_t(m);
            }
          }
        recv[k] = to;
        slope[k] = best;
      }
    }
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return h[a] > h[b]; });
    std::fill(area.begin(), area.end(), 1.0);
    for (uint32_t k : order)
      if (recv[k] != k) area[recv[k]] += area[k];
    for (uint32_t k : order) {
      if (recv[k] == k) continue;
      double cut = erodibility * std::sqrt(area[k]) * slope[k] * cell;
      double room = 0.9 * (h[k] - h[recv[k]]);
      h[k] = float(h[k] - std::min(cut, std::max(room, 0.0)));
    }
  }
}

}  // namespace pmge
