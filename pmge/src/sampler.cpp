// HeightFieldEvaluator: a continuous H(x, y) evaluated on demand. No heightmap is needed to
// answer a point query, so windows, chunks and hex grids of any size never show a seam.
#include <limits>
#include <stdexcept>

#include "core.hpp"
#include "pmge/pmge.hpp"

namespace pmge {
namespace {

inline uint64_t cell_key(double x, double y, double cell) {
  int64_t ix = int64_t(std::floor(x / cell)), iy = int64_t(std::floor(y / cell));
  return (uint64_t(uint32_t(int32_t(ix))) << 32) | uint64_t(uint32_t(int32_t(iy)));
}

template <class F>
void for_cells(double x0, double y0, double x1, double y1, double cell, F&& f) {
  int64_t ax = int64_t(std::floor(x0 / cell)), bx = int64_t(std::floor(x1 / cell));
  int64_t ay = int64_t(std::floor(y0 / cell)), by = int64_t(std::floor(y1 / cell));
  for (int64_t ix = ax; ix <= bx; ++ix)
    for (int64_t iy = ay; iy <= by; ++iy)
      f((uint64_t(uint32_t(int32_t(ix))) << 32) | uint64_t(uint32_t(int32_t(iy))));
}

// distance from p to segment ab and the parameter of the closest point
inline double seg_distance(double px, double py, double ax, double ay, double bx, double by,
                           double& t) {
  double dx = bx - ax, dy = by - ay;
  double l2 = dx * dx + dy * dy;
  t = l2 > 1e-18 ? std::clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0) : 0.0;
  return std::hypot(px - (ax + t * dx), py - (ay + t * dy));
}

// Shape of a summit of each type as a function of u = distance / radius in [0, 1].
// `s` is the ridge structure there, `crest` the relative height of the peak.
double apply_peak(int type, double u, double s, double crest, double jag) {
  switch (type) {
    case PMGE_PEAK_SHARP:
      return s + 0.16 * crest * std::pow(1.0 - u, 2.4);
    case PMGE_PEAK_JAGGED:
      return s + crest * (0.14 * std::pow(1.0 - u, 2.2) + 0.1 * (1.0 - u) * (jag - 0.5));
    case PMGE_PEAK_ROUNDED:
      return std::max(s, crest * 0.94 * std::pow(1.0 - u * u, 1.6));
    case PMGE_PEAK_DOME:
      return std::max(s, crest * 0.9 * std::sqrt(std::max(0.0, 1.0 - u * u)));
    case PMGE_PEAK_FLAT: {  // table mountain: a level top with a steep rim
      double top = 0.8 * crest;
      double rim = top * smoothstep((1.0 - u) / 0.22);
      return u < 0.78 ? std::max(std::min(s, top), rim) : std::max(s * smoothstep((u - 0.6) / 0.4), rim);
    }
    case PMGE_PEAK_NEEDLE:
      return s + 0.3 * crest * std::pow(std::max(0.0, 1.0 - u / 0.38), 3.0);
    case PMGE_PEAK_VOLCANIC: {
      // a cone with a summit crater: the rim at u = 0.14 is the highest ring
      const double rim_u = 0.14;
      const double rim = std::pow(1.0 - rim_u, 1.15);
      if (u >= rim_u) return std::max(s * 0.6, crest * std::pow(1.0 - u, 1.15));
      double w = u / rim_u;
      return crest * (rim - 0.16 * (1.0 - w * w));
    }
    case PMGE_PEAK_TRUNCATED:
      return std::max(s * 0.7, std::min(crest * std::pow(1.0 - u, 1.1), 0.78 * crest));
    default:
      return s;
  }
}

}  // namespace

double Grid::at(double x, double y) const {
  if (values.empty()) return 1.0;
  double fx = std::clamp((x - x0) / cell, 0.0, double(nx - 1));
  double fy = std::clamp((y - y0) / cell, 0.0, double(ny - 1));
  int ix = std::min(int(fx), nx - 2 < 0 ? 0 : nx - 2), iy = std::min(int(fy), ny - 2 < 0 ? 0 : ny - 2);
  double tx = fx - ix, ty = fy - iy;
  int ix1 = std::min(ix + 1, nx - 1), iy1 = std::min(iy + 1, ny - 1);
  double a = values[size_t(iy) * nx + ix], b = values[size_t(iy) * nx + ix1];
  double c = values[size_t(iy1) * nx + ix], d = values[size_t(iy1) * nx + ix1];
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

void Plan::finalize() {
  segments_.clear();
  vsegments_.clear();
  index_.clear();
  vindex_.clear();
  pindex_.clear();
  passindex_.clear();
  range_limit_ = request.spur_length_m + request.ridge_half_width_m + request.foothill_width_m;
  cell_ = 1;
  for (const Ridge& r : ridges) cell_ = std::max(cell_, r.half_width);
  for (const Ridge& r : ridges) {
    for (size_t i = 0; i + 1 < r.points.size(); ++i) {
      Segment s{r.points[i].x, r.points[i].y, r.points[i + 1].x, r.points[i + 1].y,
                r.crest[i],    r.crest[i + 1], r.half_width,      r.sharpness,
                r.system,      r.id};
      uint32_t id = uint32_t(segments_.size());
      segments_.push_back(s);
      double m = r.half_width;
      for_cells(std::min(s.ax, s.bx) - m, std::min(s.ay, s.by) - m, std::max(s.ax, s.bx) + m,
                std::max(s.ay, s.by) + m, cell_, [&](uint64_t k) { index_[k].push_back(id); });
    }
  }
  vcell_ = 1;
  for (const Valley& v : valleys) vcell_ = std::max(vcell_, v.half_width);
  for (const Valley& v : valleys) {
    for (size_t i = 0; i + 1 < v.points.size(); ++i) {
      VSegment s{v.points[i].x, v.points[i].y, v.points[i + 1].x, v.points[i + 1].y,
                 v.depth[i],    v.depth[i + 1], v.half_width};
      uint32_t id = uint32_t(vsegments_.size());
      vsegments_.push_back(s);
      double m = v.half_width;
      for_cells(std::min(s.ax, s.bx) - m, std::min(s.ay, s.by) - m, std::max(s.ax, s.bx) + m,
                std::max(s.ay, s.by) + m, vcell_, [&](uint64_t k) { vindex_[k].push_back(id); });
    }
  }
  // the mountain body: everything within body_half_ of a main / branch axis, then foothills
  body_.clear();
  body_index_.clear();
  body_half_ = request.spur_length_m + 0.6 * request.ridge_half_width_m;
  foot_reach_ = body_half_ + 4.0 * std::max(request.foothill_width_m, 1.0);
  body_cell_ = std::max(foot_reach_, 1.0);
  if (!external) {
    for (const Ridge& r : ridges) {
      if (r.rank > 1) continue;
      for (size_t i = 0; i + 1 < r.points.size(); ++i) {
        Segment g{r.points[i].x, r.points[i].y, r.points[i + 1].x, r.points[i + 1].y,
                  r.crest[i],    r.crest[i + 1], body_half_,        1.0,
                  r.system,      r.id};
        uint32_t id = uint32_t(body_.size());
        body_.push_back(g);
        double m = foot_reach_;
        for_cells(std::min(g.ax, g.bx) - m, std::min(g.ay, g.by) - m, std::max(g.ax, g.bx) + m,
                  std::max(g.ay, g.by) + m, body_cell_,
                  [&](uint64_t k) { body_index_[k].push_back(id); });
      }
    }
  }
  pcell_ = 1;
  for (const Peak& p : peaks) pcell_ = std::max(pcell_, p.radius);
  for (size_t i = 0; i < peaks.size(); ++i) {
    const Peak& p = peaks[i];
    for_cells(p.pos.x - p.radius, p.pos.y - p.radius, p.pos.x + p.radius, p.pos.y + p.radius,
              pcell_, [&](uint64_t k) { pindex_[k].push_back(uint32_t(i)); });
  }
  for (size_t i = 0; i < passes.size(); ++i) {
    const Pass& p = passes[i];
    for_cells(p.pos.x - cell_, p.pos.y - cell_, p.pos.x + cell_, p.pos.y + cell_, cell_,
              [&](uint64_t k) { passindex_[k].push_back(uint32_t(i)); });
  }
}

double Plan::footprint_mask(double x, double y) const {
  if (external) return 1.0;
  const pmge_request& r = request;
  if (r.footprint_type == PMGE_FOOTPRINT_CIRCLE) {
    double d = std::hypot(x - r.center_x, y - r.center_y);
    return smoothstep((r.radius_m - d) / (0.14 * r.radius_m));
  }
  if (r.footprint_type == PMGE_FOOTPRINT_BOX) {
    double d = std::min(std::min(x - r.box_min_x, r.box_max_x - x),
                        std::min(y - r.box_min_y, r.box_max_y - y));
    double edge = 0.1 * std::min(r.box_max_x - r.box_min_x, r.box_max_y - r.box_min_y);
    return smoothstep(d / edge);
  }
  double best = std::numeric_limits<double>::infinity(), t;
  for (size_t i = 0; i + 1 < line.size(); ++i)
    best = std::min(best, seg_distance(x, y, line[i].x, line[i].y, line[i + 1].x, line[i + 1].y, t));
  return smoothstep((range_limit_ - best) / (0.22 * range_limit_));
}

int Plan::nearest_pass(double x, double y, double reach) const {
  auto it = passindex_.find(cell_key(x, y, cell_));
  if (it == passindex_.end()) return -1;
  int best = -1;
  double best_d = reach;
  for (uint32_t i : it->second) {
    double d = std::hypot(x - passes[i].pos.x, y - passes[i].pos.y);
    if (d < best_d) {
      best_d = d;
      best = passes[i].id;
    }
  }
  return best;
}

Sample Plan::sample(double x, double y) const {
  Sample out;
  const pmge_request& r = request;
  double s = 0, dmin = std::numeric_limits<double>::infinity();
  int system = -1;
  if (auto it = index_.find(cell_key(x, y, cell_)); it != index_.end()) {
    for (uint32_t id : it->second) {
      const Segment& g = segments_[id];
      double t;
      double d = seg_distance(x, y, g.ax, g.ay, g.bx, g.by, t);
      dmin = std::min(dmin, d);
      if (d >= g.half_width) continue;
      double crest = g.ca + (g.cb - g.ca) * t;
      double v = crest * std::pow(1.0 - d / g.half_width, g.sharpness);
      if (v > s) {
        s = v;
        system = g.system;
      }
    }
  }
  bool on_peak = false;
  if (auto it = pindex_.find(cell_key(x, y, pcell_)); it != pindex_.end()) {
    for (uint32_t id : it->second) {
      const Peak& p = peaks[id];
      double d = std::hypot(x - p.pos.x, y - p.pos.y);
      if (d >= p.radius) continue;
      double jag = p.type == PMGE_PEAK_JAGGED
                       ? ridged_detail(x, y, 0.35 * p.radius, hash_combine(r.seed, p.stable_id))
                       : 0.5;
      s = apply_peak(p.type, d / p.radius, s, p.crest, jag);
      system = p.system;
      if (d < 0.3 * p.radius) on_peak = true;
    }
  }
  double valley = 0;
  if (auto it = vindex_.find(cell_key(x, y, vcell_)); it != vindex_.end()) {
    for (uint32_t id : it->second) {
      const VSegment& g = vsegments_[id];
      double t;
      double d = seg_distance(x, y, g.ax, g.ay, g.bx, g.by, t);
      if (d >= g.half_width) continue;
      double u = d / g.half_width;
      double shape = external ? 1.0 - u
                              : (1.0 - r.maturity) * (1.0 - u) +
                                    r.maturity * (1.0 - u * u * u * u);  // V when young, U when old
      valley = std::max(valley, (g.da + (g.db - g.da) * t) * shape);
    }
  }
  // a valley lowers flanks and forelands but never cuts a crest
  double s_pos = std::max(s, 0.0);
  double pattern = s * (1.0 - 0.6 * valley * std::sqrt(std::max(0.0, 1.0 - std::min(s_pos, 1.0))));
  if (external) pattern -= 0.22 * valley * smoothstep((0.35 - s_pos) / 0.2);
  double envelope = std::clamp(2.0 * s_pos, 0.0, 1.0);
  double rough = r.detail > 0
                     ? 0.09 * envelope *
                           ridged_detail(x, y, 0.75 * cell_, hash_combine(r.seed, 0xDE7A11ull))
                     : 0.0;

  out.structure = std::clamp(s_pos, 0.0, 1.0);
  out.valley = valley;
  out.ridge_distance = std::isfinite(dmin) ? dmin : -1.0;
  out.system = s > 0 ? system : -1;

  if (external) {
    double amp = amplitude.at(x, y) * gain.at(x, y) * r.detail;
    amp = std::max(amp, 0.0);
    out.relief = amp * (pattern + rough - mean_offset);
    out.height = out.relief;
  } else {
    // Mountain body around the main / branch axes: a broad dome that the ridges stand on and
    // the valleys cut into. Beyond it, foothills fade out with the requested profile.
    double body = 0, foot = 0;
    double d_axis = std::numeric_limits<double>::infinity(), axis_crest = 0;
    if (auto it = body_index_.find(cell_key(x, y, body_cell_)); it != body_index_.end()) {
      // nearest distance is continuous; the crest is a smooth weighted mean over nearby
      // segments, so it never jumps where the nearest segment switches on a bend
      static thread_local std::vector<std::pair<double, double>> near;
      near.clear();
      for (uint32_t id : it->second) {
        const Segment& g = body_[id];
        double t;
        double d = seg_distance(x, y, g.ax, g.ay, g.bx, g.by, t);
        d_axis = std::min(d_axis, d);
        near.push_back({d, g.ca + (g.cb - g.ca) * t});
      }
      double wsum = 0, csum = 0, width = 0.35 * body_half_;
      for (auto& [d, c] : near) {
        double w = std::exp(-((d - d_axis) / width) * ((d - d_axis) / width));
        wsum += w;
        csum += w * c;
      }
      axis_crest = wsum > 0 ? csum / wsum : 0.0;
    }
    if (std::isfinite(d_axis)) {
      double t = d_axis / body_half_;
      double hump = 0.85 + 0.15 * ridged_detail(x, y, 2.2 * body_half_, r.seed + 31);
      body = (1.0 - smoothstep(t)) * (0.55 + 0.45 * axis_crest) * hump;
      if (r.foothill_width_m > 0 && r.foothill_profile != PMGE_FOOTHILL_ABRUPT) {
        double beyond = std::max(0.0, d_axis - 0.75 * body_half_);
        // a smooth fade at the start of the foothills avoids a crease where they meet the body
        foot = 0.16 * std::exp(-beyond / r.foothill_width_m) * smoothstep(1.4 - 0.5 * t);
        if (r.foothill_profile == PMGE_FOOTHILL_ROLLING)
          foot *= 0.55 + 0.9 * ridged_detail(x, y, 1.2 * r.foothill_width_m, r.seed + 17);
        if (r.foothill_profile == PMGE_FOOTHILL_TERRACED)
          foot = std::floor(foot * 30.0 + 0.5) / 30.0;
      }
    }
    // young ranges: sharp crests above a narrow body; old ones: worn crests on a broad,
    // lower body (the modelled erosion of the history removes the ridge contrast first)
    double m = r.maturity;
    double env = (body + (1.0 - body) * foot) * (1.0 - (0.8 - 0.25 * m) * valley);
    double v = std::max(0.0, (0.42 + 0.18 * m) * env + (0.62 - 0.32 * m) * pattern +
                                 (1.0 - 0.75 * m) * rough * (0.4 + env));
    if (v > 0.9) v = 0.9 + 0.1 * std::tanh((v - 0.9) / 0.1);  // point-wise ceiling, no rescale
    double mask = footprint_mask(x, y);
    out.relief = (r.max_m - r.base_m) * v * mask;
    out.height = std::clamp(r.base_m + out.relief, r.min_m, r.max_m);
    out.structure = std::max(out.structure, std::clamp(body * 0.5, 0.0, 0.3));
  }

  int cls = PMGE_CLASS_PLAIN;
  if (out.system >= 0 || out.relief > 0) cls = PMGE_CLASS_FOOTHILL;
  if (out.structure > 0.12) cls = PMGE_CLASS_SLOPE;
  if (valley > 0.45 && out.structure < 0.6) cls = PMGE_CLASS_VALLEY;
  if (out.structure > 0.62) cls = PMGE_CLASS_CREST;
  if (out.structure > 0.3 && nearest_pass(x, y, 0.3 * cell_) >= 0) cls = PMGE_CLASS_PASS;
  if (on_peak) cls = PMGE_CLASS_PEAK;
  out.class_id = cls;
  return out;
}

void generate_heightmap(const Plan& plan, double x0, double y0, double cell, int nx, int ny,
                        float* out) {
  if (nx <= 0 || ny <= 0 || !(cell > 0)) throw std::invalid_argument("bad raster geometry");
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
      out[size_t(j) * nx + i] = float(plan.sample(x0 + i * cell, y0 + j * cell).height);
}

void generate_patch(const Plan& plan, double x0, double y0, double cell, int nx, int ny,
                    const float* existing, int policy, float* delta) {
  if (nx <= 0 || ny <= 0 || !(cell > 0)) throw std::invalid_argument("bad raster geometry");
  if (policy < PMGE_BLEND_REPLACE || policy > PMGE_BLEND_CONSTRAINED_FIT)
    throw std::invalid_argument("policy is not a PMGE_BLEND_* value");
  const pmge_request& r = plan.request;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      double x = x0 + i * cell, y = y0 + j * cell;
      size_t k = size_t(j) * nx + i;
      double e = existing ? existing[k] : r.base_m;
      double mask = plan.footprint_mask(x, y);
      if (!plan.external && mask <= 0.0) {
        delta[k] = 0.0f;  // outside the footprint nothing is touched - exactly
        continue;
      }
      Sample s = plan.sample(x, y);
      double d = 0;
      switch (policy) {
        case PMGE_BLEND_REPLACE:
          d = plan.external ? s.relief : mask * (s.height - e);
          break;
        case PMGE_BLEND_ADD:
        case PMGE_BLEND_MASKED:
          d = s.relief;
          break;
        case PMGE_BLEND_MAX:
          d = plan.external ? std::max(s.relief, 0.0) : mask * std::max(s.height - e, 0.0);
          break;
        case PMGE_BLEND_CONSTRAINED_FIT:
          d = std::clamp(e + s.relief, r.min_m, r.max_m) - e;
          break;
      }
      delta[k] = float(d);
    }
  }
}

}  // namespace pmge
