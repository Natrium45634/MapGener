// Standalone mountain plan: PMGE designs the ridge graph inside a footprint.
// Structure first: main ridge -> peaks and passes -> spurs -> valleys. Noise only bends lines.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "core.hpp"
#include "pmge/pmge.hpp"

namespace pmge {
namespace {

std::vector<Vec2> to_vec(const std::vector<Point>& p) {
  std::vector<Vec2> v(p.size());
  for (size_t i = 0; i < p.size(); ++i) v[i] = {p[i].x, p[i].y};
  return v;
}
std::vector<Point> to_pts(const std::vector<Vec2>& p) {
  std::vector<Point> v(p.size());
  for (size_t i = 0; i < p.size(); ++i) v[i] = {p[i].x, p[i].y};
  return v;
}

int pick_type(const pmge_request& r, Rng& rng) {
  double total = 0;
  for (double w : r.peak_type_weights) total += std::max(0.0, w);
  if (total <= 0) return PMGE_PEAK_SHARP;
  double u = rng.uniform() * total;
  for (int t = 0; t < PMGE_PEAK_TYPES; ++t) {
    u -= std::max(0.0, r.peak_type_weights[t]);
    if (u <= 0) return t;
  }
  return PMGE_PEAK_SHARP;
}

double cosine_interp(const std::vector<double>& ns, const std::vector<double>& nv, double s) {
  size_t i = 0;
  while (i + 2 < ns.size() && ns[i + 1] <= s) ++i;
  double t = std::clamp((s - ns[i]) / std::max(ns[i + 1] - ns[i], 1e-9), 0.0, 1.0);
  double w = 0.5 - 0.5 * std::cos(3.141592653589793 * t);
  return nv[i] * (1 - w) + nv[i + 1] * w;
}

}  // namespace

void validate(const pmge_request& r) {
  auto fail = [](const std::string& m) { throw std::invalid_argument(m); };
  if (r.footprint_type < PMGE_FOOTPRINT_CIRCLE || r.footprint_type > PMGE_FOOTPRINT_LINE)
    fail("footprint_type is not a PMGE_FOOTPRINT_* value");
  if (r.footprint_type == PMGE_FOOTPRINT_CIRCLE && !(r.radius_m > 0))
    fail("radius_m must be positive for a circle footprint");
  if (r.footprint_type == PMGE_FOOTPRINT_BOX &&
      !(r.box_max_x > r.box_min_x && r.box_max_y > r.box_min_y))
    fail("box footprint needs box_max > box_min on both axes");
  if (!(r.max_m > r.base_m)) fail("max_m must be above base_m");
  if (!(r.min_m <= r.base_m)) fail("min_m must not be above base_m");
  if (r.peak_count < 0 || r.peak_count > 100000) fail("peak_count must be within [0, 100000]");
  if (!(r.peak_spacing_m > 0)) fail("peak_spacing_m must be positive");
  if (!(r.ridge_half_width_m > 0)) fail("ridge_half_width_m must be positive");
  if (!(r.spur_half_width_m > 0)) fail("spur_half_width_m must be positive");
  if (!(r.spur_spacing_m > 0)) fail("spur_spacing_m must be positive");
  if (r.spur_length_m < 0 || r.foothill_width_m < 0) fail("lengths must not be negative");
  if (r.maturity < 0 || r.maturity > 1) fail("maturity must be within [0, 1]");
  if (r.valley_depth < 0 || r.valley_depth > 1) fail("valley_depth must be within [0, 1]");
  if (r.detail < 0 || r.detail > 2) fail("detail must be within [0, 2]");
  if (r.style < PMGE_STYLE_AUTO || r.style >= PMGE_STYLES) fail("style is not a PMGE_STYLE_* value");
  if (!(r.complexity >= 0 && r.complexity <= 2)) fail("complexity must be within [0, 2]");
  if (!(r.age_ma >= 0 && r.age_ma <= 5000)) fail("age_ma must be within [0, 5000]");
  if (!(std::abs(r.asymmetry) <= 0.9)) fail("asymmetry must be within [-0.9, 0.9]");
  if (!(r.plateau >= 0 && r.plateau <= 0.9)) fail("plateau must be within [0, 0.9]");
  if (r.foothill_profile < PMGE_FOOTHILL_SMOOTH || r.foothill_profile > PMGE_FOOTHILL_TERRACED)
    fail("foothill_profile is not a PMGE_FOOTHILL_* value");
  double wsum = 0;
  for (double w : r.peak_type_weights) {
    if (w < 0 || !std::isfinite(w)) fail("peak_type_weights must be finite and non-negative");
    wsum += w;
  }
  if (!(wsum > 0)) fail("at least one peak type weight must be positive");
}

namespace {

// One range of a system at relative offset f across the footprint (-1..1).
struct Spec {
  double f, crest, hw, sharp;
  int role;
  double plateau = 0, asym = 0, trim_lo = 0, trim_hi = 0;
  bool abs_hw = false, peaks = true, spurs = true;
};

int choose_style(double width, double hw, double m, Rng& rng) {
  double u = rng.uniform();
  double rel = width / std::max(hw, 1.0);  // how many crest widths fit across the footprint
  if (m > 0.66) return u < 0.55 ? PMGE_STYLE_OLD_MASSIF : (u < 0.8 ? PMGE_STYLE_BROAD_UPLAND
                                                                   : PMGE_STYLE_FOLD_BELT);
  if (rel < 3.5) return u < 0.6 ? PMGE_STYLE_ALPINE : PMGE_STYLE_BLOCK;
  if (m > 0.35)
    return u < 0.35 ? PMGE_STYLE_MIXED : (u < 0.6 ? PMGE_STYLE_FOLD_BELT
                                                  : (u < 0.8 ? PMGE_STYLE_BROAD_UPLAND
                                                             : PMGE_STYLE_ALPINE));
  if (rel > 9.0 && u < 0.3) return PMGE_STYLE_HIMALAYAN;
  if (u < 0.35) return PMGE_STYLE_ALPINE;
  if (u < 0.55) return PMGE_STYLE_MIXED;
  if (u < 0.72) return PMGE_STYLE_FOLD_BELT;
  if (u < 0.86) return PMGE_STYLE_ANDEAN;
  return PMGE_STYLE_BLOCK;
}

std::vector<Spec> style_specs(int style, Rng& rng, double width, double cx) {
  std::vector<double> u(24);
  for (double& v : u) v = rng.uniform();
  bool extra = cx >= 0.5;
  std::vector<Spec> out;
  auto add = [&](Spec s) { out.push_back(s); };
  switch (style) {
    case PMGE_STYLE_VOLCANIC_ARC:
      add({0.0, 0.8, 0.7, 1.2, ROLE_MAIN});
      break;
    case PMGE_STYLE_ALPINE: {
      Spec m{(u[0] - 0.5) * 0.3, 1.0, 1.0, 1.4, ROLE_MAIN};
      m.asym = -0.25;
      add(m);
      if (extra) {
        int n = 1 + (u[3] < 0.45 * cx ? 1 : 0);
        for (int k = 0; k < n; ++k) {
          double side = ((k + int(u[4] * 2)) % 2) ? 1.0 : -1.0;
          Spec s{side * (0.45 + 0.25 * u[5 + k]), 0.55 + 0.25 * u[7 + k], 0.8, 1.25, ROLE_SECONDARY};
          s.trim_lo = 0.35 * u[9 + k];
          s.trim_hi = 0.35 * u[11 + k];
          add(s);
        }
        if (u[13] < 0.65) {
          Spec f{0.86, 0.3, 0.7, 0.9, ROLE_FRONTAL};
          f.peaks = false;
          add(f);
        }
      }
      break;
    }
    case PMGE_STYLE_HIMALAYAN: {
      Spec m{0.45, 1.0, 0.95, 1.5, ROLE_MAIN};
      m.asym = -0.35;
      add(m);
      if (extra) {
        Spec f{0.9, 0.3, 0.6, 1.0, ROLE_FRONTAL};
        f.peaks = f.spurs = false;
        add(f);
        add({0.69, 0.55, 0.7, 1.1, ROLE_SECONDARY});
        Spec pl{-0.12, 0.8, 0.4 * width, 0.8, ROLE_PLATEAU};
        pl.plateau = 0.68;
        pl.abs_hw = true;
        pl.peaks = pl.spurs = false;
        add(pl);
        Spec b{-0.8, 0.72, 0.8, 1.2, ROLE_BACK};
        b.trim_lo = 0.15 * u[0];
        b.trim_hi = 0.15 * u[1];
        add(b);
      }
      break;
    }
    case PMGE_STYLE_ANDEAN: {
      Spec m{0.35, 1.0, 0.85, 1.4, ROLE_MAIN};
      m.asym = -0.3;
      add(m);
      if (extra) {
        Spec pl{-0.1, 0.8, 0.3 * width, 0.85, ROLE_PLATEAU};
        pl.plateau = 0.6;
        pl.abs_hw = true;
        pl.peaks = pl.spurs = false;
        pl.trim_lo = 0.1 + 0.2 * u[0];
        pl.trim_hi = 0.1 + 0.2 * u[1];
        add(pl);
        Spec e{-0.5, 0.88, 0.8, 1.3, ROLE_SECONDARY};
        e.trim_hi = 0.25 * u[3];
        add(e);
        Spec f{-0.85, 0.35, 0.6, 0.9, ROLE_FRONTAL};
        f.peaks = false;
        add(f);
      }
      break;
    }
    case PMGE_STYLE_FOLD_BELT: {
      int n = extra ? 3 + int(std::lround((2 + 2 * u[0]) * std::max(cx, 0.25))) : 1;
      int centre = int(u[1] * n);
      for (int k = 0; k < n; ++k) {
        double f = n > 1 ? -0.75 + 1.5 * (k + 0.5) / n + (u[2 + k % 10] - 0.5) * 0.2 / n : 0.0;
        Spec s{f, k == centre ? 1.0 : 0.45 + 0.35 * u[12 + k % 10], 0.45, 1.15, ROLE_FOLD};
        s.trim_lo = 0.25 * u[(k + 3) % 24];
        s.trim_hi = 0.25 * u[(k + 7) % 24];
        add(s);
      }
      break;
    }
    case PMGE_STYLE_OLD_MASSIF: {
      add({0.0, 0.8, 1.9, 0.7, ROLE_MAIN});
      if (extra)
        for (int k = 0; k < 1 + (u[0] < 0.5 ? 1 : 0); ++k) {
          Spec s{(k ? -1.0 : 1.0) * (0.5 + 0.2 * u[1 + k]), 0.5, 1.3, 0.7, ROLE_SECONDARY};
          s.trim_lo = 0.3 * u[3 + k];
          s.trim_hi = 0.3 * u[5 + k];
          add(s);
        }
      break;
    }
    case PMGE_STYLE_BROAD_UPLAND: {
      Spec pl{0.0, 0.72, 0.42 * width, 0.6, ROLE_PLATEAU};
      pl.plateau = 0.55;
      pl.abs_hw = true;
      pl.peaks = pl.spurs = false;
      add(pl);
      Spec m{(u[0] - 0.5) * 0.6, 0.85, 0.9, 0.9, ROLE_MAIN};
      m.trim_lo = 0.2 * u[1];
      m.trim_hi = 0.2 * u[2];
      add(m);
      break;
    }
    case PMGE_STYLE_MIXED: {
      Spec base{0.0, 0.58, 2.2, 0.6, ROLE_BASE};
      base.peaks = base.spurs = false;
      add(base);
      Spec core{(u[0] - 0.5) * 0.25, 1.0, 0.7, 1.55, ROLE_CORE};
      core.trim_lo = 0.08 + 0.25 * u[1];
      core.trim_hi = 0.08 + 0.25 * u[2];
      add(core);
      if (extra && u[3] < 0.7) {
        Spec s{(u[4] < 0.5 ? 1.0 : -1.0) * 0.55, 0.65, 0.8, 1.2, ROLE_SECONDARY};
        s.trim_lo = 0.3 * u[5];
        s.trim_hi = 0.3 * u[6];
        add(s);
      }
      break;
    }
    default: {  // block: tilted fault blocks en echelon, steep on alternating sides
      int n = 1 + int(u[0] * (extra ? 3 : 1));
      for (int k = 0; k < n; ++k) {
        Spec s{(u[7 + k] - 0.5) * 0.6, 0.75 + 0.25 * u[10 + k], 0.9, 1.25, ROLE_BLOCK};
        s.asym = (k % 2) ? 0.6 : -0.6;
        s.trim_lo = std::max(double(k) / n + 0.06 * u[1 + k] - 0.05, 0.0);
        s.trim_hi = std::max(1.0 - double(k + 1) / n + 0.06 * u[4 + k] - 0.05, 0.0);
        add(s);
      }
    }
  }
  return out;
}

// Normals (left of the direction of travel) of a polyline, smoothed over a few points.
std::vector<Vec2> normals_of(const std::vector<Vec2>& p) {
  size_t n = p.size();
  std::vector<Vec2> out(n, Vec2{0, 1});
  for (size_t i = 0; i < n; ++i) {
    size_t a = i >= 3 ? i - 3 : 0, b = std::min(n - 1, i + 3);
    out[i] = perp(normalized(p[b] - p[a]));
  }
  return out;
}

std::vector<Vec2> smooth_line(const std::vector<Vec2>& p, int radius) {
  if (radius <= 0 || p.size() < 3) return p;
  std::vector<Vec2> out(p.size());
  for (size_t i = 0; i < p.size(); ++i) {
    Vec2 acc{0, 0};
    double w = 0;
    for (int k = -radius; k <= radius; ++k) {
      int j = std::clamp(int(i) + k, 0, int(p.size()) - 1);
      double g = std::exp(-0.5 * (double(k) / (0.5 * radius)) * (double(k) / (0.5 * radius)));
      acc = acc + p[j] * g;
      w += g;
    }
    out[i] = acc * (1.0 / w);
  }
  return out;
}

double gamma_gap(Rng& rng, double mean, double spread) {
  // a sum of two exponentials (shape 2): irregular, but never zero or huge
  double a = -std::log(std::max(1e-9, rng.uniform())), b = -std::log(std::max(1e-9, rng.uniform()));
  double g = mean * (1.0 - spread + spread * 0.5 * (a + b));
  return std::clamp(g, 0.3 * mean, 2.6 * mean);
}

}  // namespace

Plan make_plan(const pmge_request& req, const std::vector<Point>& line_in) {
  validate(req);
  if (req.footprint_type == PMGE_FOOTPRINT_LINE && line_in.size() < 2)
    throw std::invalid_argument("a line footprint needs at least two points");
  Plan plan;
  plan.request = req;
  plan.line = line_in;
  plan.finalize();  // footprint mask must work while spurs are being clipped
  Rng rng(hash_combine(req.seed, 0xA11CE5EEDull));
  const double hw = req.ridge_half_width_m;
  const double step = std::max(0.2 * hw, 100.0);
  // time: erosion widens and rounds the ranges and lowers them (isostasy keeps a remnant)
  const double m = 1.0 - (1.0 - req.maturity) * std::exp(-req.age_ma / 80.0);
  plan.maturity_eff = m;
  plan.height_scale = 0.35 + 0.65 * std::exp(-req.age_ma / 120.0);
  const double sharp = 1.55 - 0.85 * m;
  const double cx = std::clamp(req.complexity, 0.0, 2.0);
  // with time ranges lose their needles and horns: summit types shift towards rounded forms
  // (only the age does this; the requested weights are kept for a young range)
  const double m_age = 1.0 - std::exp(-req.age_ma / 80.0);
  pmge_request eroded = req;
  for (int t : {PMGE_PEAK_SHARP, PMGE_PEAK_JAGGED, PMGE_PEAK_NEEDLE})
    eroded.peak_type_weights[t] *= (1.0 - 0.85 * m_age);
  eroded.peak_type_weights[PMGE_PEAK_ROUNDED] += 1.5 * m_age;
  eroded.peak_type_weights[PMGE_PEAK_DOME] += 0.8 * m_age * m_age;

  auto add_peak = [&](int ridge, Vec2 pos, double crest, double radius) {
    Peak p;
    p.id = int(plan.peaks.size());
    p.ridge = ridge;
    p.system = 0;
    p.pos = {pos.x, pos.y};
    p.crest = crest;
    p.type = pick_type(eroded, rng);
    p.radius = radius;
    p.stable_id = hash_combine(req.seed, hash_combine(uint64_t(ridge) + 1, uint64_t(p.id) + 1));
    plan.peaks.push_back(p);
    return p.id;
  };

  // -- the axis and the width of the footprint --------------------------------------
  std::vector<Vec2> axis;
  double width = 0;  // across the footprint, m
  bool massif = false;
  if (req.footprint_type == PMGE_FOOTPRINT_LINE) {
    axis = to_vec(line_in);
    width = 2.0 * (req.spur_length_m + hw);
  } else if (req.footprint_type == PMGE_FOOTPRINT_BOX) {
    Vec2 lo{req.box_min_x, req.box_min_y}, hi{req.box_max_x, req.box_max_y};
    Vec2 size = hi - lo, mid = (lo + hi) * 0.5;
    bool wide = size.x >= size.y;
    Vec2 along = wide ? Vec2{1, 0} : Vec2{0, 1};
    double len = (wide ? size.x : size.y) * 0.5 - 1.6 * hw;
    double side = (wide ? size.y : size.x) * 0.5;
    width = 2.0 * side * 0.85;
    if (len < hw) len = hw;
    for (int k = 0; k <= 4; ++k) {
      double t = -1.0 + 0.5 * k;
      double off = (k == 0 || k == 4) ? 0.0 : rng.range(-0.22, 0.22) * side;
      axis.push_back(mid + along * (t * len) + perp(along) * off);
    }
  } else {
    Vec2 c{req.center_x, req.center_y};
    double R = req.radius_m;
    width = 1.2 * R;
    int wanted = req.peak_count > 0 ? req.peak_count
                                    : std::max(1, int(std::lround(1.3 * R / req.peak_spacing_m)));
    int lone_type = pick_type(eroded, rng);
    bool edifice = lone_type == PMGE_PEAK_VOLCANIC || lone_type == PMGE_PEAK_DOME ||
                   lone_type == PMGE_PEAK_FLAT || lone_type == PMGE_PEAK_TRUNCATED ||
                   lone_type == PMGE_PEAK_ROUNDED;
    if (wanted <= 1 && edifice) {
      // a lone volcano, dome or table mountain is one edifice, not a star of ridges
      massif = true;
      int id = add_peak(-1, c, 1.0, 0.82 * R);
      plan.peaks[id].type = lone_type;
    } else if (wanted <= 1) {
      massif = true;  // one horn: aretes radiate from the summit, unevenly
      int arms = 3 + rng.below(4);
      double a0 = rng.range(0, 6.283185307179586);
      for (int k = 0; k < arms; ++k) {
        double a = a0 + 6.283185307179586 * (k + rng.range(-0.3, 0.3)) / arms;
        double len = 0.8 * R * rng.range(0.5, 1.0);
        Vec2 d{std::cos(a), std::sin(a)};
        Vec2 bend = perp(d) * (rng.range(-0.3, 0.3) * len);
        std::vector<Vec2> raw{c, c + d * (0.55 * len) + bend, c + d * len};
        auto pts = resample(chaikin(raw, 2), step);
        Ridge r;
        r.id = int(plan.ridges.size());
        r.rank = k == 0 ? 0 : 1;
        r.parent = k == 0 ? -1 : 0;
        r.role = k == 0 ? ROLE_MAIN : ROLE_BRANCH;
        r.half_width = hw * 0.8 * (1.0 + 0.9 * m);
        r.sharpness = std::max(0.45, sharp);
        r.points = to_pts(pts);
        auto s = arclength(pts);
        for (size_t i = 0; i < pts.size(); ++i) {
          double f = s[i] / std::max(s.back(), 1e-9);
          r.crest.push_back(0.12 + 0.88 * std::pow(1.0 - f, 1.25));
        }
        plan.ridges.push_back(std::move(r));
      }
      int id = add_peak(0, c, 1.0, 0.9 * hw);
      plan.peaks[id].type = lone_type;
    } else {
      double a = rng.range(0, 3.141592653589793);
      Vec2 d{std::cos(a), std::sin(a)};
      for (int k = 0; k <= 3; ++k) {
        double t = -0.72 + 0.48 * k;
        double off = (k == 0 || k == 3) ? 0.0 : rng.range(-0.22, 0.22) * R;
        axis.push_back(c + d * (t * R) + perp(d) * off);
      }
    }
  }
  int style = req.style == PMGE_STYLE_AUTO ? choose_style(width, hw, m, rng) : req.style;
  plan.style = massif ? PMGE_STYLE_ALPINE : style;

  if (!massif) {
    auto base_axis = resample(chaikin(axis, 3), step);
    auto s_axis = arclength(base_axis);
    double total = s_axis.back();
    auto specs = style_specs(style, rng, width, cx);
    {  // a plateau or a broad base is several overlapping lobes, not one band
      std::vector<Spec> lobed;
      for (const Spec& sp : specs) {
        if (sp.role != ROLE_PLATEAU && sp.role != ROLE_BASE) {
          lobed.push_back(sp);
          continue;
        }
        double span = 1.0 - sp.trim_lo - sp.trim_hi;
        int n = 2 + rng.below(3);
        for (int k = 0; k < n; ++k) {
          Spec l = sp;
          l.f = sp.f + rng.range(-0.15, 0.15);
          l.crest = sp.crest * rng.range(0.9, 1.1);
          l.hw = sp.hw * rng.range(0.7, 1.3);
          l.asym = rng.range(-0.3, 0.3);
          l.trim_lo = std::max(0.0, sp.trim_lo + span * k / n - (k > 0 ? 0.12 * span / n : 0.0));
          l.trim_hi = std::max(0.0, 1.0 - (sp.trim_lo + span * (k + 1) / n +
                                           (k < n - 1 ? 0.12 * span / n : 0.0)));
          lobed.push_back(l);
        }
      }
      specs.swap(lobed);
    }
    if (req.plateau > 0)  // the caller asked for a level top on the main range
      for (Spec& sp : specs)
        if (sp.role == ROLE_MAIN) sp.plateau = std::max(sp.plateau, req.plateau);
    if (req.asymmetry != 0)
      for (Spec& sp : specs)
        if (sp.role == ROLE_MAIN || sp.role == ROLE_CORE) sp.asym = req.asymmetry;
    // an explicit peak count is the number of summits of the primary range, all of them
    int primary = -1;
    for (size_t k = 0; k < specs.size(); ++k)
      if (specs[k].peaks) {
        primary = int(k);
        break;
      }
    if (req.peak_count > 0)
      for (size_t k = 0; k < specs.size(); ++k)
        if (int(k) != primary) specs[k].peaks = false;
    std::vector<int> range_ids, range_spec;
    std::vector<std::pair<double, Spec>> placed;  // offset in m, spec
    // ranges: offset lines of a smoothed axis, clipped to the footprint, en echelon when long
    for (size_t spec_index = 0; spec_index < specs.size(); ++spec_index) {
      const Spec& sp = specs[spec_index];
      double lateral = sp.f * 0.5 * width * 0.82;
      int radius = int(std::min(24.0, std::max(2.0, 0.6 * std::abs(lateral) / step)));
      auto smooth = smooth_line(base_axis, radius);
      auto nrm = normals_of(smooth);
      Rng wr(hash_combine(req.seed, uint64_t(range_ids.size()) * 977u + 13u));
      double ph1 = wr.range(0, 6.283), ph2 = wr.range(0, 6.283), k1 = wr.range(0.6, 1.8);
      bool level = sp.role == ROLE_PLATEAU || sp.role == ROLE_BASE;
      std::vector<Vec2> pts(smooth.size());
      std::vector<char> keep(smooth.size(), 0);
      auto ss = arclength(smooth);
      for (size_t i = 0; i < smooth.size(); ++i) {
        double f = ss[i] / std::max(ss.back(), 1e-9);
        double wob = level ? 0.0
                           : 0.045 * width * (0.7 * std::sin(6.283 * k1 * f + ph1) +
                                              0.3 * std::sin(6.283 * 3.1 * f + ph2));
        pts[i] = smooth[i] + nrm[i] * (lateral + wob);
        bool in_trim = f >= sp.trim_lo && f <= 1.0 - sp.trim_hi;
        keep[i] = in_trim && (level || plan.footprint_mask(pts[i].x, pts[i].y) > 0.35);
      }
      // longest kept run
      size_t best_a = 0, best_b = 0;
      for (size_t i = 0; i < keep.size();) {
        if (!keep[i]) { ++i; continue; }
        size_t j = i;
        while (j < keep.size() && keep[j]) ++j;
        if (j - i > best_b - best_a) best_a = i, best_b = j;
        i = j;
      }
      if (best_b - best_a < 3) continue;
      std::vector<Vec2> run(pts.begin() + long(best_a), pts.begin() + long(best_b));
      if (arclength(run).back() < 4.0 * hw && sp.role != ROLE_MAIN) continue;
      // en echelon: long ranges break into overlapping, offset segments
      std::vector<std::vector<Vec2>> pieces{run};
      double limit = rng.range(14.0, 28.0) * hw;
      double run_len = arclength(run).back();
      bool counted = req.peak_count > 0 && int(spec_index) == primary;
      if (!level && !counted && cx >= 0.5 && run_len > 1.3 * limit) {
        pieces.clear();
        int n = int(std::ceil(run_len / limit));
        auto rs = arclength(run);
        auto rn = normals_of(run);
        for (int k = 0; k < n; ++k) {
          double a = k * run_len / n - (k > 0 ? rng.range(0.04, 0.1) * run_len / n : 0.0);
          double b = (k + 1) * run_len / n + (k < n - 1 ? rng.range(0.04, 0.1) * run_len / n : 0.0);
          double shift = rng.range(-1.0, 1.0) * 0.9 * hw;
          std::vector<Vec2> piece;
          for (size_t i = 0; i < run.size(); ++i)
            if (rs[i] >= a && rs[i] <= b) piece.push_back(run[i] + rn[i] * shift);
          if (piece.size() >= 3) pieces.push_back(piece);
        }
        if (pieces.empty()) pieces.push_back(run);
      }
      for (auto& piece : pieces) {
        auto rp = resample(chaikin(piece, 2), step);
        Ridge r;
        r.id = int(plan.ridges.size());
        r.rank = 0;
        r.parent = -1;
        r.role = sp.role;
        r.half_width = sp.abs_hw ? sp.hw : sp.hw * hw * (1.0 + 0.9 * m);
        r.sharpness = std::max(0.45, sp.sharp * (1.0 - 0.5 * m));
        r.plateau = sp.plateau;
        r.asymmetry = std::clamp(sp.asym, -0.8, 0.8);
        r.points = to_pts(rp);
        r.crest.assign(rp.size(), sp.crest);
        range_ids.push_back(r.id);
        range_spec.push_back(int(spec_index));
        plan.ridges.push_back(std::move(r));
      }
      placed.push_back({lateral, sp});
    }

    // crest profiles: unevenly spaced peaks, massifs of higher summits, a dominant summit,
    // occasional deep passes; old ranges have accordant summits
    for (size_t q = 0; q < range_ids.size(); ++q) {
      int rid = range_ids[q];
      Ridge& r = plan.ridges[size_t(rid)];
      auto pts = to_vec(r.points);
      auto s = arclength(pts);
      double len = s.back(), base = r.crest.empty() ? 1.0 : r.crest[0];
      bool peaks = specs[size_t(range_spec[q])].peaks;
      if (r.role == ROLE_PLATEAU || r.role == ROLE_BASE) {
        double ph = rng.range(0, 6.283), k = rng.range(1, 3);
        for (size_t i = 0; i < pts.size(); ++i)
          r.crest[i] = base * (1.0 + 0.04 * std::sin(6.283 * k * s[i] / std::max(len, 1.0) + ph));
        continue;
      }
      double e1 = rng.range(0.7, 2.2), e2 = rng.range(2.5, 5.5), p1 = rng.range(0, 6.283),
             p2 = rng.range(0, 6.283);
      auto envelope = [&](double where) {
        double f = where / std::max(len, 1.0);
        return 0.82 + 0.1 * std::sin(6.283 * e1 * f + p1) + 0.08 * std::sin(6.283 * e2 * f + p2);
      };
      double spacing = req.peak_spacing_m * (1.0 + 0.6 * m);
      std::vector<double> ps;
      int count_wanted = req.peak_count > 0 && range_spec[q] == primary ? req.peak_count : 0;
      if (count_wanted > 0) {
        for (int k = 0; k < count_wanted; ++k)
          ps.push_back((k + 0.25 + 0.5 * rng.uniform()) * len / count_wanted);
      } else {
        double pos = 0.5 * gamma_gap(rng, spacing, 0.55);
        while (pos < len - 0.25 * spacing && ps.size() < 10000) {
          ps.push_back(pos);
          pos += gamma_gap(rng, spacing, 0.55);
        }
        if (ps.empty()) ps.push_back(0.5 * len);
      }
      size_t n = ps.size();
      double spread = 0.3 * (1.0 - 0.65 * m);
      std::vector<double> pv(n);
      for (size_t k = 0; k < n; ++k) pv[k] = envelope(ps[k]) * (1.0 - spread + spread * rng.uniform());
      {
        // the highest summit of a range reaches its crest value (the range's own height);
        // the main range also gets one dominant summit
        double top = *std::max_element(pv.begin(), pv.end());
        for (double& v : pv) v /= std::max(top, 1e-9);
        if (r.role == ROLE_MAIN || r.role == ROLE_CORE) {
          size_t dom = size_t(rng.uniform() * double(n)) % n;
          for (size_t k = 0; k < n; ++k)
            if (k != dom) pv[k] *= 0.93;
        }
      }
      std::vector<double> ns{0.0}, nv{0.38}, saddle_s;
      for (size_t k = 0; k < n; ++k) {
        if (k > 0) {
          bool deep = rng.uniform() < 0.14;
          double depth = (deep ? 0.4 : 0.55 + 0.2 * rng.uniform()) + 0.12 * m;
          saddle_s.push_back(0.5 * (ps[k - 1] + ps[k]));
          ns.push_back(saddle_s.back());
          nv.push_back(std::min(pv[k - 1], pv[k]) * std::min(depth, 0.9));
        }
        ns.push_back(ps[k]);
        nv.push_back(pv[k]);
      }
      ns.push_back(len + 1e-6);
      nv.push_back(0.38);
      for (size_t i = 0; i < pts.size(); ++i) r.crest[i] = base * cosine_interp(ns, nv, s[i]);
      if (!peaks) continue;
      auto at = [&](double where) {
        size_t i = 0;
        while (i + 1 < s.size() && s[i] < where) ++i;
        return i;
      };
      std::vector<int> ids;
      for (size_t k = 0; k < n; ++k) {
        size_t i = at(ps[k]);
        ids.push_back(add_peak(rid, pts[i], r.crest[i], 0.65 * r.half_width / (1.0 + 0.9 * m)));
        if (k > 0) {
          size_t j = at(saddle_s[k - 1]);
          if (r.crest[j] < std::min(plan.peaks[size_t(ids[k - 1])].crest, plan.peaks[size_t(ids[k])].crest)) {
            Pass p;
            p.id = int(plan.passes.size());
            p.ridge = rid;
            p.pos = {pts[j].x, pts[j].y};
            p.crest = r.crest[j];
            p.peak_a = ids[k - 1];
            p.peak_b = ids[k];
            plan.passes.push_back(p);
          }
        }
      }
    }

    // the flanks: spurs and valleys are lines of steepest descent of the ranges' relief
    Plan field_plan;
    field_plan.request = req;
    field_plan.external = true;
    for (int rid : range_ids) field_plan.ridges.push_back(plan.ridges[size_t(rid)]);
    field_plan.finalize();
    const double reach = req.spur_length_m + hw;
    uint64_t nseed = hash_combine(req.seed, 0xF1A4Cull);
    auto field = [&](Vec2 p) {
      Sample sm = field_plan.sample(p.x, p.y);
      double rd = sm.ridge_distance < 0 ? 1e30 : sm.ridge_distance;
      double body = 1.0 - smoothstep(rd / (1.6 * reach));
      return sm.structure + 0.3 * body + 0.04 * grad_noise(p.x / (5.0 * hw), p.y / (5.0 * hw), nseed);
    };
    auto trace = [&](Vec2 start, Vec2 dir, double max_len, double stop) {
      std::vector<Vec2> path{start};
      double v0 = field(start), vp = v0, h = 0.6 * step;
      Vec2 pos = start, prev = dir;
      for (int k = 0; k * step < max_len; ++k) {
        double gx = field(pos + Vec2{h, 0}) - field(pos - Vec2{h, 0});
        double gy = field(pos + Vec2{0, h}) - field(pos - Vec2{0, h});
        double g = std::hypot(gx, gy);
        double inertia = k < 2 ? 0.75 : 0.4;
        Vec2 d = g > 1e-9 ? normalized(prev * inertia + Vec2{-gx / g, -gy / g} * (1 - inertia)) : prev;
        Vec2 nxt = pos + d * step;
        double v = field(nxt);
        if (v > vp + 0.004 || v <= stop * v0 || plan.footprint_mask(nxt.x, nxt.y) < 0.15) break;
        path.push_back(nxt);
        pos = nxt;
        prev = d;
        vp = v;
      }
      return path;
    };
    auto add_spur = [&](const Ridge& parent, size_t i, const std::vector<Vec2>& path, double half) {
      std::vector<Vec2> raw{to_vec(parent.points)[i]};
      raw.insert(raw.end(), path.begin(), path.end());
      auto sp = resample(chaikin(raw, 2), std::max(0.2 * req.spur_half_width_m, 80.0));
      auto ss = arclength(sp);
      Ridge r;
      r.id = int(plan.ridges.size());
      r.rank = 2;
      r.parent = parent.id;
      r.role = ROLE_SPUR;
      r.half_width = req.spur_half_width_m * half * (1.0 + 0.6 * m);
      r.sharpness = std::max(0.45, 0.85 * sharp);
      r.points = to_pts(sp);
      double c0 = parent.crest[i], ph = rng.range(0, 6.283), k = rng.range(1.5, 3.5);
      for (size_t q = 0; q < sp.size(); ++q) {
        double f = ss[q] / std::max(ss.back(), 1e-9);
        r.crest.push_back(c0 * (0.74 - 0.52 * std::pow(f, 1.15)) * (1.0 + 0.1 * std::sin(6.283 * k * f + ph)));
      }
      plan.ridges.push_back(std::move(r));
      return int(plan.ridges.size()) - 1;
    };
    double density = std::max(cx, 0.3);
    if (style == PMGE_STYLE_FOLD_BELT) density *= 0.35;
    if (style == PMGE_STYLE_BROAD_UPLAND || style == PMGE_STYLE_OLD_MASSIF) density *= 0.6;
    for (int rid : std::vector<int>(range_ids)) {
      Ridge parent = plan.ridges[size_t(rid)];
      if (parent.role == ROLE_PLATEAU || parent.role == ROLE_BASE || parent.role == ROLE_FRONTAL ||
          req.spur_length_m <= 0)
        continue;
      auto pts = to_vec(parent.points);
      auto s = arclength(pts);
      double len = s.back();
      double gap = req.spur_spacing_m * (1.0 + 0.45 * m) / density;
      struct Root { double at; size_t i; int side; int spur; };
      std::vector<Root> roots;
      for (int side : {1, -1}) {
        double widen = 1.0 + 0.8 * parent.asymmetry * side;
        double pos = gap * rng.uniform();
        while (pos < len - 0.1 * gap) {
          size_t i = 1;
          while (i + 2 < s.size() && s[i] < pos) ++i;
          Vec2 nrm = perp(normalized(pts[i + 1] - pts[i - 1])) * double(side);
          double ang = (rng.uniform() + rng.uniform() - 1.0) * 0.55;
          Vec2 d{std::cos(ang) * nrm.x - std::sin(ang) * nrm.y, std::sin(ang) * nrm.x + std::cos(ang) * nrm.y};
          double lognorm = std::exp((rng.uniform() + rng.uniform() - 1.0) * 0.6);
          double max_len = req.spur_length_m * widen * lognorm * (1.0 - 0.3 * m);
          auto path = trace(pts[i] + nrm * (0.2 * parent.half_width * widen), d, max_len, 0.3);
          int spur = -1;
          if (path.size() >= 4) spur = add_spur(parent, i, path, 1.0);
          roots.push_back({pos, i, side, spur});
          pos += gamma_gap(rng, gap, 0.55);
        }
      }
      // second-order spurs leave a spur sideways and descend with the flank
      for (const Root& rt : roots) {
        if (rt.spur < 0 || rng.uniform() > 0.6 * (1.0 - 0.5 * m)) continue;
        Ridge sp = plan.ridges[size_t(rt.spur)];
        auto sp_pts = to_vec(sp.points);
        if (sp_pts.size() < 6) continue;
        size_t q = std::clamp(size_t((0.25 + 0.4 * rng.uniform()) * double(sp_pts.size() - 1)),
                              size_t(1), sp_pts.size() - 2);
        Vec2 tq = normalized(sp_pts[q + 1] - sp_pts[q - 1]);
        Vec2 d2 = normalized(tq * 0.55 + perp(tq) * (rng.uniform() < 0.5 ? 0.8 : -0.8));
        double l2 = arclength(sp_pts).back() * rng.range(0.25, 0.5);
        auto path = trace(sp_pts[q] + d2 * (0.4 * sp.half_width), d2, l2, 0.3);
        if (path.size() >= 3) add_spur(sp, q, path, 0.7);
      }
      // valleys: heads between neighbouring spurs of one flank
      if (req.valley_depth > 0) {
        for (int side : {1, -1}) {
          std::vector<Root> list;
          for (const Root& rt : roots)
            if (rt.side == side && rt.spur >= 0) list.push_back(rt);
          std::sort(list.begin(), list.end(), [](const Root& a, const Root& b) { return a.at < b.at; });
          for (size_t k = 0; k + 1 < list.size(); ++k) {
            if (list[k + 1].at - list[k].at > 2.6 * gap) continue;
            size_t mid = (list[k].i + list[k + 1].i) / 2;
            mid = std::clamp(mid, size_t(1), pts.size() - 2);
            Vec2 nrm = perp(normalized(pts[mid + 1] - pts[mid - 1])) * double(side);
            double l = 1.1 * 0.5 *
                       (arclength(to_vec(plan.ridges[size_t(list[k].spur)].points)).back() +
                        arclength(to_vec(plan.ridges[size_t(list[k + 1].spur)].points)).back());
            auto path = trace(pts[mid] + nrm * (0.45 * parent.half_width), nrm, l, 0.2);
            if (path.size() < 3) continue;
            auto vp = resample(chaikin(path, 2), std::max(0.25 * hw, 100.0));
            auto vs = arclength(vp);
            Valley v;
            v.id = int(plan.valleys.size());
            v.ridge = parent.id;
            v.half_width = (0.42 + 0.12 * m) * (list[k + 1].at - list[k].at);
            v.points = to_pts(vp);
            for (size_t q = 0; q < vp.size(); ++q) {
              double f = vs[q] / std::max(vs.back(), 1e-9);
              v.depth.push_back(req.valley_depth * std::clamp(f / 0.25, 0.0, 1.0) * (1.0 - 0.75 * f * f));
            }
            plan.valleys.push_back(std::move(v));
          }
        }
      }
    }

    // intermontane basins between neighbouring parallel ranges
    if (req.valley_depth > 0 && placed.size() >= 2) {
      std::sort(placed.begin(), placed.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      auto nrm = normals_of(base_axis);
      for (size_t k = 0; k + 1 < placed.size(); ++k) {
        const auto& [la, sa] = placed[k];
        const auto& [lb, sb] = placed[k + 1];
        if (sa.role == ROLE_PLATEAU || sa.role == ROLE_BASE || sb.role == ROLE_PLATEAU ||
            sb.role == ROLE_BASE)
          continue;
        double gap_m = lb - la, half = 0.32 * gap_m;
        if (half < 0.8 * hw) continue;
        double lo = std::max(sa.trim_lo, sb.trim_lo), hi = std::max(sa.trim_hi, sb.trim_hi);
        std::vector<Vec2> pts;
        for (size_t i = 0; i < base_axis.size(); ++i) {
          double f = s_axis[i] / std::max(total, 1e-9);
          if (f >= lo && f <= 1.0 - hi) pts.push_back(base_axis[i] + nrm[i] * (0.5 * (la + lb)));
        }
        if (pts.size() < 4) continue;
        auto vp = resample(chaikin(pts, 2), std::max(0.25 * hw, 100.0));
        auto vs = arclength(vp);
        Valley v;
        v.id = int(plan.valleys.size());
        v.ridge = -1;
        v.half_width = half;
        v.points = to_pts(vp);
        for (size_t q = 0; q < vp.size(); ++q) {
          double f = vs[q] / std::max(vs.back(), 1e-9);
          v.depth.push_back(std::min(req.valley_depth, 0.42 + 0.12 * m) *
                            std::clamp(std::min(f, 1.0 - f) / 0.12, 0.0, 1.0));
        }
        plan.valleys.push_back(std::move(v));
      }
    }
  }
  plan.finalize();
  return plan;
}

std::string Plan::to_json() const {
  std::string out;
  char buf[160];
  auto num = [&](double v) {
    std::snprintf(buf, sizeof buf, "%.3f", v);
    out += buf;
  };
  auto pts = [&](const std::vector<Point>& p) {
    out += "[";
    for (size_t i = 0; i < p.size(); ++i) {
      if (i) out += ",";
      out += "[";
      num(p[i].x);
      out += ",";
      num(p[i].y);
      out += "]";
    }
    out += "]";
  };
  auto arr = [&](const std::vector<double>& v) {
    out += "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) out += ",";
      num(v[i]);
    }
    out += "]";
  };
  static const char* ranks[] = {"main", "branch", "spur"};
  static const char* types[] = {"sharp", "jagged", "rounded", "dome",
                                "flat", "needle", "volcanic", "truncated"};
  out += "{\"schema_version\":\"1.0\",\"units\":\"m\",\"library\":\"";
  out += pmge_version();
  static const char* styles[] = {"auto",     "volcanic_arc", "alpine",       "himalayan",
                                 "andean",   "fold_belt",    "old_massif",   "broad_upland",
                                 "mixed",    "block"};
  std::snprintf(buf, sizeof buf,
                "\",\"seed\":%llu,\"external\":%s,\"style\":\"%s\",\"maturity_eff\":%.4f,"
                "\"height_scale\":%.4f,\"ridges\":[",
                static_cast<unsigned long long>(request.seed), external ? "true" : "false",
                styles[std::clamp(style, 0, 9)], maturity_eff, height_scale);
  out += buf;
  for (size_t i = 0; i < ridges.size(); ++i) {
    const Ridge& r = ridges[i];
    if (i) out += ",";
    std::snprintf(buf, sizeof buf, "{\"id\":%d,\"system\":%d,\"rank\":\"%s\",\"parent\":%d,", r.id,
                  r.system, ranks[std::clamp(r.rank, 0, 2)], r.parent);
    out += buf;
    static const char* roles[] = {"main",  "frontal", "secondary", "back", "plateau", "base",
                                  "core",  "fold",    "block",     "branch", "spur"};
    out += "\"role\":\"";
    out += roles[std::clamp(r.role, 0, 10)];
    out += "\",\"half_width_m\":";
    num(r.half_width);
    out += ",\"sharpness\":";
    num(r.sharpness);
    out += ",\"plateau\":";
    num(r.plateau);
    out += ",\"asymmetry\":";
    num(r.asymmetry);
    out += ",\"points_m\":";
    pts(r.points);
    out += ",\"crest\":";
    arr(r.crest);
    out += "}";
  }
  out += "],\"peaks\":[";
  for (size_t i = 0; i < peaks.size(); ++i) {
    const Peak& p = peaks[i];
    if (i) out += ",";
    std::snprintf(buf, sizeof buf,
                  "{\"id\":%d,\"stable_id\":\"%016llx\",\"ridge\":%d,\"system\":%d,\"type\":\"%s\",",
                  p.id, static_cast<unsigned long long>(p.stable_id), p.ridge, p.system,
                  types[std::clamp(p.type, 0, 7)]);
    out += buf;
    out += "\"x_m\":";
    num(p.pos.x);
    out += ",\"y_m\":";
    num(p.pos.y);
    out += ",\"crest\":";
    num(p.crest);
    out += ",\"radius_m\":";
    num(p.radius);
    out += "}";
  }
  out += "],\"passes\":[";
  for (size_t i = 0; i < passes.size(); ++i) {
    const Pass& p = passes[i];
    if (i) out += ",";
    std::snprintf(buf, sizeof buf, "{\"id\":%d,\"ridge\":%d,\"between_peaks\":[%d,%d],", p.id,
                  p.ridge, p.peak_a, p.peak_b);
    out += buf;
    out += "\"x_m\":";
    num(p.pos.x);
    out += ",\"y_m\":";
    num(p.pos.y);
    out += ",\"crest\":";
    num(p.crest);
    out += "}";
  }
  out += "],\"valleys\":[";
  for (size_t i = 0; i < valleys.size(); ++i) {
    const Valley& v = valleys[i];
    if (i) out += ",";
    std::snprintf(buf, sizeof buf, "{\"id\":%d,\"ridge\":%d,\"half_width_m\":", v.id, v.ridge);
    out += buf;
    num(v.half_width);
    out += ",\"points_m\":";
    pts(v.points);
    out += ",\"depth\":";
    arr(v.depth);
    out += "}";
  }
  out += "]}";
  return out;
}

}  // namespace pmge
