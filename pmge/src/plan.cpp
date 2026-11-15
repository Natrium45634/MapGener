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
  if (r.foothill_profile < PMGE_FOOTHILL_SMOOTH || r.foothill_profile > PMGE_FOOTHILL_TERRACED)
    fail("foothill_profile is not a PMGE_FOOTHILL_* value");
  double wsum = 0;
  for (double w : r.peak_type_weights) {
    if (w < 0 || !std::isfinite(w)) fail("peak_type_weights must be finite and non-negative");
    wsum += w;
  }
  if (!(wsum > 0)) fail("at least one peak type weight must be positive");
}

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
  const double sharp = 1.55 - 0.85 * req.maturity;

  auto add_peak = [&](int ridge, Vec2 pos, double crest, double radius) {
    Peak p;
    p.id = int(plan.peaks.size());
    p.ridge = ridge;
    p.system = 0;
    p.pos = {pos.x, pos.y};
    p.crest = crest;
    p.type = pick_type(req, rng);
    p.radius = radius;
    p.stable_id = hash_combine(req.seed, hash_combine(uint64_t(ridge) + 1, uint64_t(p.id) + 1));
    plan.peaks.push_back(p);
    return p.id;
  };

  // -- the main axis ----------------------------------------------------------
  std::vector<Vec2> axis;
  bool massif = false;
  if (req.footprint_type == PMGE_FOOTPRINT_LINE) {
    axis = to_vec(line_in);
  } else if (req.footprint_type == PMGE_FOOTPRINT_BOX) {
    Vec2 lo{req.box_min_x, req.box_min_y}, hi{req.box_max_x, req.box_max_y};
    Vec2 size = hi - lo, mid = (lo + hi) * 0.5;
    bool wide = size.x >= size.y;
    Vec2 along = wide ? Vec2{1, 0} : Vec2{0, 1};
    double len = (wide ? size.x : size.y) * 0.5 - 1.6 * hw;
    double side = (wide ? size.y : size.x) * 0.5;
    if (len < hw) len = hw;
    for (int k = 0; k <= 4; ++k) {
      double t = -1.0 + 0.5 * k;
      double off = (k == 0 || k == 4) ? 0.0 : rng.range(-0.22, 0.22) * side;
      axis.push_back(mid + along * (t * len) + perp(along) * off);
    }
  } else {
    Vec2 c{req.center_x, req.center_y};
    double R = req.radius_m;
    int wanted = req.peak_count > 0 ? req.peak_count
                                    : std::max(1, int(std::lround(1.3 * R / req.peak_spacing_m)));
    int lone_type = pick_type(req, rng);
    bool edifice = lone_type == PMGE_PEAK_VOLCANIC || lone_type == PMGE_PEAK_DOME ||
                   lone_type == PMGE_PEAK_FLAT || lone_type == PMGE_PEAK_TRUNCATED ||
                   lone_type == PMGE_PEAK_ROUNDED;
    if (wanted <= 1 && edifice) {
      // a lone volcano, dome or table mountain is one edifice, not a star of ridges
      massif = true;
      int id = add_peak(-1, c, 1.0, 0.82 * R);
      plan.peaks[id].type = lone_type;
    } else if (wanted <= 1) {
      massif = true;  // one horn: aretes radiate from the summit
      int arms = 4 + rng.below(3);
      double a0 = rng.range(0, 6.283185307179586);
      for (int k = 0; k < arms; ++k) {
        double a = a0 + 6.283185307179586 * (k + rng.range(-0.18, 0.18)) / arms;
        double len = 0.8 * R * rng.range(0.72, 1.0);
        Vec2 d{std::cos(a), std::sin(a)};
        Vec2 bend = perp(d) * (rng.range(-0.18, 0.18) * len);
        std::vector<Vec2> raw{c, c + d * (0.55 * len) + bend, c + d * len};
        auto pts = resample(chaikin(raw, 2), step);
        Ridge r;
        r.id = int(plan.ridges.size());
        r.rank = k == 0 ? 0 : 1;
        r.parent = k == 0 ? -1 : 0;
        r.half_width = hw * 0.8;
        r.sharpness = sharp;
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

  if (!massif) {
    auto pts = resample(chaikin(axis, 2), step);
    auto s = arclength(pts);
    double total = s.back();
    Ridge main;
    main.id = 0;
    main.rank = 0;
    main.half_width = hw;
    main.sharpness = sharp;
    main.points = to_pts(pts);
    int n = req.peak_count > 0 ? req.peak_count
                               : std::max(1, int(std::lround(total / req.peak_spacing_m)));
    std::vector<double> ps(n), pv(n), ns{0.0}, nv{0.38};
    for (int k = 0; k < n; ++k) {
      ps[k] = (k + 0.25 + 0.5 * rng.uniform()) * total / n;
      pv[k] = 0.8 + 0.2 * rng.uniform();
    }
    std::vector<double> saddle_s, saddle_v;
    for (int k = 0; k < n; ++k) {
      if (k > 0) {
        double depth = 0.52 + 0.2 * rng.uniform() + 0.12 * req.maturity;
        saddle_s.push_back(0.5 * (ps[k - 1] + ps[k]));
        saddle_v.push_back(std::min(pv[k - 1], pv[k]) * std::min(depth, 0.9));
        ns.push_back(saddle_s.back());
        nv.push_back(saddle_v.back());
      }
      ns.push_back(ps[k]);
      nv.push_back(pv[k]);
    }
    ns.push_back(total + 1e-6);
    nv.push_back(0.38);
    main.crest.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) main.crest[i] = cosine_interp(ns, nv, s[i]);
    auto at = [&](double where) {
      size_t i = 0;
      while (i + 1 < s.size() && s[i] < where) ++i;
      return i;
    };
    std::vector<int> ids;
    for (int k = 0; k < n; ++k) {
      size_t i = at(ps[k]);
      main.crest[i] = pv[k];
      ids.push_back(add_peak(0, pts[i], pv[k], 0.65 * hw));
      if (k > 0) {
        size_t j = at(saddle_s[k - 1]);
        Pass p;
        p.id = int(plan.passes.size());
        p.ridge = 0;
        p.pos = {pts[j].x, pts[j].y};
        p.crest = main.crest[j];
        p.peak_a = ids[k - 1];
        p.peak_b = ids[k];
        plan.passes.push_back(p);
      }
    }
    plan.ridges.push_back(main);

    // -- spurs on alternating sides, valleys between neighbours on one side -------
    struct Root {
      double at;
      size_t i;
      Vec2 dir;
      double len;
    };
    std::vector<Root> roots[2];
    // spurs on both flanks, each flank with the full spacing (staggered by half a step)
    int per_side = req.spur_length_m > 0 ? int(total / req.spur_spacing_m) : 0;
    int n_spurs = 2 * per_side;
    for (int k = 0; k < n_spurs; ++k) {
      int side = k % 2;
      int slot = k / 2;
      double where = (slot + 0.5 * side + 0.25 + 0.4 * rng.uniform()) * total / (per_side + 0.5);
      size_t i = std::clamp(at(where), size_t(1), pts.size() - 2);
      Vec2 tangent = normalized(pts[i + 1] - pts[i - 1]);
      Vec2 normal = perp(tangent) * (side ? 1.0 : -1.0);
      double ang = rng.range(-0.44, 0.44);
      Vec2 dir{std::cos(ang) * normal.x - std::sin(ang) * normal.y,
               std::sin(ang) * normal.x + std::cos(ang) * normal.y};
      double len = req.spur_length_m * rng.range(0.7, 1.25);
      double bend = rng.range(-0.18, 0.18);
      for (int tries = 0; tries < 6; ++tries) {
        Vec2 tip = pts[i] + dir * len;
        if (plan.footprint_mask(tip.x, tip.y) > 0.3) break;
        len *= 0.8;
      }
      if (len < 0.25 * req.spur_length_m) continue;
      std::vector<Vec2> raw{pts[i], pts[i] + (dir + perp(dir) * bend) * (0.55 * len),
                            pts[i] + dir * len};
      auto sp = resample(chaikin(raw, 2), std::max(0.2 * req.spur_half_width_m, 80.0));
      auto ss = arclength(sp);
      Ridge r;
      r.id = int(plan.ridges.size());
      r.rank = 2;
      r.parent = 0;
      r.half_width = req.spur_half_width_m;
      r.sharpness = 0.85 * sharp;  // flank spurs are rounder than the main crest
      r.points = to_pts(sp);
      for (size_t q = 0; q < sp.size(); ++q) {
        double f = ss[q] / std::max(ss.back(), 1e-9);
        r.crest.push_back(main.crest[i] * (0.74 - 0.56 * std::pow(f, 1.2)) *
                          (1.0 + 0.08 * std::sin(6.3 * f + rng.range(0, 3))));
      }
      int spur_id = r.id;
      std::vector<Vec2> spur_pts = sp;
      std::vector<double> spur_crest = r.crest;
      plan.ridges.push_back(std::move(r));
      roots[side].push_back({where, i, dir, len});
      // second-order spurs: the dendritic pattern that fills a real mountain flank
      for (int side2 : {-1, 1}) {
        if (rng.uniform() < 0.25) continue;
        double at = rng.range(0.3, 0.6);
        size_t q = std::min(spur_pts.size() - 2, size_t(at * (spur_pts.size() - 1)));
        Vec2 t2 = normalized(spur_pts[q + 1] - spur_pts[q]);
        Vec2 d2 = normalized(t2 * 0.75 + perp(t2) * (0.65 * side2));
        double l2 = len * rng.range(0.32, 0.5);
        std::vector<Vec2> raw2{spur_pts[q], spur_pts[q] + d2 * l2};
        auto sp2 = resample(raw2, std::max(0.2 * req.spur_half_width_m, 80.0));
        auto ss2 = arclength(sp2);
        Ridge r2;
        r2.id = int(plan.ridges.size());
        r2.rank = 2;
        r2.parent = spur_id;
        r2.half_width = 0.75 * req.spur_half_width_m;
        r2.sharpness = sharp;
        r2.points = to_pts(sp2);
        for (size_t k2 = 0; k2 < sp2.size(); ++k2) {
          double f = ss2[k2] / std::max(ss2.back(), 1e-9);
          r2.crest.push_back(spur_crest[q] * (0.9 - 0.6 * f));
        }
        plan.ridges.push_back(std::move(r2));
      }
    }
    if (req.valley_depth > 0) {
      for (auto& list : roots) {
        for (size_t k = 0; k + 1 < list.size(); ++k) {
          const Root &a = list[k], &b = list[k + 1];
          if (b.at - a.at > 2.4 * req.spur_spacing_m) continue;
          size_t mid = (a.i + b.i) / 2;
          Vec2 dir = normalized(a.dir + b.dir);
          Vec2 head = pts[mid] + dir * (0.45 * hw);
          double len = 0.95 * 0.5 * (a.len + b.len);
          auto vp = resample({head, head + dir * len}, std::max(0.25 * hw, 100.0));
          auto vs = arclength(vp);
          Valley v;
          v.id = int(plan.valleys.size());
          v.ridge = 0;
          // a valley fills the gap between two spurs on the same side
          v.half_width = (0.42 + 0.12 * req.maturity) * (b.at - a.at);
          v.points = to_pts(vp);
          for (size_t q = 0; q < vp.size(); ++q) {
            double f = vs[q] / std::max(vs.back(), 1e-9);
            v.depth.push_back(req.valley_depth * std::clamp(f / 0.25, 0.0, 1.0) *
                              (1.0 - 0.75 * f * f));
          }
          plan.valleys.push_back(std::move(v));
        }
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
  std::snprintf(buf, sizeof buf, "\",\"seed\":%llu,\"external\":%s,\"ridges\":[",
                static_cast<unsigned long long>(request.seed), external ? "true" : "false");
  out += buf;
  for (size_t i = 0; i < ridges.size(); ++i) {
    const Ridge& r = ridges[i];
    if (i) out += ",";
    std::snprintf(buf, sizeof buf, "{\"id\":%d,\"system\":%d,\"rank\":\"%s\",\"parent\":%d,", r.id,
                  r.system, ranks[std::clamp(r.rank, 0, 2)], r.parent);
    out += buf;
    out += "\"half_width_m\":";
    num(r.half_width);
    out += ",\"sharpness\":";
    num(r.sharpness);
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
