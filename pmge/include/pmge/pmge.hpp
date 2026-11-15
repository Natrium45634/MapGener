// PMGE C++ API. The C ABI in pmge.h is a thin layer over these types.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pmge.h"

namespace pmge {

struct Vec2;

struct Point {
  double x = 0, y = 0;
};

struct Ridge {
  int id = 0, system = 0, rank = 0, parent = -1;  // rank: 0 main, 1 branch, 2 spur
  std::vector<Point> points;
  std::vector<double> crest;  // relative crest height 0..1 per point
  double half_width = 0;      // m
  double sharpness = 1.2;     // exponent of the cross profile
};

struct Peak {
  int id = 0, ridge = 0, system = 0;
  Point pos;
  double crest = 1;   // relative
  int type = 0;       // PMGE_PEAK_*
  double radius = 0;  // m
  uint64_t stable_id = 0;
};

struct Pass {
  int id = 0, ridge = 0;
  Point pos;
  double crest = 0;
  int peak_a = -1, peak_b = -1;
};

struct Valley {
  int id = 0, system = 0, ridge = -1;
  std::vector<Point> points;
  std::vector<double> depth;  // relative 0..1 per point
  double half_width = 0;
};

struct Grid {
  double x0 = 0, y0 = 0, cell = 1;
  int nx = 0, ny = 0;
  std::vector<float> values;
  bool empty() const { return values.empty(); }
  double at(double x, double y) const;  // bilinear, clamped
};

struct Sample {
  double height = 0, relief = 0, structure = 0, valley = 0, ridge_distance = -1;
  int system = -1, class_id = 0;
};

class Plan {
 public:
  pmge_request request{};
  bool external = false;
  double mean_offset = 0;  // external plans: share of the amplitude subtracted everywhere
  std::vector<Point> line;  // footprint polyline (PMGE_FOOTPRINT_LINE)
  std::vector<Ridge> ridges;
  std::vector<Peak> peaks;
  std::vector<Pass> passes;
  std::vector<Valley> valleys;
  Grid amplitude, gain;

  void finalize();  // builds the spatial index; call after editing the lists
  Sample sample(double x, double y) const;
  double footprint_mask(double x, double y) const;  // 1 inside, 0 outside, smooth edge
  int nearest_pass(double x, double y, double reach) const;
  std::string to_json() const;

 private:
  struct Segment {
    double ax, ay, bx, by, ca, cb, half_width, sharpness;
    int system, ridge;
  };
  struct VSegment {
    double ax, ay, bx, by, da, db, half_width;
  };
  std::vector<Segment> segments_;
  std::vector<VSegment> vsegments_;
  std::vector<Segment> body_;  // main and branch axes: the mountain body is built around them
  std::unordered_map<uint64_t, std::vector<uint32_t>> body_index_;
  double body_cell_ = 1, body_half_ = 0, foot_reach_ = 0;
  double cell_ = 1, vcell_ = 1, pcell_ = 1;
  std::unordered_map<uint64_t, std::vector<uint32_t>> index_, vindex_, pindex_, passindex_;
  double range_limit_ = 0;
};

// Throws std::invalid_argument with a readable reason if the request is unusable.
void validate(const pmge_request& request);
Plan make_plan(const pmge_request& request, const std::vector<Point>& line);
void generate_heightmap(const Plan& plan, double x0, double y0, double cell, int nx, int ny,
                        float* out);
void generate_patch(const Plan& plan, double x0, double y0, double cell, int nx, int ny,
                    const float* existing, int policy, float* delta);
void sample_hexes(const Plan& plan, const double* cx, const double* cy, int64_t n, double radius,
                  int orientation, int strategy, pmge_hex_sample* out);
void erode_thermal(float* h, int nx, int ny, double cell, double talus_deg, int iterations,
                   double strength);
void erode_fluvial(float* h, int nx, int ny, double cell, double erodibility, int iterations);

}  // namespace pmge
