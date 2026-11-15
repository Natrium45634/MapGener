// Internal helpers: deterministic hashing, RNG, gradient noise, small geometry.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace pmge {

struct Vec2 {
  double x = 0, y = 0;
};
inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(Vec2 a, double s) { return {a.x * s, a.y * s}; }
inline double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline double length(Vec2 a) { return std::hypot(a.x, a.y); }
inline Vec2 normalized(Vec2 a) {
  double l = length(a);
  return l > 1e-12 ? Vec2{a.x / l, a.y / l} : Vec2{1, 0};
}
inline Vec2 perp(Vec2 a) { return {-a.y, a.x}; }

// splitmix64: the only source of randomness. No std::rand, no global state.
inline uint64_t mix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
inline uint64_t hash_combine(uint64_t a, uint64_t b) { return mix64(a ^ (mix64(b) + 0x632BE59BD9B4E019ull)); }

struct Rng {
  uint64_t state;
  explicit Rng(uint64_t seed) : state(mix64(seed)) {}
  uint64_t next() { return state = mix64(state); }
  double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
  double range(double lo, double hi) { return lo + (hi - lo) * uniform(); }
  int below(int n) { return std::min(n - 1, int(uniform() * n)); }
};

inline double smoothstep(double t) {
  t = std::clamp(t, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// Gradient noise from hashed lattice gradients (no tables, so no global state).
inline double grad_noise(double x, double y, uint64_t seed) {
  double xf = std::floor(x), yf = std::floor(y);
  int64_t xi = int64_t(xf), yi = int64_t(yf);
  double dx = x - xf, dy = y - yf;
  auto corner = [&](int64_t ix, int64_t iy, double fx, double fy) {
    uint64_t h = hash_combine(seed, hash_combine(uint64_t(ix), uint64_t(iy) * 0x9E3779B1ull));
    double a = double(h >> 11) * (6.283185307179586 / 9007199254740992.0);
    return std::cos(a) * fx + std::sin(a) * fy;
  };
  double u = dx * dx * dx * (dx * (dx * 6 - 15) + 10);
  double v = dy * dy * dy * (dy * (dy * 6 - 15) + 10);
  double n00 = corner(xi, yi, dx, dy), n10 = corner(xi + 1, yi, dx - 1, dy);
  double n01 = corner(xi, yi + 1, dx, dy - 1), n11 = corner(xi + 1, yi + 1, dx - 1, dy - 1);
  double a = n00 + u * (n10 - n00), b = n01 + u * (n11 - n01);
  return (a + v * (b - a)) * 1.4142135623730951;
}

// Bounded multi-scale detail in [0, 1]: ridged octaves, never the main structure.
inline double ridged_detail(double x, double y, double wavelength, uint64_t seed) {
  double sum = 0, amp = 1, norm = 0;
  for (int k = 0; k < 3; ++k) {
    double n = grad_noise(x / wavelength, y / wavelength, seed + uint64_t(k) * 7919u);
    sum += amp * (1.0 - std::min(1.0, std::abs(n)));
    norm += amp;
    amp *= 0.55;
    wavelength *= 0.5;
  }
  return sum / norm;
}

inline std::vector<Vec2> chaikin(std::vector<Vec2> p, int rounds) {
  for (int r = 0; r < rounds && p.size() >= 3; ++r) {
    std::vector<Vec2> q;
    q.reserve(p.size() * 2);
    q.push_back(p.front());
    for (size_t i = 0; i + 1 < p.size(); ++i) {
      q.push_back(p[i] * 0.75 + p[i + 1] * 0.25);
      q.push_back(p[i] * 0.25 + p[i + 1] * 0.75);
    }
    q.push_back(p.back());
    p.swap(q);
  }
  return p;
}

inline std::vector<double> arclength(const std::vector<Vec2>& p) {
  std::vector<double> s(p.size(), 0.0);
  for (size_t i = 1; i < p.size(); ++i) s[i] = s[i - 1] + length(p[i] - p[i - 1]);
  return s;
}

inline std::vector<Vec2> resample(const std::vector<Vec2>& p, double step) {
  auto s = arclength(p);
  double total = s.empty() ? 0.0 : s.back();
  if (p.size() < 2 || total < 1e-9) return p;
  int n = std::max(2, int(std::lround(total / step)) + 1);
  std::vector<Vec2> out(n);
  size_t j = 0;
  for (int i = 0; i < n; ++i) {
    double t = total * i / (n - 1);
    while (j + 2 < p.size() && s[j + 1] < t) ++j;
    double seg = std::max(s[j + 1] - s[j], 1e-12);
    double w = std::clamp((t - s[j]) / seg, 0.0, 1.0);
    out[i] = p[j] * (1 - w) + p[j + 1] * w;
  }
  return out;
}

// Worker count: PMGE_THREADS if set (1 = single-threaded), otherwise all hardware threads.
inline int worker_count() {
  if (const char* env = std::getenv("PMGE_THREADS")) {
    int n = std::atoi(env);
    if (n >= 1) return n;
  }
  unsigned hw = std::thread::hardware_concurrency();
  return hw ? int(hw) : 1;
}

// Runs f(begin, end) over [0, n) in contiguous blocks. Every index is computed by exactly
// one worker and independently of the others, so results never depend on the thread count.
template <class F>
void parallel_for(int64_t n, F&& f) {
  int workers = int(std::min<int64_t>(worker_count(), std::max<int64_t>(1, n / 256)));
  if (workers <= 1) {
    f(int64_t(0), n);
    return;
  }
  std::vector<std::thread> pool;
  std::exception_ptr error;
  std::mutex guard;
  int64_t block = (n + workers - 1) / workers;
  for (int w = 0; w < workers; ++w) {
    int64_t a = w * block, b = std::min(n, a + block);
    if (a >= b) break;
    pool.emplace_back([&, a, b] {
      try {
        f(a, b);
      } catch (...) {
        std::lock_guard<std::mutex> lock(guard);
        if (!error) error = std::current_exception();
      }
    });
  }
  for (auto& t : pool) t.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace pmge
