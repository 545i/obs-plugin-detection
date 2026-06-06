// tracker.hpp -- lightweight SORT-style multi-object tracker with per-box Kalman
// smoothing. std-only. Operates in the SAME coordinate space as the detections
// it is fed (here: 640x640 network space).
//
// Each track smooths center (cx,cy) and size (w,h) with four DECOUPLED 1-D
// constant-velocity Kalman filters (state = [value, velocity]). Decoupling
// avoids 4x4 matrix inversion and is plenty for box smoothing. predict() runs
// every frame (smooth motion between detections); update() runs when a new
// detection set arrives (greedy per-class IoU association).
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "detection_postprocess.hpp"  // Detection { x,y,w,h (top-left), cls, score }

// 1-D constant-velocity Kalman filter. State: value x, velocity v.
struct Kalman1D {
	float x = 0.0f, v = 0.0f;
	float p00 = 1, p01 = 0, p10 = 0, p11 = 1;  // covariance
	float q_pos = 1.0f, q_vel = 25.0f, r = 16.0f;

	void reset(float z)
	{
		x = z;
		v = 0.0f;
		p00 = 1; p01 = 0; p10 = 0; p11 = 1;
	}

	void predict(float dt)
	{
		x += v * dt;  // F = [[1,dt],[0,1]]
		// P = F P F^T + Q
		const float p00n = p00 + dt * (p01 + p10) + dt * dt * p11 + q_pos;
		const float p01n = p01 + dt * p11;
		const float p10n = p10 + dt * p11;
		const float p11n = p11 + q_vel * dt;
		p00 = p00n; p01 = p01n; p10 = p10n; p11 = p11n;
	}

	void update(float z)
	{
		const float S = p00 + r;       // H = [1,0]
		const float k0 = p00 / S;      // Kalman gain
		const float k1 = p10 / S;
		const float y = z - x;         // innovation
		x += k0 * y;
		v += k1 * y;
		const float p00n = (1.0f - k0) * p00;
		const float p01n = (1.0f - k0) * p01;
		const float p10n = p10 - k1 * p00;
		const float p11n = p11 - k1 * p01;
		p00 = p00n; p01 = p01n; p10 = p10n; p11 = p11n;
	}
};

struct Track {
	Kalman1D cx, cy, w, h;  // center x/y, width, height
	int cls = 0;
	int id = 0;
	float score = 0.0f;  // last matched detection score
	int hits = 0;     // total matched detections
	int misses = 0;   // consecutive frames without a match
	int age = 0;

	void init(const Detection &d, float r, float q_pos, float q_vel)
	{
		for (Kalman1D *k : {&cx, &cy, &w, &h}) {
			k->r = r;
			k->q_pos = q_pos;
			k->q_vel = q_vel;
		}
		cx.reset(d.x + d.w * 0.5f);
		cy.reset(d.y + d.h * 0.5f);
		w.reset(d.w);
		h.reset(d.h);
		cls = d.cls;
		score = d.score;
		hits = 1;
		misses = 0;
		age = 0;
	}

	void predict(float dt)
	{
		cx.predict(dt);
		cy.predict(dt);
		w.predict(dt);
		h.predict(dt);
		++age;
	}

	void correct(const Detection &d)
	{
		cx.update(d.x + d.w * 0.5f);
		cy.update(d.y + d.h * 0.5f);
		w.update(d.w);
		h.update(d.h);
		score = d.score;
		++hits;
		misses = 0;
	}

	// Smoothed box as top-left x,y + w,h (clamped to non-negative size).
	Detection box() const
	{
		Detection d;
		d.w = std::max(0.0f, w.x);
		d.h = std::max(0.0f, h.x);
		d.x = cx.x - d.w * 0.5f;
		d.y = cy.x - d.h * 0.5f;
		d.cls = cls;
		d.score = score;
		return d;
	}
};

// A smoothed box plus its stable track id (for downstream coordinate use).
struct TrackedBox {
	Detection box;  // tracker coord space (here: 640-net space)
	int id;         // stable across frames
	float vx = 0.0f, vy = 0.0f;  // center velocity, tracker units/second
	                             // (Kalman estimate; 0 in raw/no-track mode)
};

inline float track_iou(const Detection &a, const Detection &b)
{
	const float ax2 = a.x + a.w, ay2 = a.y + a.h;
	const float bx2 = b.x + b.w, by2 = b.y + b.h;
	const float ix1 = std::max(a.x, b.x), iy1 = std::max(a.y, b.y);
	const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
	const float iw = std::max(0.0f, ix2 - ix1);
	const float ih = std::max(0.0f, iy2 - iy1);
	const float inter = iw * ih;
	const float uni = a.w * a.h + b.w * b.h - inter;
	return uni > 0.0f ? inter / uni : 0.0f;
}

class BoxTracker {
public:
	// Tuning. Higher r / lower q_* => smoother but laggier.
	float iou_thresh = 0.3f;
	float r = 16.0f;       // measurement noise
	float q_pos = 1.0f;    // position process noise
	float q_vel = 25.0f;   // velocity process noise
	int min_hits = 2;      // detections before a track is drawn
	int max_misses = 30;   // cull a track after this many missed frames
	int show_misses = 12;  // keep drawing a (predicted) track this long after loss

	void clear() { tracks_.clear(); }

	// Advance all tracks by dt (seconds); call once per frame.
	void predict(float dt)
	{
		for (Track &t : tracks_)
			t.predict(dt);
	}

	// Associate a fresh detection set and correct/spawn/cull tracks.
	void update(const std::vector<Detection> &dets)
	{
		const size_t nt = tracks_.size();
		std::vector<char> t_matched(nt, 0);
		std::vector<char> d_matched(dets.size(), 0);

		struct Pair { float iou; int di; int tj; };
		std::vector<Pair> pairs;
		for (size_t di = 0; di < dets.size(); ++di) {
			const Detection &d = dets[di];
			for (size_t tj = 0; tj < nt; ++tj) {
				if (tracks_[tj].cls != d.cls)
					continue;
				const float i = track_iou(d, tracks_[tj].box());
				if (i >= iou_thresh)
					pairs.push_back({i, (int)di, (int)tj});
			}
		}
		std::sort(pairs.begin(), pairs.end(),
		          [](const Pair &a, const Pair &b) { return a.iou > b.iou; });
		for (const Pair &p : pairs) {
			if (d_matched[p.di] || t_matched[p.tj])
				continue;
			tracks_[p.tj].correct(dets[p.di]);
			d_matched[p.di] = 1;
			t_matched[p.tj] = 1;
		}

		for (size_t tj = 0; tj < nt; ++tj)
			if (!t_matched[tj])
				++tracks_[tj].misses;

		for (size_t di = 0; di < dets.size(); ++di) {
			if (d_matched[di])
				continue;
			Track t;
			t.init(dets[di], r, q_pos, q_vel);
			t.id = next_id_++;
			tracks_.push_back(t);
		}

		tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
		                             [this](const Track &t) {
			                             return t.misses > max_misses;
		                             }),
		              tracks_.end());
	}

	// Smoothed boxes (+ stable ids) for tracks confident enough to draw.
	std::vector<TrackedBox> active() const
	{
		std::vector<TrackedBox> out;
		out.reserve(tracks_.size());
		for (const Track &t : tracks_)
			if (t.hits >= min_hits && t.misses <= show_misses)
				out.push_back({t.box(), t.id, t.cx.v, t.cy.v});
		return out;
	}

private:
	std::vector<Track> tracks_;
	int next_id_ = 1;
};
