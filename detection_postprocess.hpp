// detection_postprocess.hpp  -- self-contained, std-only detection postprocessing.
//
// ONNX output semantics (verified):
//   shape [1, 84, 8400], row-major, CHANNEL-major within the 84x8400 block.
//   element (c, a) = data[c * num_anchors + a]   (a in [0,8400), c in [0,84))
//   c =  0..3  -> cx, cy, w, h   (box center + size, in PIXELS of the 640x640
//                                  network image; NOT normalized)
//   c =  4..83 -> 80 class probabilities (sigmoid already applied by the model)
//   There is NO separate objectness channel: score = max over the 80 class probs.
//
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

struct Detection {
    float x, y, w, h;   // top-left x,y and width,height (see decode() for space)
    int   cls;          // class index [0,80)
    float score;        // confidence = max class probability
};

// ---------------------------------------------------------------------------
// decode
//   out            : pointer to the 84*8400 floats of the ONNX output
//   num_anchors    : 8400
//   num_classes    : 80
//   conf_thresh    : keep detections with max-class-prob >= conf_thresh
//   letterbox_scale: forward scale used when mapping into the network image.
//                    For the obs-detection-overlay STRETCHED 640x640 render path,
//                    pass 1.0f (no letterbox), pad_x=pad_y=0, src_w=src_h=640,
//                    so detections come back in 640-network-space and the
//                    caller scales them to source pixels independently.
//   pad_x, pad_y   : left/top padding added during letterboxing (pixels)
//   src_w, src_h   : clamp bounds for the decoded coords
//
//   Mapping from network-space coord to "src" coord:
//       src_coord = (net_coord - pad) / scale
// ---------------------------------------------------------------------------
// channel_major=true  -> output laid out [1, 4+num_classes, num_anchors]
//                        (Ultralytics YOLOv8 default; element(c,a)=out[c*A+a]).
// channel_major=false -> output laid out [1, num_anchors, 4+num_classes]
//                        (some exports/transposed; element(c,a)=out[a*C+c]).
// num_classes is derived from the model's real output shape by the caller, so
// any class count works (e.g. COCO-80, or a 2-class body/head model).
inline std::vector<Detection> decode(const float* out,
                                     int   num_anchors,
                                     int   num_classes,
                                     float conf_thresh,
                                     float letterbox_scale,
                                     float pad_x,
                                     float pad_y,
                                     int   src_w,
                                     int   src_h,
                                     bool  channel_major = true)
{
    std::vector<Detection> dets;
    dets.reserve(256);

    const int C = 4 + num_classes;  // box(4) + per-class scores
    // Element accessor independent of memory layout.
    auto at = [&](int c, int a) -> float {
        return channel_major
                   ? out[static_cast<size_t>(c) * num_anchors + a]
                   : out[static_cast<size_t>(a) * C + c];
    };

    const float fw  = static_cast<float>(src_w);
    const float fh  = static_cast<float>(src_h);
    const float inv = (letterbox_scale != 0.0f) ? (1.0f / letterbox_scale) : 0.0f;

    for (int a = 0; a < num_anchors; ++a) {
        // --- argmax over the class probabilities for this anchor ---
        float best = at(4, a);
        int   best_c = 0;
        for (int c = 1; c < num_classes; ++c) {
            float v = at(4 + c, a);
            if (v > best) { best = v; best_c = c; }
        }
        if (best < conf_thresh) continue;

        // --- cx,cy,w,h (network pixel space) -> xyxy, un-letterbox, clamp ---
        const float cx = at(0, a), cy = at(1, a), bw = at(2, a), bh = at(3, a);
        float x1 = (cx - bw * 0.5f - pad_x) * inv;
        float y1 = (cy - bh * 0.5f - pad_y) * inv;
        float x2 = (cx + bw * 0.5f - pad_x) * inv;
        float y2 = (cy + bh * 0.5f - pad_y) * inv;

        x1 = std::min(std::max(x1, 0.0f), fw);
        y1 = std::min(std::max(y1, 0.0f), fh);
        x2 = std::min(std::max(x2, 0.0f), fw);
        y2 = std::min(std::max(y2, 0.0f), fh);

        Detection d;
        d.x     = x1;
        d.y     = y1;
        d.w     = x2 - x1;
        d.h     = y2 - y1;
        d.cls   = best_c;
        d.score = best;

        if (d.w > 0.0f && d.h > 0.0f)
            dets.push_back(d);
    }
    return dets;
}

// ---------------------------------------------------------------------------
// nms : per-class greedy non-maximum suppression.
//   Sorts by descending score, keeps the top box, drops same-class boxes whose
//   IoU with it exceeds iou_thresh. IoU = intersection / union.
//   Boxes of different classes never suppress each other.
// ---------------------------------------------------------------------------
inline std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thresh)
{
    std::vector<int> idx(dets.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);

    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return dets[a].score > dets[b].score; });

    std::vector<char> removed(dets.size(), 0);
    std::vector<Detection> keep;
    keep.reserve(dets.size());

    auto iou = [](const Detection& A, const Detection& B) -> float {
        const float ax1 = A.x, ay1 = A.y, ax2 = A.x + A.w, ay2 = A.y + A.h;
        const float bx1 = B.x, by1 = B.y, bx2 = B.x + B.w, by2 = B.y + B.h;

        const float ix1 = std::max(ax1, bx1);
        const float iy1 = std::max(ay1, by1);
        const float ix2 = std::min(ax2, bx2);
        const float iy2 = std::min(ay2, by2);

        const float iw = std::max(0.0f, ix2 - ix1);
        const float ih = std::max(0.0f, iy2 - iy1);
        const float inter = iw * ih;

        const float areaA = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
        const float areaB = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
        const float uni   = areaA + areaB - inter;
        return (uni > 0.0f) ? (inter / uni) : 0.0f;
    };

    for (size_t i = 0; i < idx.size(); ++i) {
        const int a = idx[i];
        if (removed[a]) continue;
        keep.push_back(dets[a]);
        for (size_t j = i + 1; j < idx.size(); ++j) {
            const int b = idx[j];
            if (removed[b]) continue;
            if (dets[b].cls != dets[a].cls) continue;   // per-class
            if (iou(dets[a], dets[b]) > iou_thresh)
                removed[b] = 1;
        }
    }
    return keep;
}

// ---------------------------------------------------------------------------
// soft_nms : per-class LINEAR Soft-NMS.
//   Greedy hard-NMS deletes any same-class box overlapping the winner, so two
//   distinct targets standing close together (high mutual IoU) collapse into a
//   single box. Soft-NMS instead DECAYS the overlapping box's score by
//   (1 - IoU): a true duplicate (IoU ~ 0.9) decays below score_thresh and drops
//   out, while two genuinely separate-but-close objects (moderate IoU) keep
//   enough score to both survive. Boxes of different classes never interact.
//     iou_thresh   : only boxes overlapping the winner by more than this decay
//     score_thresh : drop a box once its (decayed) score falls below this
// ---------------------------------------------------------------------------
inline std::vector<Detection> soft_nms(std::vector<Detection> dets,
                                       float iou_thresh,
                                       float score_thresh)
{
    auto iou = [](const Detection &A, const Detection &B) -> float {
        const float ax1 = A.x, ay1 = A.y, ax2 = A.x + A.w, ay2 = A.y + A.h;
        const float bx1 = B.x, by1 = B.y, bx2 = B.x + B.w, by2 = B.y + B.h;
        const float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
        const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
        const float iw = std::max(0.0f, ix2 - ix1);
        const float ih = std::max(0.0f, iy2 - iy1);
        const float inter = iw * ih;
        const float uni = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1) +
                          std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1) -
                          inter;
        return (uni > 0.0f) ? (inter / uni) : 0.0f;
    };

    std::vector<Detection> keep;
    keep.reserve(dets.size());

    while (!dets.empty()) {
        // pull out the current highest-scoring box
        size_t m = 0;
        for (size_t i = 1; i < dets.size(); ++i)
            if (dets[i].score > dets[m].score) m = i;
        const Detection best = dets[m];
        dets[m] = dets.back();
        dets.pop_back();
        keep.push_back(best);

        // decay (and possibly drop) same-class boxes that overlap the winner
        for (size_t i = 0; i < dets.size();) {
            if (dets[i].cls == best.cls) {
                const float ov = iou(best, dets[i]);
                if (ov > iou_thresh)
                    dets[i].score *= (1.0f - ov);  // linear decay
            }
            if (dets[i].score < score_thresh) {
                dets[i] = dets.back();             // drop, swap-with-last
                dets.pop_back();
            } else {
                ++i;
            }
        }
    }
    return keep;
}

// ---------------------------------------------------------------------------
// 80 COCO class names (index order matches the model's class channels).
// ---------------------------------------------------------------------------
static const char* coco_names[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
