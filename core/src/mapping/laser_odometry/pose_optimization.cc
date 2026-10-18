/**
 * @file pose_optimization.cc
 * @author hang chen (chen@hang.plus)
 * @brief
 * @version 0.1
 * @date 2026-07-26
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "mapping/laser_odometry/pose_optimization.h"

#include <algorithm>
#include <utility>

#include "mapping/grid_mapping/probability_values.h"

namespace evergreenslam::mapping {

PoseOptimization::PoseOptimization(const PoseOptimizationOption& option)
    : option_(option),
      correlative_matcher_(option.linear_search_window, option.angular_search_window),
      ceres_matcher_(option.grid_match_weight) {
  if (option.translation_weight > 0.0 && option.rotation_weight > 0.0) {
    ceres_matcher_.SetPosePrior(option.translation_weight, option.rotation_weight);
    correlative_matcher_.SetMotionPenalty(option.translation_weight, option.rotation_weight);
  }
}

double PoseOptimization::Match(const sensor::PointCloud& point_cloud, const GridMapu8& grid_map,
                               Eigen::Affine2d& pose, Eigen::Affine2d* coarse_pose) {
  if (point_cloud.empty()) {
    return 0.0;
  }

  // Unknown space is a neutral prior for local alignment, not evidence of free space.
  // Penalizing it locks sparse axis-aligned maps onto the repeating beam pattern.
  auto cells = grid_map.data();
  const uint8_t unknown = ProbabilityToValue(0.5);
  std::replace(cells.begin(), cells.end(), grid_map.unknown_value(), unknown);
  const GridMapu8 matching_map(std::move(cells), grid_map.width(), grid_map.height(),
                             grid_map.resolution(), grid_map.origin_x(), grid_map.origin_y(),
                             unknown);
  const Eigen::Affine2d predicted_pose = pose;
  correlative_matcher_.Match(point_cloud, matching_map, pose);
  double score = 0.0;
  for (const auto& point : point_cloud) {
    score += ValueToProbability(grid_map.GetValueAtPoint(pose * point.point));
  }
  score /= point_cloud.size();
  if (coarse_pose != nullptr) {
    *coarse_pose = pose;
  }
  ceres_matcher_.Match(point_cloud, matching_map, pose, predicted_pose);
  return score;
}

}  // namespace evergreenslam::mapping
