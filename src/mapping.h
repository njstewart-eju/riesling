#pragma once

#include "info.h"
#include "kernel.h"
#include "trajectory.h"
#include "types.h"

struct CartesianIndex
{
  int16_t x, y, z;
};

struct NoncartesianIndex
{
  int32_t spoke;
  int16_t read;
};

struct Bucket
{
  Sz3 minCorner, maxCorner;
  std::vector<int32_t> indices;

  bool empty() const
  {
    return indices.empty();
  };

  Index const size() const
  {
    return indices.size();
  }

  Sz3 gridSize() const
  {
    return Sz3{maxCorner[0] - minCorner[0], maxCorner[1] - minCorner[1], maxCorner[2] - minCorner[2]};
  }
};

struct Mapping
{
  Mapping(Trajectory const &t, Kernel const *k, float const osamp, Index const bucketSize = 32, Index const read0 = 0);

  Info::Type type;
  Sz2 noncartDims;
  Sz3 cartDims;
  int8_t frames;
  Eigen::ArrayXf frameWeights;
  float scale; // Overall scaling from oversampling

  std::vector<CartesianIndex> cart;
  std::vector<NoncartesianIndex> noncart;
  std::vector<int8_t> frame;
  std::vector<Point3> offset;
  std::vector<Bucket> buckets;
};