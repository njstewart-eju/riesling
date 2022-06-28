#pragma once

#include "info.h"
#include "log.h"
#include "types.h"
#include "kernel.h"

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
  std::vector<CartesianIndex> cart;
  std::vector<Point3> offset;
  std::vector<NoncartesianIndex> noncart;
  std::vector<int8_t> frame;

  bool empty() const
  {
    return cart.empty();
  };

  Index const size() const {
    return cart.size();
  }

  Sz3 gridSize() const
  {
    return Sz3{
      maxCorner[0] - minCorner[0],
      maxCorner[1] - minCorner[1],
      maxCorner[2] - minCorner[2]};
  }
};

struct BucketMapping
{
  Info::Type type;
  Sz2 noncartDims;
  Sz3 cartDims;
  int8_t frames;
  Eigen::ArrayXf frameWeights;
  float scale; // Overall scaling from oversampling
  std::vector<Bucket> buckets;

  Index const size() const {
    return buckets.size();
  }
};

struct Mapping
{
  Info::Type type;
  std::vector<CartesianIndex> cart;
  std::vector<NoncartesianIndex> noncart;
  std::vector<int8_t> frame;
  std::vector<Point3> offset;
  std::vector<int32_t> sortedIndices;
  Sz2 noncartDims;
  Sz3 cartDims;
  int8_t frames;
  Eigen::ArrayXf frameWeights;
  float scale; // Overall scaling due to oversampling
};

struct Trajectory
{
  Trajectory();
  Trajectory(Info const &info, R3 const &points);
  Trajectory(Info const &info, R3 const &points, I1 const &frames);
  Info const &info() const;
  R3 const &points() const;
  I1 const &frames() const;
  Point3 point(int16_t const read, int32_t const spoke, float const nomRad) const;
  BucketMapping bucketMapping(Index const bucketSz, Kernel const *k, float const os, Index const read0 = 0) const;
  Mapping mapping(Index const kw, float const os, Index const read0 = 0) const;
  std::tuple<Trajectory, Index> downsample(float const res, Index const lores, bool const shrink) const;

private:
  void init();

  Info info_;
  R3 points_;
  I1 frames_;

  Eigen::ArrayXf mergeHi_, mergeLo_;
};
