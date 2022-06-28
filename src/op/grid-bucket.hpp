#pragma once

#include <mutex>

struct GridBucketBase : Operator<5, 3>
{
  GridBucketBase(BucketMapping const &map, Index const nC, Index const d1)
    : mapping_{map}
    , inputDims_{AddFront(map.cartDims, nC, d1)}
    , outputDims_{AddFront(map.noncartDims, nC)}
    , ws_{std::make_shared<Cx5>(inputDims_)}
    , weightFrames_{true}
  {
  }

  virtual ~GridBucketBase(){};
  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid
  virtual Output A(Input const &cart) const = 0;
  virtual Input &Adj(Output const &noncart) const = 0;

  Sz3 outputDimensions() const override
  {
    return outputDims_;
  }

  Sz5 inputDimensions() const override
  {
    return inputDims_;
  }

  std::shared_ptr<Cx5> workspace() const
  {
    return ws_;
  }

  void doNotWeightFrames()
  {
    weightFrames_ = false;
  }

  BucketMapping const &mapping() const
  {
    return mapping_;
  }

protected:
  BucketMapping mapping_;
  Sz5 inputDims_;
  Sz3 outputDims_;
  std::shared_ptr<Cx5> ws_;
  bool weightFrames_;
};

template <int IP, int TP>
struct SizedBucketGrid : GridBucketBase
{
  SizedBucketGrid(SizedKernel<IP, TP> const *k, BucketMapping const &map, Index const nC, Index const d1)
    : GridBucketBase(map, nC, d1)
    , kernel_{k}
  {
  }

  virtual ~SizedBucketGrid(){};

  R3 apodization(Sz3 const sz) const
  {
    auto gridSz = this->mapping().cartDims;
    Cx3 temp(gridSz);
    auto const fft = FFT::Make<3, 3>(gridSz);
    temp.setZero();
    auto const k = kernel_->k(Point3{0, 0, 0});
    Crop3(temp, k.dimensions()) = k.template cast<Cx>();
    Log::Tensor(temp, "apo-kernel");
    fft->reverse(temp);
    R3 a = Crop3(R3(temp.real()), sz);
    float const scale = sqrt(Product(gridSz));
    Log::Print(FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    Log::Tensor(a, "apo-final");
    return a;
  }

protected:
  SizedKernel<IP, TP> const *kernel_;
};

template <int IP, int TP>
struct GridBucket final : SizedBucketGrid<IP, TP>
{
  using typename SizedBucketGrid<IP, TP>::Input;
  using typename SizedBucketGrid<IP, TP>::Output;

  GridBucket(SizedKernel<IP, TP> const *k, BucketMapping const &mapping, Index const nC)
    : SizedBucketGrid<IP, TP>(k, mapping, nC, mapping.frames)
  {
    Log::Debug(FMT_STRING("Bucket Grid<{},{}>, dims {}"), IP, TP, this->inputDimensions());
  }

  Output A(Input const &cart) const
  {
    if (cart.dimensions() != this->inputDimensions()) {
      Log::Fail(FMT_STRING("Cartesian k-space dims {} did not match {}"), cart.dimensions(), this->inputDimensions());
    }
    Output noncart(this->outputDimensions());
    noncart.setZero();
    auto const &cdims = this->inputDimensions();
    Index const nC = cdims[0];
    Index const nF = cdims[1];
    auto const scale = this->mapping_.scale;

    auto grid_task = [&](Index const ib) {
      auto const &bucket = this->mapping_.buckets[ib];
      auto bSz = bucket.gridSize();

      // Get slice of grid. Adjust for edges
      Cx5 slice(AddFront(bSz, nC, nF));
      slice.setZero();
      auto sz = slice.dimensions();
      sz[2] -= std::max(bucket.maxCorner[0] - cdims[2], 0L);
      sz[3] -= std::max(bucket.maxCorner[1] - cdims[3], 0L);
      sz[4] -= std::max(bucket.maxCorner[2] - cdims[4], 0L);

      Sz5 wsSt{
        0, 0, std::max(bucket.minCorner[0], 0L), std::max(bucket.minCorner[1], 0L), std::max(bucket.minCorner[2], 0L)};
      Sz5 slSt{
        0, 0, -std::min(bucket.minCorner[0], 0L), -std::min(bucket.minCorner[1], 0L), -std::min(bucket.minCorner[2], 0L)};

      slice.slice(slSt, sz) = this->ws_->slice(wsSt, sz);

      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const c = bucket.cart[ii];
        auto const n = bucket.noncart[ii];
        auto const ifr = bucket.frame[ii];
        auto const k = this->kernel_->k(bucket.offset[ii]);

        Index const stX = c.x - ((IP - 1) / 2) - bucket.minCorner[0];
        Index const stY = c.y - ((IP - 1) / 2) - bucket.minCorner[1];
        Index const stZ = c.z - ((TP - 1) / 2) - bucket.minCorner[2];
        Cx1 sum(nC);
        sum.setZero();
        for (Index iz = 0; iz < TP; iz++) {
          for (Index iy = 0; iy < IP; iy++) {
            for (Index ix = 0; ix < IP; ix++) {
              float const kval = k(ix, iy, iz) * scale;
              for (Index ic = 0; ic < nC; ic++) {
                sum(ic) += slice(ic, ifr, stX + ix, stY + iy, stZ + iz) * kval;
              }
            }
          }
        }
        noncart.chip(n.spoke, 2).chip(n.read, 1) = sum;
      }
    };

    Threads::For(grid_task, this->mapping_.size(), "Bucket Grid Forward");
    return noncart;
  }

  Input &Adj(Output const &noncart) const
  {
    Log::Debug("Grid Adjoint");
    if (noncart.dimensions() != this->outputDimensions()) {
      Log::Fail(
        FMT_STRING("Noncartesian k-space dims {} did not match {}"), noncart.dimensions(), this->outputDimensions());
    }
    auto const &cdims = this->inputDimensions();
    Index const nC = cdims[0];
    Index const nF = cdims[1];

    std::mutex writeMutex;
    auto grid_task = [&](Index ib) {
      auto const &bucket = this->mapping_.buckets[ib];
      auto const bSz = bucket.gridSize();
      Cx5 out(AddFront(bSz, nC, nF));
      out.setZero();

      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const si = bucket.sortedIndices[ii];
        auto const c = bucket.cart[si];
        auto const n = bucket.noncart[si];
        auto const k = this->kernel_->k(bucket.offset[si]);
        auto const ifr = bucket.frame[si];
        auto const scale = this->mapping_.scale * (this->weightFrames_ ? this->mapping_.frameWeights[ifr] : 1.f);

        Index const stX = c.x - ((IP - 1) / 2) - bucket.minCorner[0];
        Index const stY = c.y - ((IP - 1) / 2) - bucket.minCorner[1];
        Index const stZ = c.z - ((TP - 1) / 2) - bucket.minCorner[2];
        Cx1 const sample = noncart.chip(n.spoke, 2).chip(n.read, 1);
        for (Index iz = 0; iz < TP; iz++) {
          for (Index iy = 0; iy < IP; iy++) {
            for (Index ix = 0; ix < IP; ix++) {
              float const kval = k(ix, iy, iz) * scale;
              for (Index ic = 0; ic < nC; ic++) {
                out(ic, ifr, stX + ix, stY + iy, stZ + iz) += sample(ic) * kval;
              }
            }
          }
        }
      }

      auto sz = out.dimensions();
      // Adjust for edge of grid
      sz[2] -= -std::min(bucket.minCorner[0], 0L);
      sz[3] -= -std::min(bucket.minCorner[1], 0L);
      sz[4] -= -std::min(bucket.minCorner[2], 0L);

      sz[2] -= std::max(bucket.maxCorner[0] - cdims[2], 0L);
      sz[3] -= std::max(bucket.maxCorner[1] - cdims[3], 0L);
      sz[4] -= std::max(bucket.maxCorner[2] - cdims[4], 0L);

      Sz5 wsSt{
        0, 0, std::max(bucket.minCorner[0], 0L), std::max(bucket.minCorner[1], 0L), std::max(bucket.minCorner[2], 0L)};
      Sz5 outSt{
        0, 0, -std::min(bucket.minCorner[0], 0L), -std::min(bucket.minCorner[1], 0L), -std::min(bucket.minCorner[2], 0L)};

      {
        std::scoped_lock lock(writeMutex);
        // this->ws_->slice(wsSt, sz) += out.slice(outSt, sz);
        for (Index iz = 0; iz < sz[4]; iz++) {
          for (Index iy = 0; iy < sz[3]; iy++) {
            for (Index ix = 0; ix < sz[2]; ix++) {
              for (Index ifr = 0; ifr < sz[1]; ifr++) {
                for (Index ic = 0; ic < nC; ic++) {
                  this->ws_->operator()(ic, ifr, wsSt[2] + ix, wsSt[3] + iy, wsSt[4] + iz) += out(ic, ifr, outSt[2] + ix, outSt[3] + iy, outSt[4] + iz);
                }
              }
            }
          }
        }
      }
    };

    Threads::For(grid_task, this->mapping_.size(), "Bucket Grid Adjoint");
    return *(this->ws_);
  }

private:
  using FixIn = Eigen::type2index<IP>;
  using FixThrough = Eigen::type2index<TP>;
};
