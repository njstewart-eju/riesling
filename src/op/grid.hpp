#pragma once

#include "grid-base.hpp"
#include <mutex>

template <int IP, int TP>
struct Grid final : SizedGrid<IP, TP>
{
  using typename SizedGrid<IP, TP>::Input;
  using typename SizedGrid<IP, TP>::Output;

  Grid(SizedKernel<IP, TP> const *k, Mapping const &mapping, Index const nC)
    : SizedGrid<IP, TP>(k, mapping, nC, mapping.frames)
  {
    Log::Debug(FMT_STRING("Grid<{},{}>, dims {}"), IP, TP, this->inputDimensions());
  }

  Grid(SizedKernel<IP, TP> const *k, Mapping const &mapping, Index const nC, R2 const basis)
    : SizedGrid<IP, TP>(k, mapping, nC, basis.dimension(1))
    , basis_{basis}
  {
    Log::Debug(FMT_STRING("Grid<{},{}>, dims {}"), IP, TP, this->inputDimensions());
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
    Index const nB = cdims[1];
    auto const scale = this->mapping_.scale;

    auto grid_task = [&](Index const ii) {
      auto const c = this->mapping_.cart[ii];
      auto const n = this->mapping_.noncart[ii];
      auto const ifr = this->mapping_.frame[ii];
      auto const k = this->kernel_->k(this->mapping_.offset[ii]);

      Index const stX = c.x - ((IP - 1) / 2);
      Index const stY = c.y - ((IP - 1) / 2);
      Index const stZ = c.z - ((TP - 1) / 2);
      Index const btp = basis_.size() ? n.spoke % basis_.dimension(0) : 0;
      Cx1 sum(nC);
      sum.setZero();
      for (Index iz = 0; iz < TP; iz++) {
        Index const iiz = stZ + iz;
        if ((iiz > -1) && (iiz < cdims[4])) {
          for (Index iy = 0; iy < IP; iy++) {
            Index const iiy = stY + iy;
            if ((iiy > -1) && (iiy < cdims[3])) {
              for (Index ix = 0; ix < IP; ix++) {
                Index const iix = stX + ix;
                if ((iix > -1) && (iix < cdims[2])) {
                  float const kval = k(ix, iy, iz) * scale;
                  if (basis_.size()) {
                    for (Index ib = 0; ib < nB; ib++) {
                      float const bval = basis_(btp, ib) * kval;
                      for (Index ic = 0; ic < nC; ic++) {
                        sum(ic) += cart(ic, ib, iix, iiy, iiz) * bval;
                      }
                    }
                  } else {
                    for (Index ic = 0; ic < nC; ic++) {
                      sum(ic) += cart(ic, ifr, iix, iiy, iiz) * kval;
                    }
                  }
                }
              }
            }
          }
        }
      }
      noncart.chip(n.spoke, 2).chip(n.read, 1) = sum;
    };

    Threads::For(grid_task, this->mapping_.cart.size(), "Grid Forward");
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
    Index const nB = cdims[1];

    std::mutex writeMutex;
    auto grid_task = [&](Index ibucket) {
      auto const &bucket = this->mapping_.buckets[ibucket];
      auto const bSz = bucket.gridSize();
      Cx5 out(AddFront(bSz, nC, nB));
      out.setZero();

      for (auto ii = 0; ii < bucket.size(); ii++) {
        auto const si = bucket.indices[ii];
        auto const c = this->mapping_.cart[si];
        auto const n = this->mapping_.noncart[si];
        auto const k = this->kernel_->k(this->mapping_.offset[si]);
        auto const ifr = this->mapping_.frame[si];
        auto const scale = this->mapping_.scale * (this->weightFrames_ ? this->mapping_.frameWeights[ifr] : 1.f);

        Index const stX = c.x - ((IP - 1) / 2) - bucket.minCorner[0];
        Index const stY = c.y - ((IP - 1) / 2) - bucket.minCorner[1];
        Index const stZ = c.z - ((TP - 1) / 2) - bucket.minCorner[2];
        Index const btp = basis_.size() ? n.spoke % basis_.dimension(0) : 0;
        Cx1 const sample = noncart.chip(n.spoke, 2).chip(n.read, 1);
        for (Index iz = 0; iz < TP; iz++) {
          Index const iiz = stZ + iz;
          for (Index iy = 0; iy < IP; iy++) {
            Index const iiy = stY + iy;
            for (Index ix = 0; ix < IP; ix++) {
              Index const iix = stX + ix;
              float const kval = k(ix, iy, iz) * scale;
              if (basis_.size() > 0) {
                for (Index ib = 0; ib < nB; ib++) {
                  float const bval = kval * basis_(btp, ib);
                  for (Index ic = 0; ic < nC; ic++) {
                    out(ic, ib, iix, iiy, iiz) += noncart(ic, n.read, n.spoke) * bval;
                  }
                }
              } else {
                for (Index ic = 0; ic < nC; ic++) {
                  out(ic, ifr, iix, iiy, iiz) += sample(ic) * kval;
                }
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
        0,
        0,
        -std::min(bucket.minCorner[0], 0L),
        -std::min(bucket.minCorner[1], 0L),
        -std::min(bucket.minCorner[2], 0L)};

      {
        std::scoped_lock lock(writeMutex);
        for (Index iz = 0; iz < sz[4]; iz++) {
          for (Index iy = 0; iy < sz[3]; iy++) {
            for (Index ix = 0; ix < sz[2]; ix++) {
              for (Index ifr = 0; ifr < sz[1]; ifr++) {
                for (Index ic = 0; ic < nC; ic++) {
                  this->ws_->operator()(ic, ifr, wsSt[2] + ix, wsSt[3] + iy, wsSt[4] + iz) +=
                    out(ic, ifr, outSt[2] + ix, outSt[3] + iy, outSt[4] + iz);
                }
              }
            }
          }
        }
      }
    };

    this->ws_->setZero();
    Threads::For(grid_task, this->mapping_.buckets.size(), "Grid Adjoint");
    return *(this->ws_);
  }

private:
  using FixIn = Eigen::type2index<IP>;
  using FixThrough = Eigen::type2index<TP>;

  R2 basis_;
};
