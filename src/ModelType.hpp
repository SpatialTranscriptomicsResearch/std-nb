#ifndef MODELTYPE_HPP
#define MODELTYPE_HPP

#include "PartialModel.hpp"

namespace PoissonFactorization {
template <Partial::Kind feat_kind = Partial::Kind::Gamma,
          Partial::Kind mix_kind = Partial::Kind::HierGamma>
struct ModelType {
  using features_t = Partial::Model<Partial::Variable::Feature, feat_kind>;
  using weights_t = Partial::Model<Partial::Variable::Mix, mix_kind>;
};
}

#endif
