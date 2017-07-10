#include "aux.hpp"
#include "covariate.hpp"

std::istream &operator>>(std::istream &is, Covariate::Kind &kind) {
  std::string token;
  is >> token;
  token = to_lower(token);
  if (token == "gene")
    kind = Covariate::Kind::Gene;
  else if (token == "type")
    kind = Covariate::Kind::Type;
  else if (token == "Section")
    kind = Covariate::Kind::Section;
  else if (token == "Spot")
    kind = Covariate::Kind::Spot;
  else
    kind = Covariate::Kind::Custom;
  return is;
};

std::string to_string(const Covariate::Kind &kind) {
  switch (kind) {
    case Covariate::Kind::Gene:
      return "Gene";
    case Covariate::Kind::Type:
      return "Type";
    case Covariate::Kind::Section:
      return "Section";
    case Covariate::Kind::Spot:
      return "Spot";
    case Covariate::Kind::Custom:
      return "Custom";
  }
  return "Custom";
}

std::ostream &operator<<(std::ostream &os, const Covariate::Kind &kind) {
  os << to_string(kind);
  return os;
}
