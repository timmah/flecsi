/*~--------------------------------------------------------------------------~*
 * Copyright (c) 2015 Los Alamos National Security, LLC
 * All rights reserved.
 *~--------------------------------------------------------------------------~*/

#ifndef flexi_mesh_utils_h
#define flexi_mesh_utils_h

#include "flexi/utils/common.h"

/*!
   \file mesh_utils.h
   \authors nickm@lanl.gov, bergen@lanl.gov
   \date Initial file creation: Dec 23, 2015
 */

/*----------------------------------------------------------------------------*
 * debug dump function
 *----------------------------------------------------------------------------*/

#define ndump(X)                                                               \
  std::cout << __FILE__ << ":" << __LINE__ << ": " << __PRETTY_FUNCTION__      \
            << ": " << #X << " = " << X << std::endl

#define nlog(X)                                                                \
  std::cout << __FILE__ << ":" << __LINE__ << ": " << __PRETTY_FUNCTION__      \
            << ": " << X << std::endl

namespace flexi {

/*----------------------------------------------------------------------------*
 * Id utilities.
 *----------------------------------------------------------------------------*/

/*!
  Convert a local id into a global id using a dimension and
  domain.  Currently, the dimension and domain are each given
  2 bits.  In this version, both the dimension and domain are
  given statically as template parameters.

  \tparam D The dimension.
  \tparam M The domain.

  \param local_id The local id from which to form a global id.

  \return The corresponding global id.
 */
template<size_t D, size_t M>
id_t to_global_id(id_t local_id){
  return (id_t(D) << 62) | (id_t(M) << 60) | local_id;
}

/*!
  Convert a local id into a global id using a dimension and
  domain.  Currently, the dimension and domain are each given
  2 bits.  In this version, only the domain is
  given statically as template parameters.

  \tparam M The domain.

  \param dim The dimension.
  \param local_id The local id from which to form a global id.

  \return The corresponding global id.
 */
template<size_t M>
id_t to_global_id(size_t dim, id_t local_id){
  return (id_t(dim) << 62) | (id_t(M) << 60) | local_id;
}

id_t to_local_id(id_t global_id){
  return global_id & 0x0000ffffffffffff;
}

/*----------------------------------------------------------------------------*
 * Tuple search utilities.
 *----------------------------------------------------------------------------*/

/*!
  \struct find_entity__ mesh_utils.h
  \brief find_entity__ provides static search capabilities.
 */
template<size_t I, class T, size_t D, size_t M>
struct find_entity__ {

  /*!
    Find the position index of a type that matches the
    given domain and dimension.

    \tparam I The current index in tuple.
    \tparam T The tuple type.
    \tparam D The dimension to match.
    \tparam M The domain to match.
   */
  static constexpr size_t find() {
    // grab current types
    using E = typename std::tuple_element<I - 1, T>::type;
    using D1 = typename std::tuple_element<0, E>::type;
    using T1 = typename std::tuple_element<1, E>::type;

    // Check match for domain and dimension and return
    // index if matched or recurse if not matched.
    return D1::domain == M && T1::dimension == D ? 
      I : find_entity__<I - 1, T, D, M>::find(); 
  }
};

/*!
  \struct find_entity__ mesh_utils.h
  \brief find_entity__ provides a specialization for the root recursion.
 */
template<class T, size_t D, size_t M>
struct find_entity__<0, T, D, M> {

  /*!
    Search last tuple element.

    \tparam T The tuple type.
    \tparam D The dimension to match.
    \tparam M The domain to match.
   */
  static constexpr size_t find() {
    return 0; 
  } // find

}; // struct find_entity__

/*!
  \struct find_entity_ mesh_utils.h
  \brief find_entity_ provides static search capabilities.

  Top-level interface for recursive type search matching dimension and
  domain.
 */
template<class MT, size_t D, size_t M>
struct find_entity_ {
  using entity_types = typename MT::entity_types;

  using pair_ = 
  typename std::tuple_element<find_entity__<
     std::tuple_size<entity_types>::value, entity_types, D, M>::find() - 1,
     entity_types>::type;

  /*!
    Define the type returned by searching the tuple for matching
    dimension and domain.
   */
  using type = typename std::tuple_element<1, pair_>::type;
};

/*----------------------------------------------------------------------------*
 * Connectivity utilities.
 *----------------------------------------------------------------------------*/

/*!
  \struct compute_connectivity_ mesh_utils.h
  \brief compute_connectivity_ provides static recursion to process
  connectivity computation of mesh entity types.
 */
template<size_t DM, size_t I, class TS>
struct compute_connectivity_ {

  /*!
    Compute mesh connectivity for the given domain and tuple element.

    \tparam DM The domain to match.
    \tparam I The current tuple index.
    \tparam TS The tuple typel
   */
  template<class M>
  static int compute(M & mesh) {
    using T = typename std::tuple_element<I - 1, TS>::type;
    using D1 = typename std::tuple_element<0, T>::type;
    using T1 = typename std::tuple_element<1, T>::type;
    using T2 = typename std::tuple_element<2, T>::type;

    if (D1::domain == DM) {
      mesh.template compute<DM>(T1::dimension, T2::dimension);
    }
    return compute_connectivity_<DM, I - 1, TS>::compute(mesh);
  } // compute

}; // struct compute_connectivity_

/*!
  \struct compute_connectivity_ mesh_utils.h
  \brief compute_connectivity_ provides a specialization for
  the root recursion.
 */
template<size_t DM, class TS>
struct compute_connectivity_<DM, 0, TS> {

  /*!
    Terminate recursion.

    \tparam DM The domain to match.
    \tparam TS The tuple typel
   */
  template<class M>
  static int compute(M &) {
    return 0;
  } // compute

}; // struct compute_connectivity_

/*----------------------------------------------------------------------------*
 * Binding utilities.
 *----------------------------------------------------------------------------*/

/*!
  \struct compute_bindings_ mesh_utils.h
  \brief compute_bindings_ provides static recursion to process
  binding computation of mesh entity types.
 */
template<size_t DM, size_t I, class TS>
struct compute_bindings_ {

  /*!
    Compute mesh connectivity for the given domain and tuple element.

    \tparam M The mesh type.
    \tparam DM The domain to match.
    \tparam I The current tuple index.
    \tparam TS The tuple typel
   */
  template<class M>
  static int compute(M & mesh) {

    // Get the indexed tuple
    using T = typename std::tuple_element<I - 1, TS>::type;

    // Get domains and dimension
    using M1 = typename std::tuple_element<0, T>::type;
    using M2 = typename std::tuple_element<1, T>::type;
    using D = typename std::tuple_element<2, T>::type;

    if (M1::domain == DM) {
      mesh.template compute_bindings<M1::domain, M2::domain, D::value>();
    } // if

    return compute_bindings_<DM, I - 1, TS>::compute(mesh);
  } // compute

}; // struct compute_bindings_

/*!
  \struct compute_bindings_ mesh_utils.h
  \brief compute_bindings_ provides a specialization for
  the root recursion.
 */
template<size_t DM, class TS>
struct compute_bindings_<DM, 0, TS> {

  /*!
    Terminate recursion.

    \tparam DM The domain to match.
    \tparam TS The tuple typel
   */
  template<class M>
  static int compute(M &) {
    return 0;
  } // compute

}; // struct compute_bindings_

} // namespace flexi

#endif // flexi_mesh_utils_h

/*~-------------------------------------------------------------------------~-*
 * Formatting options for vim.
 * vim: set tabstop=2 shiftwidth=2 expandtab :
 *~-------------------------------------------------------------------------~-*/