/*~--------------------------------------------------------------------------~*
 * Copyright (c) 2017 Los Alamos National Security, LLC
 * All rights reserved.
 *~--------------------------------------------------------------------------~*/

#ifndef lax_wendroff_driver_h
#define lax_wendroff_driver_h

#include <iostream>

#include "flecsi/execution/execution.h"

///
// \file driver.h
// \authors jgraham
// \date Initial file creation: Feb 2, 2017
///

#define NX 32
#define NY 32
#define CFL 0.5
#define U 1.0
#define V 1.0
#define DX (1.0  / static_cast<double>(NX - 1))
#define DY (1.0  / static_cast<double>(NY - 1))
#define DT std::min(CFL * DX / U, CFL * DY / V)
#define X_ADVECTION U * DT / DX
#define Y_ADVECTION V * DT / DX

template<typename T>
using accessor_t = flecsi::data::legion::dense_accessor_t<T, flecsi::data::legion_meta_data_t<flecsi::default_user_meta_data_t> >;

namespace flecsi {
namespace execution {

void initialize_data(accessor_t<size_t> global_IDs,
        accessor_t<double> acc_cells);
flecsi_register_task(initialize_data, loc, single);

void calculate_exclusive_x_update (accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        accessor_t<double> phi_update);
flecsi_register_task(calculate_exclusive_x_update, loc, single);

void calculate_exclusive_y_update (accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        accessor_t<double> phi_update);
flecsi_register_task(calculate_exclusive_y_update, loc, single);

void advect_owned_cells_in_x (accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        accessor_t<double> phi_update);
flecsi_register_task(advect_owned_cells_in_x, loc, single);

void advect_owned_cells_in_y (accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        accessor_t<double> phi_update);
flecsi_register_task(advect_owned_cells_in_y, loc, single);

void write_to_disk (accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        size_t my_color);
flecsi_register_task(write_to_disk , loc, single);

void
driver(
  int argc,
  char ** argv
)
{
  flecsi::execution::context_t & context_ = flecsi::execution::context_t::instance();
  const LegionRuntime::HighLevel::Task *task = context_.task(flecsi::utils::const_string_t{"driver"}.hash());
  const size_t my_color = task->index_point.point_data[0];

  flecsi::data_client& client = *((flecsi::data_client*)argv[argc - 1]);

  size_t index_space = 0;

  auto write_exclusive_shared =
    flecsi_get_handle(client, lax, phi, double, dense, index_space, rw, rw, none);
  auto rw_excl_shrd_ro_ghost =
    flecsi_get_handle(client, lax, phi, double, dense, index_space, rw, rw, ro);
  auto cell_IDs =
   flecsi_get_handle(client, lax, cell_ID, size_t, dense, index_space, ro, ro, ro);
  auto read_exclusive_shared =
   flecsi_get_handle(client, lax, phi, double, dense, index_space, ro, ro, none);
  auto write_exclusive_update =
   flecsi_get_handle(client, lax, phi_update, double, dense, index_space, rw, none, none);
  auto write_shared_update =
   flecsi_get_handle(client, lax, phi_update, double, dense, index_space, ro, rw, none);

  flecsi_execute_task(initialize_data, loc, single, cell_IDs, write_exclusive_shared);

  double time = 0.0;
  while (time < 0.165) {
    time += DT;

    if (my_color == 0)
        std::cout << "t=" << time << std::endl;

    flecsi_execute_task(calculate_exclusive_x_update, loc, single, cell_IDs, read_exclusive_shared,
            write_exclusive_update);

    flecsi_execute_task(advect_owned_cells_in_x, loc, single, cell_IDs, rw_excl_shrd_ro_ghost,
            write_shared_update);

    flecsi_execute_task(calculate_exclusive_y_update, loc, single, cell_IDs, read_exclusive_shared,
            write_exclusive_update);

    flecsi_execute_task(advect_owned_cells_in_y, loc, single, cell_IDs, rw_excl_shrd_ro_ghost,
            write_shared_update);

  }

  if (my_color == 0)
      std::cout << "time " << time << std::endl;

  flecsi_execute_task(write_to_disk, loc, single, cell_IDs, read_exclusive_shared, my_color);

  if (my_color == 0)
        std::cout << "lax wendroff ... all tasks issued" << std::endl;
} //driver



static void create_gid_to_index_maps (accessor_t<size_t>& global_IDs,
        std::map<size_t, size_t>* map_gid_to_exclusive_index,
        std::map<size_t, size_t>* map_gid_to_shared_index,
        std::map<size_t, size_t>* map_gid_to_ghost_index);

static void calc_x_indices(const size_t gid_pt,
          size_t* gid_plus_i, size_t* gid_minus_i);

static void calc_y_indices(const size_t gid_pt,
          size_t* gid_plus_j, size_t* gid_minus_j);

static double initial_value(const size_t pt);


void
initialize_data(
        accessor_t<size_t> global_IDs,
        accessor_t<double> acc_cells
)
{

    for (size_t i = 0; i < acc_cells.size(); i++)
        acc_cells[i] = initial_value(global_IDs[i]);

    for (size_t i = 0; i < acc_cells.shared_size(); i++)
        acc_cells.shared(i) = initial_value(global_IDs.shared(i));

}

void
calculate_exclusive_x_update (
        accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        accessor_t<double> phi_update
)
{
    const double adv = X_ADVECTION;

    std::map<size_t, size_t> map_gid_to_excl_index;
    std::map<size_t, size_t> map_gid_to_shared_index;
    std::map<size_t, size_t> map_gid_to_ghost_index;

    create_gid_to_index_maps(global_IDs, &map_gid_to_excl_index, &map_gid_to_shared_index, &map_gid_to_ghost_index);

    for (size_t index = 0; index < phi.size(); index++) {
        size_t gid_plus_i, gid_minus_i;
        calc_x_indices(global_IDs[index], &gid_plus_i, &gid_minus_i);

        double update = -adv * adv * phi[index];

        if (map_gid_to_excl_index.find(gid_plus_i) != map_gid_to_excl_index.end())
                update += 0.5 * (adv * adv - adv) * phi[map_gid_to_excl_index.find(gid_plus_i)->second];
        else if (map_gid_to_shared_index.find(gid_plus_i) != map_gid_to_shared_index.end())
            update += 0.5 * (adv * adv - adv) * phi.shared(map_gid_to_shared_index.find(gid_plus_i)->second);

        if (map_gid_to_excl_index.find(gid_minus_i) != map_gid_to_excl_index.end())
            update += 0.5 * (adv * adv + adv) * phi[map_gid_to_excl_index.find(gid_minus_i)->second];
        else if (map_gid_to_shared_index.find(gid_minus_i) != map_gid_to_shared_index.end())
            update += 0.5 * (adv * adv + adv) * phi.shared(map_gid_to_shared_index.find(gid_minus_i)->second);

        phi_update[index] = update;
    }

}

void
calculate_exclusive_y_update (
        accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        accessor_t<double> phi_update
)
{
    const double adv = Y_ADVECTION;

    std::map<size_t, size_t> map_gid_to_excl_index;
    std::map<size_t, size_t> map_gid_to_shared_index;
    std::map<size_t, size_t> map_gid_to_ghost_index;

    create_gid_to_index_maps(global_IDs, &map_gid_to_excl_index, &map_gid_to_shared_index, &map_gid_to_ghost_index);

    for (size_t index = 0; index < phi.size(); index++) {
        size_t gid_plus_j, gid_minus_j;
        calc_y_indices(global_IDs[index], &gid_plus_j, &gid_minus_j);

        double update = -adv * adv * phi[index];

        if (map_gid_to_excl_index.find(gid_plus_j) != map_gid_to_excl_index.end())
                update += 0.5 * (adv * adv - adv) * phi[map_gid_to_excl_index.find(gid_plus_j)->second];
        else if (map_gid_to_shared_index.find(gid_plus_j) != map_gid_to_shared_index.end())
            update += 0.5 * (adv * adv - adv) * phi.shared(map_gid_to_shared_index.find(gid_plus_j)->second);

        if (map_gid_to_excl_index.find(gid_minus_j) != map_gid_to_excl_index.end())
            update += 0.5 * (adv * adv + adv) * phi[map_gid_to_excl_index.find(gid_minus_j)->second];
        else if (map_gid_to_shared_index.find(gid_minus_j) != map_gid_to_shared_index.end())
            update += 0.5 * (adv * adv + adv) * phi.shared(map_gid_to_shared_index.find(gid_minus_j)->second);

        phi_update[index] = update;
    }
}

void
advect_owned_cells_in_x (
        accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        accessor_t<double> phi_update
)
{
    const double adv = X_ADVECTION;

    std::map<size_t, size_t> map_gid_to_excl_index;
    std::map<size_t, size_t> map_gid_to_shared_index;
    std::map<size_t, size_t> map_gid_to_ghost_index;

    create_gid_to_index_maps(global_IDs, &map_gid_to_excl_index, &map_gid_to_shared_index, &map_gid_to_ghost_index);

    for (size_t index = 0; index < phi.shared_size(); index++) {
        size_t gid_plus_i, gid_minus_i;
        calc_x_indices(global_IDs.shared(index), &gid_plus_i, &gid_minus_i);

        double update = -adv * adv * phi.shared(index);

        if (map_gid_to_shared_index.find(gid_plus_i) != map_gid_to_shared_index.end())
            update += 0.5 * (adv * adv - adv) * phi.shared(map_gid_to_shared_index.find(gid_plus_i)->second);
        else if (map_gid_to_ghost_index.find(gid_plus_i) != map_gid_to_ghost_index.end())
            update += 0.5 * (adv * adv - adv) * phi.ghost(map_gid_to_ghost_index.find(gid_plus_i)->second);
        else if (map_gid_to_excl_index.find(gid_plus_i) != map_gid_to_excl_index.end())
            update += 0.5 * (adv * adv - adv) * phi[map_gid_to_excl_index.find(gid_plus_i)->second];

        if (map_gid_to_shared_index.find(gid_minus_i) != map_gid_to_shared_index.end())
            update += 0.5 * (adv * adv + adv) * phi.shared(map_gid_to_shared_index.find(gid_minus_i)->second);
        else if (map_gid_to_ghost_index.find(gid_minus_i) != map_gid_to_ghost_index.end())
            update += 0.5 * (adv * adv + adv) * phi.ghost(map_gid_to_ghost_index.find(gid_minus_i)->second);
        else if (map_gid_to_excl_index.find(gid_minus_i) != map_gid_to_excl_index.end())
            update += 0.5 * (adv * adv + adv) * phi[map_gid_to_excl_index.find(gid_minus_i)->second];

        phi_update.shared(index) = update;
    }

    for (size_t index = 0; index < phi.size(); index++)
        phi[index] += phi_update[index];

    for (size_t index = 0; index < phi.shared_size(); index++)
        phi.shared(index) += phi_update.shared(index);

}

void
advect_owned_cells_in_y (
        accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        accessor_t<double> phi_update
)
{
    const double adv = Y_ADVECTION;

    std::map<size_t, size_t> map_gid_to_excl_index;
    std::map<size_t, size_t> map_gid_to_shared_index;
    std::map<size_t, size_t> map_gid_to_ghost_index;

    create_gid_to_index_maps(global_IDs, &map_gid_to_excl_index, &map_gid_to_shared_index, &map_gid_to_ghost_index);

    for (size_t index = 0; index < phi.shared_size(); index++) {
        size_t gid_plus_j, gid_minus_j;
        calc_y_indices(global_IDs.shared(index), &gid_plus_j, &gid_minus_j);

        double update = -adv * adv * phi.shared(index);

        if (map_gid_to_shared_index.find(gid_plus_j) != map_gid_to_shared_index.end())
            update += 0.5 * (adv * adv - adv) * phi.shared(map_gid_to_shared_index.find(gid_plus_j)->second);
        else if (map_gid_to_ghost_index.find(gid_plus_j) != map_gid_to_ghost_index.end())
            update += 0.5 * (adv * adv - adv) * phi.ghost(map_gid_to_ghost_index.find(gid_plus_j)->second);
        else if (map_gid_to_excl_index.find(gid_plus_j) != map_gid_to_excl_index.end())
            update += 0.5 * (adv * adv - adv) * phi[map_gid_to_excl_index.find(gid_plus_j)->second];

        if (map_gid_to_shared_index.find(gid_minus_j) != map_gid_to_shared_index.end())
            update += 0.5 * (adv * adv + adv) * phi.shared(map_gid_to_shared_index.find(gid_minus_j)->second);
        else if (map_gid_to_ghost_index.find(gid_minus_j) != map_gid_to_ghost_index.end())
            update += 0.5 * (adv * adv + adv) * phi.ghost(map_gid_to_ghost_index.find(gid_minus_j)->second);
        else if (map_gid_to_excl_index.find(gid_minus_j) != map_gid_to_excl_index.end())
            update += 0.5 * (adv * adv + adv) * phi[map_gid_to_excl_index.find(gid_minus_j)->second];

        phi_update.shared(index) = update;
    }

    for (size_t index = 0; index < phi.size(); index++)
        phi[index] += phi_update[index];

    for (size_t index = 0; index < phi.shared_size(); index++)
        phi.shared(index) += phi_update.shared(index);

}

void
write_to_disk (
        accessor_t<size_t> global_IDs,
        accessor_t<double> phi,
        size_t my_color
)
{
  char buf[40];
  sprintf(buf,"lax%d.part", my_color);
  std::ofstream myfile;
  myfile.open(buf);

  for (size_t i = 0; i < phi.size(); i++) {
      const size_t pt = global_IDs[i];
      const size_t y_index = pt / NX;
      const size_t x_index = pt % NX;
      myfile << x_index << " " << y_index << " " << phi[i] << std::endl;
  }

  for (size_t i = 0; i < phi.shared_size(); i++) {
      const size_t pt = global_IDs.shared(i);
      const size_t y_index = pt / NX;
      const size_t x_index = pt % NX;
      myfile << x_index << " " << y_index << " " << phi.shared(i) << std::endl;
  }

    myfile.close();
}


static void create_gid_to_index_maps (accessor_t<size_t>& global_IDs,
        std::map<size_t, size_t>* map_gid_to_exclusive_index,
        std::map<size_t, size_t>* map_gid_to_shared_index,
        std::map<size_t, size_t>* map_gid_to_ghost_index)
{
    // TODO profile effects of this indirection

    for (size_t index = 0; index < global_IDs.size(); index++)
        (*map_gid_to_exclusive_index)[global_IDs[index]] = index;

    for (size_t index = 0; index < global_IDs.shared_size(); index++)
        (*map_gid_to_shared_index)[global_IDs.shared(index)] = index;

    for (size_t index = 0; index < global_IDs.ghost_size(); index++)
        (*map_gid_to_ghost_index)[global_IDs.ghost(index)] = index;

}

static void calc_x_indices(const size_t gid_pt,
          size_t* gid_plus_i, size_t* gid_minus_i)
{
  const size_t gid_y_index = gid_pt / NX;
  const size_t gid_x_index = gid_pt % NX;
  *gid_plus_i = (gid_x_index + 1) != NX ? gid_x_index + 1 + gid_y_index * NX: -1;
  *gid_minus_i = gid_x_index != 0 ? gid_x_index - 1 + gid_y_index * NX: -1;
}

static void calc_y_indices(const size_t gid_pt,
          size_t* gid_plus_j, size_t* gid_minus_j)
{
  const size_t gid_y_index = gid_pt / NX;
  const size_t gid_x_index = gid_pt % NX;
  *gid_plus_j = (gid_y_index + 1) != NY ? gid_x_index + (1 + gid_y_index) * NX: -1;
  *gid_minus_j = gid_y_index != 0 ? gid_x_index + (gid_y_index - 1) * NX: -1;
}

static double initial_value(const size_t pt) {
    double value = 0.0;
    const size_t y_index = pt / NX;
    const size_t x_index = pt % NX;
    double x = static_cast<double>(x_index) / static_cast<double>(NX - 1);
    double y = static_cast<double>(y_index) / static_cast<double>(NY - 1);
    if ( (x <= 0.5) && (y <= 0.5) )
        value = 1.0;
    return value;
}

} // namespace execution
} // namespace flecsi

#endif // lax_wendroff_driver_h

/*~-------------------------------------------------------------------------~-*
 * Formatting options for vim.
 * vim: set tabstop=4 shiftwidth=4 expandtab :
 *~-------------------------------------------------------------------------~-*/
