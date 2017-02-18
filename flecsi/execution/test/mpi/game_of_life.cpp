//
// Created by ollie on 2/16/17.
//

#include <cinchtest.h>

#include "flecsi/execution/context.h"

TEST(game_of_live, execution) {

  char dummy[] = "1";
  char * argv = &dummy[0];

  flecsi::execution::context_t::instance().initialize(1, &argv);

}