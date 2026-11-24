// Offline diagnostic adapter. Reuse the production C++ value/resource functions
// on saved PUBLIC views; there is no simulator, hidden RNG or policy mutation.
#include "combat_search/PublicRules.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("public input file required");
        std::ifstream file(argv[1]);spire_search::J input;file>>input;
        spire_search::J rows=spire_search::J::array();
        const auto reserve=spire_search::inventoryReserve(input.at("root"));
        for(const auto& item:input.at("views")) {
            const auto& view=item.at("public");int terminal=item.at("outcome");
            const int hp=view.at("player").at("current_hp");
            rows.push_back({{"hp",hp},{"outcome",terminal},
                {"value",spire_search::combatValue(view,terminal)},
                {"resource_hp",hp+spire_search::inventoryReserve(view)-reserve}});
        }
        std::cout<<rows.dump()<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
