#ifndef CONSENSUS_H
#define CONSENSUS_H

#include <vector>
#include <map>

struct NodeInfo {
  double d, d_av, y, k, c;
};

class Node {
public:
  int index;
  std::map<int, NodeInfo> node_info;
  double n, m, o, L;
  double rho;

  void set_node();

  void initialization(const std::map<int, double>& coupling_gains, int LUMINAIRE);
  bool check_feasibility(const std::map<int, double>& d) const;
  double evaluate_cost(const std::map<int, double>& d) const; 
  Node& consensus_iterate();

  void set_occupancy(int occupancy);  
  int get_occupancy();
  void set_lower_bound_occupied(double lower_bound_occupied);
  double get_lower_bound_occupied();
  void set_lower_bound_unoccupied(double lower_bound_unoccupied);
  double get_lower_bound_unoccupied();
  void set_lower_bound();
  double get_lower_bound();
  void set_cost(double cost);
  double get_cost();

private:
  int _occupancy;
  double _lower_bound_occupied, _lower_bound_unoccupied, _lower_bound, _cost;
};

#endif