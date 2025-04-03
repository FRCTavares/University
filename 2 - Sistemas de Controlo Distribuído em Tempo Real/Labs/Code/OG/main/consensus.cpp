#include <numeric>
#include "consensus.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <functional>
#include <iterator>
#include "Arduino.h"
#include <map>

void print_map2(std::map<int, double> m)
{
  Serial.print("{");
  int i = 0;
  for (auto const &pair : m)
  {
    Serial.printf("%d: %lf", pair.first, pair.second);
    i++;
    if (i < m.size())
      Serial.print(", ");
  }
  Serial.print("}");
}

void Node::set_node() {
  rho = 0.07;
  
  _occupancy = 0;
  _lower_bound_occupied = 0, _lower_bound_unoccupied = 0, _cost = 1;
  _lower_bound = _lower_bound_unoccupied;
}

void Node::initialization(const std::map<int, double>& coupling_gains, int LUMINAIRE) {
  this->index = LUMINAIRE;
  for (const auto & item : coupling_gains) {
    if (item.first == -1)
      continue;
    this->node_info[item.first].d = 0;
    this->node_info[item.first].d_av = 0;
    this->node_info[item.first].y = 0;
    this->node_info[item.first].k = item.second / 100;
    this->node_info[item.first].c = 0;
  }
  this->n = std::accumulate(this->node_info.begin(), this->node_info.end(), 0.0, [](double acc, const std::pair<int, NodeInfo>& ni){return acc + ni.second.k * ni.second.k;});
  this->m = this->n - pow(this->node_info[LUMINAIRE].k, 2);
  this->node_info[LUMINAIRE].c = this->_cost;
  this->o = coupling_gains.at(-1);

  double max_lower_bound = this->o;  
  for (const auto & item : this->node_info)
    max_lower_bound += 100 * item.second.k;
  this->L = min(max(this->_lower_bound, this->o), max_lower_bound);
}

bool Node::check_feasibility(const std::map<int, double>& d) const
{
  const double tol = 0.001; // tolerance for rounding errors

  if (d.at(this->index) < -tol || d.at(this->index) > 100 + tol)
    return false;

  double d_dot_k = 0.0;
  for (const auto & item : this->node_info)
    d_dot_k += d.at(item.first) * item.second.k;
  
  if (d_dot_k < this->L - this->o - tol)
    return false;

  return true;
}

double Node::evaluate_cost(const std::map<int, double>& d) const
{
  double cost = 0.0;
  for (const auto & item : this->node_info)
    cost += item.second.c * d.at(item.first) + item.second.y * (d.at(item.first) - item.second.d_av) + this->rho / 2 * pow(d.at(item.first) - item.second.d_av, 2);

  return cost;
}

Node& Node::consensus_iterate()
{
  std::map<int, double> d_best;
  double cost_best = 1000000; // large number
  std::map<int, double> z;
  double z_dot_k = 0.0;

  for (const auto & item : this->node_info) {
    d_best[item.first] = -1;
    z[item.first] = this->rho * item.second.d_av - item.second.y - item.second.c;
    z_dot_k += z[item.first] * item.second.k;
  }

  // unconstrained minimum -> (1 / rho) * z
  std::map<int, double> d_u;
  for (auto & item : z)
    d_u[item.first] = 1 / this->rho * item.second;

  Serial.print("d_u = ");
  print_map2(d_u);
  if (check_feasibility(d_u)) {
    Serial.print(" -> Feasible");
    Serial.printf(" -> cost = %lf", evaluate_cost(d_u));
    if (evaluate_cost(d_u) < cost_best)
      Serial.print(" -> The best\n");
    else
      Serial.print("\n");
  }
  else
    Serial.print(" -> Not Feasible\n");
    
  if (check_feasibility(d_u))
  {
    // If unconstrained solution exists then it's optimal and there's no need to compute the others
    for (auto & item : this->node_info)
      item.second.d = d_u[item.first];
    return *this;
  }

  // compute minimum constrained to linear boundary -> d_bl = (1 / rho) * z - node.k / node.n * (node.o - node.L + (1 / rho) * z' * node.k)
  std::map<int, double> d_bl;
  for (const auto & item : this->node_info)
    d_bl[item.first] = 1 / this->rho * z[item.first] - item.second.k / this->n * (this->o - this->L + 1 / this->rho * z_dot_k);

  Serial.print("d_bl = ");
  print_map2(d_bl);
  if (check_feasibility(d_bl)) {
    Serial.print(" -> Feasible");
    Serial.printf(" -> cost = %lf", evaluate_cost(d_bl));
    if (evaluate_cost(d_bl) < cost_best)
      Serial.print(" -> The best\n");
    else
      Serial.print("\n");
  }
  else
    Serial.print(" -> Not Feasible\n");
    
  if (check_feasibility(d_bl))
  {
    double cost_boundary_linear = evaluate_cost(d_bl);
    if (cost_boundary_linear < cost_best)
    {
      d_best = d_bl;
      cost_best = cost_boundary_linear;
    }
  }

  // compute minimum constrained to 0 boundary -> (1 / rho) * z
  std::map<int, double> d_b0 = d_u;
  d_b0[this->index] = 0;

  Serial.print("d_b0 = ");
  print_map2(d_b0);
  if (check_feasibility(d_b0)) {
    Serial.print(" -> Feasible");
    Serial.printf(" -> cost = %lf", evaluate_cost(d_b0));
    if (evaluate_cost(d_b0) < cost_best)
      Serial.print(" -> The best\n");
    else
      Serial.print("\n");
  }
  else
    Serial.print(" -> Not Feasible\n");

  if (check_feasibility(d_b0))
  {
    double cost_boundary_0 = evaluate_cost(d_b0);
    if (cost_boundary_0 < cost_best)
    {
      d_best = d_b0;
      cost_best = cost_boundary_0;
    }
  }

  // compute minimum constrained to 100 boundary -> (1 / rho) * z
  std::map<int, double> d_b1 = d_u;
  d_b1[this->index] = 100;

  Serial.print("d_b1 = ");
  print_map2(d_b1);
  if (check_feasibility(d_b1)) {
    Serial.print(" -> Feasible");
    Serial.printf(" -> cost = %lf", evaluate_cost(d_b1));
    if (evaluate_cost(d_b1) < cost_best)
      Serial.print(" -> The best\n");
    else
      Serial.print("\n");
  }
  else
    Serial.print(" -> Not Feasible\n");
    
  if (check_feasibility(d_b1))
  {
    double cost_boundary_100 = evaluate_cost(d_b1);
    if (cost_boundary_100 < cost_best)
    {
      d_best = d_b1;
      cost_best = cost_boundary_100;
    }
  }

  // compute minimum constrained to linear and 0 boundary -> (1/rho)*z - (1/node.m)*node.k*(node.o-node.L) + (1/rho/node.m)*node.k*(node.k(node.index)*z(node.index)-z'*node.k);
  std::map<int, double> d_l0;
  for (const auto & item : this->node_info)
    d_l0[item.first] = 1 / this->rho * z[item.first] - 1 / this->m * item.second.k * (this->o - this->L) + 1 / this->rho / this->m * item.second.k * (this->node_info[this->index].k * z[this->index] - z_dot_k);  
  d_l0[this->index] = 0;

  Serial.print("d_l0 = ");
  print_map2(d_l0);
  if (check_feasibility(d_l0)) {
    Serial.print(" -> Feasible");
    Serial.printf(" -> cost = %lf", evaluate_cost(d_l0));
    if (evaluate_cost(d_l0) < cost_best)
      Serial.print(" -> The best\n");
    else
      Serial.print("\n");
  }
  else
    Serial.print(" -> Not Feasible\n");

  if (check_feasibility(d_l0))
  {
    double cost_linear_0 = evaluate_cost(d_l0);
    if (cost_linear_0 < cost_best)
    {
      d_best = d_l0;
      cost_best = cost_linear_0;
    }
  }

  // compute minimum constrained to linear and 100 boundary -> (1/rho)*z - (1/node.m)*node.k*(node.o-node.L+100*node.k(node.index)) + (1/rho/node.m)*node.k*(node.k(node.index)*z(node.index)-z'*node.k);
  std::map<int, double> d_l1;
  for (const auto & item : this->node_info)
    d_l1[item.first] = 1 / this->rho * z[item.first] - 1 / this->m * item.second.k * (this->o - this->L + 100 * this->node_info[this->index].k) + 1 / this->rho / this->m * item.second.k * (this->node_info[this->index].k * z[this->index] - z_dot_k);
  d_l1[this->index] = 100;

  Serial.print("d_l1 = ");
  print_map2(d_l1);
  if (check_feasibility(d_l1)) {
    Serial.print(" -> Feasible");
    Serial.printf(" -> cost = %lf", evaluate_cost(d_l1));
    if (evaluate_cost(d_l1) < cost_best)
      Serial.print(" -> The best\n");
    else
      Serial.print("\n");
  }
  else
    Serial.print(" -> Not Feasible\n");
    
  if (check_feasibility(d_l1))
  {
    double cost_linear_100 = evaluate_cost(d_l1);
    if (cost_linear_100 < cost_best)
    {
      d_best = d_l1;
      cost_best = cost_linear_100;
    }
  }

  for (auto & item : this->node_info)
    item.second.d = d_best[item.first];

  return *this;
}

void Node::set_occupancy(int occupancy) {
  _occupancy = occupancy;
  set_lower_bound();
}

int Node::get_occupancy() {return _occupancy;}

void Node::set_lower_bound_occupied(double lower_bound_occupied) {
  _lower_bound_occupied = lower_bound_occupied;
  set_lower_bound();
}

double Node::get_lower_bound_occupied() {return _lower_bound_occupied;}

void Node::set_lower_bound_unoccupied(double lower_bound_unoccupied) {
  _lower_bound_unoccupied = lower_bound_unoccupied;
  set_lower_bound();
}

double Node::get_lower_bound_unoccupied() {return _lower_bound_unoccupied;}

void Node::set_lower_bound() {
  if (_occupancy == 1)
    _lower_bound = _lower_bound_occupied;
  else if (_occupancy == 0)
    _lower_bound = _lower_bound_unoccupied;
}

double Node::get_lower_bound() {return _lower_bound;}

void Node::set_cost(double cost) {_cost = cost;}

double Node::get_cost() {return _cost;}