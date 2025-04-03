void run_consensus(bool first)
{
  // Consensus is only allowed after calibrating
  if (coupling_gains.size() != other_luminaires.size() + 2)
    return;
  if (first)
    enqueue_message(BROADCAST, msg_t::RUN_CONSENSUS, nullptr, 0);    
  is_running_consensus = true;
  is_the_first_iteration = true;
  consensus_stage = consensus_stage_t::CONSENSUS_ITERATION;
  d_other_luminaires.clear();
  consensus_iteration = 0;
}

void consensus_loop() 
{
  if (is_running_consensus)
  {
    switch (consensus_stage)
    {
      case consensus_stage_t::CONSENSUS_ITERATION:
      {
        consensus_iteration++;
        Serial.printf("Iteration number: %d\n", consensus_iteration);
        if (is_the_first_iteration)
        {
          node.initialization(coupling_gains, LUMINAIRE);
          is_the_first_iteration = false;
        }

        // Run consensus
        node.consensus_iterate();

        // Communications - Send messages
        uint8_t data[5] = {0};
        for (const auto & item : node.node_info) {
          Serial.printf("First: %d, Second: %lf\n", item.first, item.second);
          data[0] = (uint8_t) item.first;
          float value = (float) item.second.d;
          memcpy(data+1, &value, sizeof(value));
          enqueue_message(BROADCAST, CONSENSUS_VALUE, data, sizeof(data));
        }
        consensus_stage = consensus_stage_t::WAIT_FOR_MESSAGES;
        d_other_luminaires.clear();
        break;
      }

      case consensus_stage_t::WAIT_FOR_MESSAGES:
        if (d_other_luminaires.size() == other_luminaires.size() * (other_luminaires.size() + 1))
        {
          // Compute the mean duty-cycle
          for (auto & item1 : node.node_info)
          {
            item1.second.d_av = item1.second.d;
            for (auto & item2 : d_other_luminaires)
            {
              if (item2.first.second == item1.first)
                item1.second.d_av += item2.second;
            }
            item1.second.d_av /= (other_luminaires.size() + 1);
          }
          
          // Computation of lagrangian updates
          for (auto & item : node.node_info)
            item.second.y = item.second.y + node.rho * (item.second.d - item.second.d_av);

          // Next iteration
          if (consensus_iteration != maxiter)
            consensus_stage = consensus_stage_t::CONSENSUS_ITERATION;
          else // End of consensus
          {
            Serial.printf("Chegou ao final do consensus\n");
            is_running_consensus = false;
            double k_dot_d = 0.0;
            for (auto & item : node.node_info) {
              item.second.d_av = min(max(item.second.d_av, 0), 100);
              k_dot_d += item.second.k * item.second.d_av;
              Serial.printf("duty cycle: %lf\n", item.second.d_av);
            }
            r = k_dot_d + node.o;
            controller.update_control_signal(node.node_info[LUMINAIRE].d_av / 100 * DAC_RANGE);
          }
        }
        break;
      
      default:
        break;
    }
  }
}

void master_calibrate_routine()
{
  luminaire_ids.clear();  
  luminaire_ids.push_back(LUMINAIRE);
  std::copy(other_luminaires.begin(), other_luminaires.end(), std::back_inserter(luminaire_ids));
  coupling_gains.clear();
  total_calibrations = 0;
  is_calibrating_as_master = true;
  is_calibrating = true;
  ready_luminaires.clear();
  calibration_stage = calibration_stage_t::WAIT_FOR_ACK;
  next_stage = calibration_stage_t::CALIBRATING;
  controller.set_feedback(false);
  analogWrite(LED_PIN, 0);
  enqueue_message(BROADCAST, msg_t::OFF, nullptr, 0);
  r = 0;
  float r_float = (float) r;
  enqueue_message(BROADCAST, msg_t::SET_REFERENCE, (uint8_t*) &r_float, sizeof(r_float));
}

void enqueue_message(uint8_t recipient, msg_t type, uint8_t *message, std::size_t msg_size)
{
  if (type != msg_t::PING)
    Serial.printf("Enqueuing message of type %s with size %lu to recipient %d\n", message_type_translations[type], msg_size, recipient);
  can_frame *new_frame = new can_frame;
  std::size_t length = min(msg_size + 2, CAN_MAX_DLEN);
  new_frame->can_id = recipient;
  new_frame->can_dlc = length;
  if (msg_size > 0 && message != nullptr)
    memcpy(new_frame->data + 2, message, length - 2);
  if (msg_size > length - 2 && message != nullptr)
    new_frame->can_id |= ((uint32_t) message[length - 2]) << 8;  
  if (msg_size > length - 1 && message != nullptr)
    new_frame->can_id |= ((uint32_t) message[length - 1]) << 16;    
  new_frame->data[0] = LUMINAIRE;
  new_frame->data[1] = type;
  rp2040.fifo.push_nb((uint32_t)new_frame);
}

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

std::set<uint8_t> other_luminaires;
std::vector<uint8_t> luminaire_ids;
std::set<uint8_t> ready_luminaires;
std::map<int, double> coupling_gains;
std::map<int, std::set<unsigned short>> stream_l, stream_d, stream_j; 
bool is_calibrating_as_master = false, is_calibrating = false;
uint8_t calibration_master = 0, calibrating_luminaire = 0;
std::size_t total_calibrations = 0;

bool is_running_consensus = false;
bool is_the_first_iteration = false;
int consensus_iteration = 0;
int maxiter = 50;
std::map<std::pair<int, int>, double> d_other_luminaires;

enum calibration_stage_t : uint8_t
{
  WAIT_FOR_ACK = 0,
  CHANGE_STATE,
  CALIBRATING,
  DONE
};
calibration_stage_t calibration_stage, next_stage;

enum consensus_stage_t : uint8_t
{
  CONSENSUS_ITERATION,
  WAIT_FOR_MESSAGES
};

consensus_stage_t consensus_stage;

std::set<uint8_t> ready_luminaires;