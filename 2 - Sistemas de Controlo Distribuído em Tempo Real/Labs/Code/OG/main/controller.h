#ifndef CONTROLLER_H
#define CONTROLLER_H

class Controller {
public:
  void set_controller(int LUMINAIRE);
  double get_control_signal(double r, double y, double h);
  double get_duty_cycle();

  void set_anti_windup(int anti_windup);
  int get_anti_windup();
  void set_feedback(int feedback);
  int get_feedback();

  double get_u();

  void set_k(double K);
  double get_k();
  void set_ti(double Ti);
  double get_ti();
  void set_b(double b);
  double get_b();
  void set_tt(double Tt);
  double get_tt();

  void set_modeOp(char subcmd, int i);

  void update_control_signal(double u_consensus);

private:
  double _K, _Ti, _b, _Tt, _up, _ui, _uff, _u;
  int _anti_windup, _feedback;
  double _Kold, _bold;
};

#endif