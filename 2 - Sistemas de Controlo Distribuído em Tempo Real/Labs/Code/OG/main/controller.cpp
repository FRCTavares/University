#include "Arduino.h"
#include "controller.h"

const int DAC_RANGE = 4096;

void Controller::set_controller(int LUMINAIRE) {

  if (LUMINAIRE == 33) {
    /* 1st Luminaire */
    _K = 5.0;
    _Ti = 250.0;
    _b = 7.0;
    _Tt = 100.0;
  }
  else if (LUMINAIRE == 38) {
    /* 2nd Luminaire */
    _K = 7.5;
    _Ti = 200.0;
    _b = 7.0;
    _Tt = 100.0;
  }
  else if (LUMINAIRE == 39) {
    /* 3rd Luminaire */
    _K = 7.5;
    _Ti = 300.0;
    _b = 7.0;
    _Tt = 100.0;
  }

  _up = 0;
  _ui = 0;
  _uff = 0;
  _u = 0;

  _anti_windup = 1;
  _feedback = 1;

  _Kold = _K, _bold = _b;
}

/* PI Controller with Set Point Weighting, Anti-Windup solved by Back-Calculation and Bumpless Transfer */
double Controller::get_control_signal(double r, double y, double h) {  
  double v;

  // Update integral for Bumpless Transfer
  // _ui += _Kold * (_bold * r - y) - _K * (_b * r - y);
  // _Kold = _K, _bold = _b;

  // Update proportional part
  //_up = _K * (_b * r - y);

  // Temporary output
  //v = _up + _ui;
  v = _uff + _ui;

  // Anti-Windup - Bound the actuation signal
  if (_anti_windup)
    _u = min(max(v, 0.0), DAC_RANGE);
  else // Let the actuator saturate (_anti_windup = 0)
    _u = v;

  // Update the integral part, with an additional Back-Calculation term (to slowly discharge the integrator)
  _ui += h * _K / _Ti * (r - y) + h / _Tt * (_u - v);

  return _u;
}

double Controller::get_duty_cycle() {return (double) _u / DAC_RANGE;}

void Controller::set_anti_windup(int anti_windup) {_anti_windup = anti_windup;}

int Controller::get_anti_windup() {return _anti_windup;}

void Controller::set_feedback(int feedback) {_feedback = feedback;}

int Controller::get_feedback() {return _feedback;}

double Controller::get_u() {return _u;}

void Controller::set_k(double K) {_Kold = _K, _K = K;}

double Controller::get_k() {return _K;}

void Controller::set_ti(double Ti) {_Ti = Ti;}

double Controller::get_ti() {return _Ti;}

void Controller::set_b(double b) {_bold = _b, _b = b;}

double Controller::get_b() {return _b;}

void Controller::set_tt(double Tt) {_Tt = Tt;}

double Controller::get_tt() {return _Tt;}

void Controller::set_modeOp(char subcmd, int i) {

  _Kold = _K, _bold = _b;

  switch (subcmd) {
    case 'l':
      if (i == 33 || i == 38)
        _K = 5.0, _b = 7.0;
      if (i == 3)
        _K = 5.0, _b = 5.0;
      break;

    case 'm':
      if (i == 33)
        _K = 5.0, _b = 7.0;
      if (i == 38 || i == 39)
        _K = 7.5, _b = 7.0;        
      break;

    case 'h':
      if (i == 33 || i == 39)
        _K = 10.0, _b = 10.0;
      if (i == 38)
        _K = 15.0, _b = 10.0;
      break;
        
    default:
      Serial.println("err -> Invalid Command, please try again.");
      return; 
  }
}

void Controller::update_control_signal(double u_consensus) {_ui = 0, _uff = u_consensus;}