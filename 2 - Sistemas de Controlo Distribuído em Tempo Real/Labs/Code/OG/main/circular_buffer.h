#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#include <vector>

struct buffer_data {
  double lux_value;
  double duty_cycle;
};

template <int _size = 10>
class CircularBuffer {
public:
  CircularBuffer() : _head(0), _tail(0), _used(0){};

  int get_buffer_size() { return _size; };

  int get_used_space() { return _used; };

  bool is_empty() { return _used == 0; };

  bool is_full() { return _used == _size; };

  void insert_new(buffer_data elem) {
    if (!is_full())
      _used++;
    else {
      _tail++;
      if (_tail == _size)
        _tail = 0;
    }

    _head++;
    if (_head == _size)
        _head = 0;

    _buffer[_head] = elem;
  };

  buffer_data remove_oldest() {
    buffer_data oldest = _buffer[_tail];  // invalid if buffer empty

    if (!is_empty()) {
      _used--;
      _tail++;

      if (_tail == _size)
        _tail = 0;
    }

      return oldest;
  };

  std::vector<double> get_recent_lux_values(int num_elements) {
    std::vector<double> lux_values;
    lux_values.reserve(num_elements);

    if (is_empty()) {
      return lux_values;
    }

    int start_idx = _head - num_elements + 1;
    if (start_idx < 0) {
      start_idx += _size;
    }

    int i = start_idx;
    while (i != _head)
    {
      lux_values.push_back(_buffer[i].lux_value);
      i++;
      if (i == _size)
        i = 0;
    }
    lux_values.push_back(_buffer[_head].lux_value);

    return lux_values;
  };

private:
  buffer_data _buffer[_size];
  int _head;
  int _tail;
  int _used;
};

#endif
