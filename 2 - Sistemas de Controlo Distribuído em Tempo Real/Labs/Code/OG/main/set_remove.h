#ifndef SET_REMOVE_H
#define SET_REMOVE_H

#include <set>

template<typename T>
void set_remove(std::set<T>& s, T elem)
{
  auto it = s.find(elem);
  if (it != s.end())
    s.erase(it);
}

#endif