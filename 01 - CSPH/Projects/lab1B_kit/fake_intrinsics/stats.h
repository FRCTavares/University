#ifndef STATS_H
#define STATS_H

struct stats{
    int num_lanes;
    int cycles;
    int vector_ins;
    int max_active;
    int active;
    float vec_use;
};

#endif