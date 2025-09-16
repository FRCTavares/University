#include <stdio.h>
#include <thread>


void thread_func(int thread_id, int groupnum) {
    int count = 0, init = groupnum*thread_id;
    while(1){
        if(init <= 1) break;
        else{
            if(init%2 == 0) init /= 2;
            else init = 3*init + 1;

            count += 1;
        }
    }
    printf("Thread ID: %d; Count: %d\n", thread_id, count);

}


int main(int argc, char** argv) {

    const int num_threads = 8;

    std::thread problem1_threads[num_threads];

    // TO DO: "spawn" your threads and make the program running correctly
    for (int i = 0; i < num_threads; i++) {
        problem1_threads[i] = std::thread(thread_func, i, 16);
    }

    for (int i = 0; i < num_threads; i++) {
        problem1_threads[i].join();
    }

    return 0;
}

