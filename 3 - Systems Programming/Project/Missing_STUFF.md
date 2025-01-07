# Project Roadmap Part B

**We will use threads and the old versions of astronaut-client.c and outer-space-dispay.c should still work with this new additions**

- `astronaut-display-client.c` will be the new client program, the main difference is that there will be two concurrent programs (threads):
    1. Like before the part where the client send messages to the server to move/disconnect the alien as well as firing the zap.
    2. A concurrent program displays the game board in real time as well as all the scores.

    This program basically merges the two old clients in one.

- `game-server.c` manages the outer space (game board) and the aliens (**We can't use forks**). It recieves messages from both `astronaut-display-client.c` & `astronaut-client.c`and sends updates to them and to the `outer-space-display.c` program.

    If the user presses *Q*, the corresponding program should terminate orderly, if it is the server then all the clients should disconnect and terminate.

    The server will have a thread to replace the existing fork system to handle alien movement. This thread will modify the data of the aliens directly. 

- `space-high-scores.c`this will be a new application that will store and display the highest scores in the game.

    It recieves all score updates from the server and only display the scores of all astronauts.

    This application needs to be developed in a *language different from C* and use ZeroMQ sockets and protocol buffers to comunicate with the server.

**If for 10 seconds, no alien is killed then its population should increase by 10%**


***We cannot use: fork, select, non-blocking com, active wait and signals***


## Missing Items

  
- [ ] Implement the 10-second alien population growth rule.  
 
- [ ] Don´t forget to edit the 'cleanup' function in `astronaut-display-client.c` to end the thread.

 



**What I will do probably to the handle the alien movement with threads**

// ...existing code...

typedef struct {
    ch_info_t aliens[ALIEN_COUNT];
    int alien_count;
    pthread_mutex_t lock;
    // ...other shared fields...
} shared_data_t;

void *alien_movement_thread(void *arg) {
    shared_data_t *data = (shared_data_t *)arg;
    while (1) {
        pthread_mutex_lock(&data->lock);
        // ...move aliens...
        pthread_mutex_unlock(&data->lock);
        sleep(1);
    }
    pthread_exit(NULL);
}

void *message_thread(void *arg) {
    shared_data_t *data = (shared_data_t *)arg;
    while (1) {
        // ...receive messages...
        pthread_mutex_lock(&data->lock);
        // ...update aliens if one dies...
        pthread_mutex_unlock(&data->lock);
    }
    pthread_exit(NULL);
}

int main() {
    shared_data_t data;
    data.alien_count = ALIEN_COUNT;
    pthread_mutex_init(&data.lock, NULL);
    // ...initialize data.aliens...
    pthread_t alien_tid, message_tid;
    pthread_create(&alien_tid, NULL, alien_movement_thread, &data);
    pthread_create(&message_tid, NULL, message_thread, &data);
    pthread_join(alien_tid, NULL);
    pthread_join(message_tid, NULL);
    pthread_mutex_destroy(&data.lock);
    return 0;
}
// ...existing code...



