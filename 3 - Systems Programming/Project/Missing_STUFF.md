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

 
- [ ] Don´t forget to edit the 'cleanup' function in `astronaut-display-client.c` to end the thread.
- [ ] Arranjar maneira de que se o server termina tanto por decisão do user do server como por fim do jogo, o server avisa os clientes todos para que eles se possam desligar sem dar erro.
- [ ] Não sei o que possa falta mais....

 



**Algo assim para tratar do Q e q do server?**

// ...existing code...

static volatile int server_running = 1;

// Broadcast a game-end message to all clients
void broadcast_game_end(void *publisher)
{
    remote_char_t msg;
    msg.msg_type = MSG_TYPE_GAME_END;
    zmq_send(publisher, &msg, sizeof(msg), 0);
}

// Thread that does a blocking getch() until 'q'/'Q' is pressed
void *keyboard_thread(void *arg)
{
    shared_data_t *data = (shared_data_t *)arg;

    // In main(), after initscr() is called, we can safely do blocking getch().
    while (server_running)
    {
        int ch = getch();   // Blocks until a key is pressed
        if (ch == 'q' || ch == 'Q')
        {
            broadcast_game_end(data->publisher);
            server_running = 0;
            break;
        }
    }
    return NULL;
}

// ...existing code...

int main()
{
    // ...existing code...

    // Initialize ncurses before creating the thread
    initscr();
    cbreak();
    noecho();

    pthread_t key_thread;
    pthread_create(&key_thread, NULL, keyboard_thread, &shared_data);

    // Example server loop (blocking calls to zmq_recv, etc.) as normal,
    // but exit when server_running is pulled low
    while (server_running)
    {
        // ...handle messages...
    }

    // Join the keyboard thread and clean up
    pthread_join(key_thread, NULL);

    // ...existing teardown code...
    endwin();
    return 0;
}