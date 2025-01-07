#include "zhelpers.h"
#include "protocol.h"
#include "message.pb-c.h"

typedef struct screen_update_t
{
    int pos_x;
    int pos_y;
    char ch;
    char players[MAX_PLAYERS];
    int scores[MAX_PLAYERS];
    int player_count;

} screen_update_t;

typedef struct
{
    ch_info_t aliens[256];
    int alien_count;
    pthread_mutex_t lock;
    int grid[16][16];
    WINDOW *my_win;
    WINDOW *score_win;
    void *publisher;
    void *publisher2;
    time_t last_alien_kill;

    // Is there any more data that needs sharing?
} shared_data_t;

/*
    Function to calculate the new position of the character based on the direction of movement

    Parameters:
    x: Pointer to the x-coordinate of the character
    y: Pointer to the y-coordinate of the character
    direction: Direction of movement
    dir: Whether the character moves vertically(1) or horizontally(0)

    Returns:
    None
*/

void new_position(int *x, int *y, direction_t direction, int dir)
{

    if (dir == 0)
    { // Horizontal movement
        switch (direction)
        {
        case LEFT:
            (*y)--;
            if (*y < 3)
                *y = 3;
            break;
        case RIGHT:
            (*y)++;
            if (*y > 18)
                *y = 18;
            break;
        default:
            break;
        }
    }
    else
    { // Vertical movement
        switch (direction)
        {
        case UP:
            (*x)--;
            if (*x < 3)
                *x = 3;
            break;
        case DOWN:
            (*x)++;
            if (*x > 18)
                *x = 18;
            break;
        default:
            break;
        }
    }
}

/*
    Function to find the index of a character in the char_data array

    Parameters:
    char_data: Array of character information
    n_chars: Number of characters in the array
    ch: Character to find

    Returns:
    Index of the character in the array, -1 if not found
*/

int find_ch_info(ch_info_t char_data[], int n_chars, int ch)
{
    for (int i = 0; i < n_chars; i++)
    {
        if (char_data[i].ch == ch)
        {
            return i;
        }
    }
    return -1; // Character not found
}

/*
    Function to fire a laser from the astronaut in the specified direction

    Parameters:
    win: Game window
    astronaut: Pointer to the astronaut firing the laser
    aliens: Array of alien information
    alien_count: Pointer to the number of aliens
    char_data: Array of character information
    n_chars: Number of characters
    publisher: ZeroMQ publisher socket
    grid: 2D array to track alien positions

    Returns:
    None
*/

void fire_laser(WINDOW *win, ch_info_t *astronaut, ch_info_t aliens[], int *alien_count, ch_info_t char_data[], int n_chars, void *publisher, int grid[16][16], pthread_mutex_t lock, time_t last_alien_kill)
{

    screen_update_t update;
    int x = astronaut->pos_x;
    int y = astronaut->pos_y;
    int flag;

    if (astronaut->dir == 0)
    { // Horizontal movement, fire vertically
        for (int i = 1; i < WINDOW_SIZE - 1; i++)
        {
            flag = 0;

            // Check for aliens
            for (int j = 0; j < *alien_count; j++)
            {
                if (aliens[j].pos_x == i && aliens[j].pos_y == y)
                {
                    last_alien_kill = time(NULL);
                    // Mark grid cell as empty
                    grid[i - 3][y - 3] = 0;

                    // Remove alien from the list
                    for (int k = j; k < *alien_count - 1; k++)
                    {
                        aliens[k] = aliens[k + 1];
                    }

                    flag = 1;
                    (*alien_count)--;
                    astronaut->score++;
                    break;
                }
            }

            // Check for other astronauts
            for (int j = 0; j < n_chars; j++)
            {
                if (char_data[j].pos_x == i && char_data[j].pos_y == y)
                {
                    if (char_data[j].ch != astronaut->ch)
                        char_data[j].stunned = time(NULL);
                    flag = 1;
                }
            }

            // Update screen
            if (flag == 0)
            {
                wmove(win, i, y);
                waddch(win, '|');
                wrefresh(win);

                update.pos_x = i;
                update.pos_y = y;
                update.ch = '|';
                // pthread_mutex_lock(&lock);
                zmq_send(publisher, &update, sizeof(screen_update_t), 0);
                // pthread_mutex_unlock(&lock);
            }
        }
    }
    else
    { // Vertical movement, fire horizontally
        for (int i = 1; i < WINDOW_SIZE - 1; i++)
        {

            flag = 0;

            // Check for aliens
            for (int j = 0; j < *alien_count; j++)
            {
                if (aliens[j].pos_x == x && aliens[j].pos_y == i)
                {
                    last_alien_kill = time(NULL);
                    // Mark grid cell as empty
                    grid[i - 3][y - 3] = 0;

                    // Remove alien from the list
                    for (int k = j; k < *alien_count - 1; k++)
                    {
                        aliens[k] = aliens[k + 1];
                    }
                    flag = 1;
                    (*alien_count)--;
                    astronaut->score++;
                    break;
                }
            }

            // Check for other astronauts
            for (int j = 0; j < n_chars; j++)
            {
                if (char_data[j].pos_x == x && char_data[j].pos_y == i)
                {
                    if (char_data[j].ch != astronaut->ch)
                        char_data[j].stunned = time(NULL);
                    flag = 1;
                }
            }

            if (flag == 0)
            {
                wmove(win, x, i);
                waddch(win, '-');
                wrefresh(win);

                update.pos_x = x;
                update.pos_y = i;
                update.ch = '-';
                // pthread_mutex_lock(&lock);
                zmq_send(publisher, &update, sizeof(screen_update_t), 0);
                // pthread_mutex_unlock(&lock);
            }
        }
    }

    usleep(500000); // Display laser for 0.5 seconds

    // Clear laser
    if (astronaut->dir == 0)
    {
        for (int i = 1; i < WINDOW_SIZE - 1; i++)
        {
            wmove(win, i, y);
            waddch(win, ' ');
            wrefresh(win);

            update.pos_x = i;
            update.pos_y = y;
            update.ch = ' ';
            // pthread_mutex_lock(&lock);
            zmq_send(publisher, &update, sizeof(screen_update_t), 0);
            // pthread_mutex_unlock(&lock);
        }
    }
    else
    {
        for (int i = 1; i < WINDOW_SIZE - 1; i++)
        {
            wmove(win, x, i);
            waddch(win, ' ');
            wrefresh(win);

            update.pos_x = x;
            update.pos_y = i;
            update.ch = ' ';
            // pthread_mutex_lock(&lock);
            zmq_send(publisher, &update, sizeof(screen_update_t), 0);
            // pthread_mutex_unlock(&lock);
        }
    }
}

/*
    Function to update the scoreboard with the current scores of all players

    Parameters:
    score_win: Scoreboard window
    char_data: Array of character information
    n_chars: Number of characters
    alien_count: Number of aliens
    publisher: ZeroMQ publisher socket

    Returns:
    None
*/

void update_scoreboard(WINDOW *score_win, ch_info_t char_data[], int n_chars, int alien_count, void *publisher, pthread_mutex_t lock, void *publisher2)
{

    werase(score_win); // Clear the scoreboard window

    // Display header
    mvwprintw(score_win, 1, 3, "SCORE");

    // Display each player's score
    for (int i = 0; i < n_chars; i++)
    {
        mvwprintw(score_win, 2 + i, 3, "%c - %d", char_data[i].ch, char_data[i].score);
    }

    // Alien Alive Counter at the bottom of the scoreboard
    mvwprintw(score_win, 20, 3, "Aliens Alive: %d", alien_count);

    box(score_win, 0, 0); // Draw the border
    wrefresh(score_win);  // Refresh to show changes

    // Build the update to send to outer-space-display
    screen_update_t update;
    update.ch = 's';
    update.player_count = n_chars;

    for (int i = 0; i < n_chars; i++)
    {
        update.players[i] = char_data[i].ch;
        update.scores[i] = char_data[i].score;
    }

    // pthread_mutex_lock(&lock);
    zmq_send(publisher, &update, sizeof(screen_update_t), 0);
    // pthread_mutex_unlock(&lock);

    // Build the update to send to outer-space-display
    ScoreUpdate score_update = SCORE_UPDATE__INIT;

    int scores[n_chars];
    char characters[n_chars][2];

    for (int i = 0; i < n_chars; i++)
    {
        scores[i] = char_data[i].score;
        characters[i][0] = char_data[i].ch;
        characters[i][1] = '\0';
    }

    score_update.n_scores = n_chars;
    score_update.n_characters = n_chars;

    score_update.scores = scores;
    score_update.characters = malloc(sizeof(char *) * score_update.n_characters);

    for (int i = 0; i < score_update.n_scores; i++)
    {
        score_update.characters[i] = (char *)characters[i];
    }

    unsigned message_size = score_update__get_packed_size(&score_update);

    void *message_data = malloc(message_size);

    // Pack the message into the byte buffer
    score_update__pack(&score_update, message_data);

    // Send the serialized message over ZeroMQ
    zmq_send(publisher2, message_data, message_size, 0);

    // Clean up

    free(message_data);
}

/*
    Function to remove an astronaut from the game

    Parameters:
    win: Game window
    char_data: Array of character information
    n_chars: Pointer to the number of characters
    ch_pos: Index of the character to remove
    publisher: ZeroMQ publisher socket
    score_win: Scoreboard window
    alien_count: Number of aliens

    Returns:
    None
*/

void remove_astronaut(WINDOW *win, ch_info_t char_data[], int *n_chars, int ch_pos, void *publisher, WINDOW *score_win, int alien_count, pthread_mutex_t lock, void *publisher2)
{

    screen_update_t update;

    // Erase astronaut from the game window
    wmove(win, char_data[ch_pos].pos_x, char_data[ch_pos].pos_y);
    waddch(win, ' ');
    wrefresh(win);

    // Send update to clients to clear the character
    update.pos_x = char_data[ch_pos].pos_x;
    update.pos_y = char_data[ch_pos].pos_y;
    update.ch = ' ';
    // pthread_mutex_lock(&lock);
    zmq_send(publisher, &update, sizeof(screen_update_t), 0);
    // pthread_mutex_unlock(&lock);

    // Remove astronaut from the char_data array
    for (int i = ch_pos; i < (*n_chars) - 1; i++)
    {
        char_data[i] = char_data[i + 1];
    }
    (*n_chars)--;

    // Update the scoreboard to remove the player's score
    update_scoreboard(score_win, char_data, *n_chars, alien_count, publisher, lock, publisher2);
}

// Main part of the Program

// PRINCIPAL
void *message_thread(void *arg)
{
    shared_data_t *data = (shared_data_t *)arg;

    // Initialize character & alien data
    ch_info_t char_data[MAX_PLAYERS];
    int n_chars = 0;
    int score = 0;

    // Initialize ncurses
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();

    // Initialize game window and score board window

    data->my_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    box(data->my_win, 0, 0);
    wrefresh(data->my_win);

    data->score_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, WINDOW_SIZE + 1);
    box(data->score_win, 0, 0);
    wrefresh(data->score_win);

    // Initial print of the scoreboard
    update_scoreboard(data->score_win, char_data, n_chars, data->alien_count, data->publisher, data->lock, data->publisher2);

    // Initialize aliens
    for (int i = 0; i < ALIEN_COUNT; i++)
    {
        data->aliens[i].ch = '*';
        // Generate a position within the 16x16 grid
        do
        {
            data->aliens[i].pos_x = rand() % 16 + 3; // Positions 3 to 18
            data->aliens[i].pos_y = rand() % 16 + 3;
        } while (data->grid[data->aliens[i].pos_x - 3][data->aliens[i].pos_y - 3] != 0);

        // Mark the position as occupied
        data->grid[data->aliens[i].pos_x - 3][data->aliens[i].pos_y - 3] = 1;
    }

    // Spawn all the aliens on the screen
    for (int i = 0; i < data->alien_count; i++)
    {
        wmove(data->my_win, data->aliens[i].pos_x, data->aliens[i].pos_y);
        waddch(data->my_win, '*' | A_BOLD);
    }

    // Initialize ZeroMQ
    void *context = zmq_ctx_new();
    if (context == NULL)
    {
        perror("Failed to create ZeroMQ context");
        pthread_exit(NULL);
    }

    // Main program starts
    int ch;
    int pos_x;
    int pos_y;
    time_t current_time;
    direction_t direction;
    remote_char_t m;
    screen_update_t update;

    int connected_astronauts[MAX_PLAYERS] = {0};

    // Socket to talk to clients
    void *responder = zmq_socket(context, ZMQ_REP);
    if (responder == NULL)
    {
        perror("Server zmq_socket failed");
        zmq_ctx_destroy(context);
        pthread_exit(NULL);
    }

    int rc = zmq_bind(responder, SERVER_REQUEST_ADDRESS);
    if (rc != 0)
    {
        perror("Server zmq_bind failed");
        zmq_close(responder);
        zmq_ctx_destroy(context);
        pthread_exit(NULL);
    }

    while (1)
    {
        rc = zmq_recv(responder, &m, sizeof(remote_char_t), 0);
        if (rc == -1)
        {
            perror("Server zmq_recv failed");
            pthread_exit(NULL);
            ;
        }

        if (m.msg_type == MSG_TYPE_CONNECT)
        {
            /*
            Handle connection request from a new player
            */

            if (n_chars == MAX_PLAYERS)
            {
                // Send a failure response if max players reached
                rc = zmq_send(responder, NULL, 0, 0);
                if (rc == -1)
                {
                    perror("Server connection response failed");
                    pthread_exit(NULL);
                    ;
                }
                continue;
            }

            // Choose cyclically a character for the connected player
            for (int i = 0; i < MAX_PLAYERS; i++)
            {
                if (connected_astronauts[i] == 0)
                {
                    connected_astronauts[i] = 1;
                    ch = 'A' + i;
                    break;
                }
            }

            // Assign positions and direction based on character
            if (ch == 'A')
            {
                pos_x = 5;
                pos_y = 1;
                char_data[n_chars].dir = 1;
            }
            else if (ch == 'B')
            {
                pos_x = 19;
                pos_y = 7;
                char_data[n_chars].dir = 0;
            }
            else if (ch == 'C')
            {
                pos_x = 20;
                pos_y = 12;
                char_data[n_chars].dir = 0;
            }
            else if (ch == 'D')
            {
                pos_x = 10;
                pos_y = 19;
                char_data[n_chars].dir = 1;
            }
            else if (ch == 'E')
            {
                pos_x = 1;
                pos_y = 8;
                char_data[n_chars].dir = 0;
            }
            else if (ch == 'F')
            {
                pos_x = 8;
                pos_y = 20;
                char_data[n_chars].dir = 1;
            }
            else if (ch == 'G')
            {
                pos_x = 2;
                pos_y = 10;
                char_data[n_chars].dir = 0;
            }
            else if (ch == 'H')
            {
                pos_x = 11;
                pos_y = 2;
                char_data[n_chars].dir = 1;
            }
            else
            {
                // Handle unexpected character
                pos_x = 0;
                pos_y = 0;
                char_data[n_chars].dir = 0;
            }

            char_data[n_chars].ch = ch;
            char_data[n_chars].pos_x = pos_x;
            char_data[n_chars].pos_y = pos_y;
            char_data[n_chars].score = 0;
            char_data[n_chars].last_fire_time = 0;
            char_data[n_chars].stunned = 0;
            char_data[n_chars].GAME_TOKEN = rand() % 1000;

            rc = zmq_send(responder, &char_data[n_chars], sizeof(ch_info_t), 0);
            if (rc == -1)
            {
                perror("Server connection response failed");
                pthread_exit(NULL);
                ;
            }

            n_chars++;

            // Update the scoreboard with the new player's score
            update_scoreboard(data->score_win, char_data, n_chars, data->alien_count, data->publisher, data->lock, data->publisher2);
        }
        else if (m.msg_type == MSG_TYPE_ZAP)
        {
            /*
            Handle laser firing request from a player
            */

            int ch_pos = find_ch_info(char_data, n_chars, m.ch);
            current_time = time(NULL);

            if (char_data[ch_pos].GAME_TOKEN != m.GAME_TOKEN)
            {
                rc = zmq_send(responder, NULL, 0, 0);
                if (rc == -1)
                {
                    perror("Server zmq_send failed");
                    pthread_exit(NULL);
                }
                continue;
            }

            if (ch_pos != -1 && difftime(current_time, char_data[ch_pos].stunned) >= 10)
            {
                current_time = time(NULL);
                if (difftime(current_time, char_data[ch_pos].last_fire_time) >= 3)
                {
                    // pthread_mutex_lock(&data->lock);
                    fire_laser(data->my_win, &char_data[ch_pos], data->aliens, &data->alien_count, char_data, n_chars, data->publisher, data->grid, data->lock, data->last_alien_kill);
                    // pthread_mutex_unlock(&data->lock);

                    char_data[ch_pos].last_fire_time = current_time;
                    if (data->alien_count == 0)
                    {
                        werase(data->my_win);
                        werase(data->score_win);
                        mvwprintw(data->my_win, 10, 10, "GAME OVER");
                        // Print final scores
                        for (int i = 0; i < n_chars; i++)
                        {
                            mvwprintw(data->my_win, 12 + i, 10, "%c - %d", char_data[i].ch, char_data[i].score);
                        }

                        // Build the update to send to outer-space-display
                        screen_update_t update;
                        update.ch = 'o';
                        update.player_count = n_chars;

                        for (int i = 0; i < n_chars; i++)
                        {
                            update.players[i] = char_data[i].ch;
                            update.scores[i] = char_data[i].score;
                        }

                        // pthread_mutex_lock(&data->lock);
                        zmq_send(data->publisher, &update, sizeof(screen_update_t), 0);
                        // pthread_mutex_unlock(&data->lock);

                        wrefresh(data->my_win);
                        wrefresh(data->score_win);
                        sleep(5);
                        break;
                    }
                }
            }

            rc = zmq_send(responder, &char_data[ch_pos].score, sizeof(int), 0);
            if (rc == -1)
            {
                perror("Server zmq_send failed");
                pthread_exit(NULL);
                ;
            }
        }
        else if (m.msg_type == MSG_TYPE_DISCONNECT)
        {
            /*
            Handle disconnection request from a player
            */
            int ch_pos = find_ch_info(char_data, n_chars, m.ch);

            // Check if the game token matches
            if (char_data[ch_pos].GAME_TOKEN != m.GAME_TOKEN)
            {
                rc = zmq_send(responder, NULL, 0, 0);
                if (rc == -1)
                {
                    perror("Server zmq_send failed");
                    pthread_exit(NULL);
                }
                continue;
            }

            // Remove the astronaut from the game
            if (ch_pos != -1)
            {
                connected_astronauts[m.ch - 'A'] = 0;
                remove_astronaut(data->my_win, char_data, &n_chars, ch_pos, data->publisher, data->score_win, data->alien_count, data->lock, data->publisher2);
            }
            // Send a response to the client to confirm disconnection
            int response = -2;
            rc = zmq_send(responder, &response, sizeof(int), 0);
            if (rc == -1)
            {
                perror("Server zmq_send failed");
                pthread_exit(NULL);
            }
        }
        else if (m.msg_type == MSG_TYPE_MOVE)
        {
            /*
            Handle movement request from a player
            */
            int ch_pos = find_ch_info(char_data, n_chars, m.ch);
            current_time = time(NULL);

            if (char_data[ch_pos].GAME_TOKEN != m.GAME_TOKEN)
            {
                rc = zmq_send(responder, NULL, 0, 0);
                if (rc == -1)
                {
                    perror("Server zmq_send failed");
                    pthread_exit(NULL);
                    ;
                }
                continue;
            }

            if (ch_pos != -1 && difftime(current_time, char_data[ch_pos].stunned) >= 10)
            {
                pos_x = char_data[ch_pos].pos_x;
                pos_y = char_data[ch_pos].pos_y;
                ch = char_data[ch_pos].ch;

                // Erase character from old position
                wmove(data->my_win, pos_x, pos_y);
                waddch(data->my_win, ' ');
                wrefresh(data->my_win);

                // Send update to clear the old position
                update.pos_x = pos_x;
                update.pos_y = pos_y;
                update.ch = ' ';
                // pthread_mutex_lock(&data->lock);
                zmq_send(data->publisher, &update, sizeof(screen_update_t), 0);
                // pthread_mutex_unlock(&data->lock);

                direction = m.direction;
                new_position(&pos_x, &pos_y, direction, char_data[ch_pos].dir);
                char_data[ch_pos].pos_x = pos_x;
                char_data[ch_pos].pos_y = pos_y;
            }

            rc = zmq_send(responder, &char_data[ch_pos].score, sizeof(int), 0);
            if (rc == -1)
            {
                perror("Server zmq_send failed");
                pthread_exit(NULL);
                ;
            }
        }
        else
        {
            /*
            Handle invalid message types
            */

            rc = zmq_send(responder, NULL, 0, 0);
            if (rc == -1)
            {
                perror("Server zmq_send failed");
                pthread_exit(NULL);
            }
            continue;
        }

        // Update display
        for (int i = 0; i < n_chars; i++)
        {
            wmove(data->my_win, char_data[i].pos_x, char_data[i].pos_y);
            waddch(data->my_win, char_data[i].ch | A_BOLD);
        }

        update_scoreboard(data->score_win, char_data, n_chars, data->alien_count, data->publisher, data->lock, data->publisher2);

        wrefresh(data->my_win);

        // Send updates to outer-space-display
        for (int i = 0; i < n_chars; i++)
        {
            update.pos_x = char_data[i].pos_x;
            update.pos_y = char_data[i].pos_y;
            update.ch = char_data[i].ch;
            // pthread_mutex_lock(&data->lock);
            if (zmq_send(data->publisher, &update, sizeof(screen_update_t), 0) == -1)
            {
                perror("Server zmq_send failed");
                pthread_exit(NULL);
            }
            // pthread_mutex_lock(&data->lock);
        }

        usleep(10000); // Sleep for 10ms
    }
    // Cleanup
    zmq_close(responder);
    zmq_close(data->publisher);
    zmq_ctx_destroy(context);
    pthread_exit(NULL);
}

// SECUNDÁRIO
void *alien_movement_thread(void *arg)
{
    shared_data_t *data = (shared_data_t *)arg;
    int aliens_moved = 0;

    while (1)
    {
        sleep(1);

        // pthread_mutex_lock(&data->lock);
        for (int i = 0; i < data->alien_count; i++)
        {
            int alien_index = i;
            direction_t direction = rand() % 4; // Random direction

            // Get current position
            int x = data->aliens[alien_index].pos_x;
            int y = data->aliens[alien_index].pos_y;

            // Move alien to new position based on direction
            int new_x = x;
            int new_y = y;
            int move_success = 0;
            aliens_moved++;

            // Check if the new position is valid
            switch (direction)
            {
            case UP:
                if (x >= 4 && data->grid[x - 4][y - 3] == 0)
                {
                    new_x--;
                    move_success = 1;
                }
                break;
            case DOWN:
                if (x <= 17 && data->grid[x - 2][y - 3] == 0)
                {
                    new_x++;
                    move_success = 1;
                }
                break;
            case LEFT:
                if (y >= 4 && data->grid[x - 3][y - 4] == 0)
                {
                    new_y--;
                    move_success = 1;
                }
                break;
            case RIGHT:
                if (y <= 17 && data->grid[x - 3][y - 2] == 0)
                {
                    new_y++;
                    move_success = 1;
                }
                break;
            default:
                break;
            }

            if (move_success == 1)
            {
                // Update grid to mark the old position as empty
                data->grid[x - 3][y - 3] = 0;

                // Update grid to mark the new position as occupied
                data->grid[new_x - 3][new_y - 3] = 1;

                data->aliens[alien_index].pos_x = new_x;
                data->aliens[alien_index].pos_y = new_y;
            }
        }

        // Check if all aliens have moved
        if (aliens_moved == data->alien_count)
        {
            // Erase all aliens from the screen ' ' just writes ' ' in the interior 16x16 window
            for (int i = 3; i < 19; i++)
            {
                for (int j = 3; j < 19; j++)
                {
                    wmove(data->my_win, i, j);
                    waddch(data->my_win, ' ');

                    // Send update to clear old position
                    screen_update_t clear_update;
                    clear_update.pos_x = i;
                    clear_update.pos_y = j;
                    clear_update.ch = ' ';
                    // pthread_mutex_lock(&data->lock);
                    zmq_send(data->publisher, &clear_update, sizeof(screen_update_t), 0);
                    // pthread_mutex_unlock(&data->lock);
                }
            }
            wrefresh(data->my_win);
            // Clycle that runns throught all the aliens and updates the server window for all the aliens and update the display client
            for (int i = 0; i < data->alien_count; i++)
            {
                // Update the display
                wmove(data->my_win, data->aliens[i].pos_x, data->aliens[i].pos_y);
                waddch(data->my_win, '*' | A_BOLD);

                screen_update_t update;
                update.pos_x = data->aliens[i].pos_x;
                update.pos_y = data->aliens[i].pos_y;
                update.ch = '*';
                // pthread_mutex_lock(&data->lock);
                zmq_send(data->publisher, &update, sizeof(screen_update_t), 0);
                // pthread_mutex_unlock(&data->lock);
            }
            wrefresh(data->my_win);
            aliens_moved = 0;
        }

        time_t now = time(NULL);
        if (difftime(now, data->last_alien_kill) >= 10)
        {
            // Increase alien count by ~10%
            int added = data->alien_count / 10;
            for (int i = 0; i < added && data->alien_count < MAX_ALIENS; i++)
            {
                // spawn new aliens
                int idx = data->alien_count++;
                data->aliens[idx].ch = '*';

                do
                {
                    data->aliens[idx].pos_x = rand() % 16 + 3;
                    data->aliens[idx].pos_y = rand() % 16 + 3;
                } while (data->grid[data->aliens[idx].pos_x - 3][data->aliens[idx].pos_y - 3] != 0);
                data->grid[data->aliens[idx].pos_x - 3][data->aliens[idx].pos_y - 3] = 1;
            }
            data->last_alien_kill = now; // Reset the timer
            // Immediately update scoreboard and screen so changes appear without client messages.
            // Update number o aliens alive in the scoreboard
            // mvwprintw(data->score_win, 20, 3, "Aliens Alive: %d", data->alien_count);
            // Clycle that runns throught all the aliens and updates the server window for all the aliens and update the display client
            for (int i = (data->alien_count - added); i < data->alien_count; i++)
            {
                // Update the display
                wmove(data->my_win, data->aliens[i].pos_x, data->aliens[i].pos_y);
                waddch(data->my_win, '*' | A_BOLD);

                screen_update_t update;
                update.pos_x = data->aliens[i].pos_x;
                update.pos_y = data->aliens[i].pos_y;
                update.ch = '*';
                // pthread_mutex_lock(&data->lock);
                zmq_send(data->publisher, &update, sizeof(screen_update_t), 0);
                // pthread_mutex_unlock(&data->lock);
            }

            // Also refresh the game window here if needed.
            wrefresh(data->my_win);
        }
        // pthread_mutex_unlock(&data->lock);
    }
}

/*
    Main function to run the game server

    Parameters:
    None

    Returns:
    0 on success, -1 on failure
*/

int main()
{
    /* Program and variables initializations */

    srand(time(NULL)); // Seed the random number generator

    // Initialize shared data
    shared_data_t data;
    memset(data.grid, 0, sizeof(data.grid));
    data.alien_count = ALIEN_COUNT;
    pthread_mutex_init(&data.lock, NULL);
    data.last_alien_kill = time(NULL);

    void *context = zmq_ctx_new();
    if (context == NULL)
    {
        perror("Failed to create ZeroMQ context");
        pthread_exit(NULL);
    }

    // Socket to publish updates to the display
    void *publisher = zmq_socket(context, ZMQ_PUB);
    if (publisher == NULL)
    {
        perror("Publisher zmq_socket failed");
        zmq_ctx_destroy(context);
        pthread_exit(NULL);
    }

    int rc = zmq_bind(publisher, SERVER_PUBLISH_ADDRESS);
    if (rc != 0)
    {
        perror("Publisher zmq_bind failed");
        zmq_close(publisher);
        zmq_ctx_destroy(context);
        pthread_exit(NULL);
    }

    void *publisher_2 = zmq_socket(context, ZMQ_PUB);
    if (publisher_2 == NULL)
    {
        perror("Publisher zmq_socket failed");
        zmq_ctx_destroy(context);
        pthread_exit(NULL);
    }

    rc = zmq_bind(publisher_2, SERVER_PUBLISH_ADDRESS_2);
    if (rc != 0)
    {
        perror("Publisher zmq_bind failed");
        zmq_close(publisher_2);
        zmq_close(publisher);
        zmq_ctx_destroy(context);
        pthread_exit(NULL);
    }

    data.publisher = publisher;
    data.publisher2 = publisher_2;

    pthread_t message_thread_id, alien_movement_thread_id;

    pthread_create(&message_thread_id, NULL, message_thread, &data);
    pthread_create(&alien_movement_thread_id, NULL, alien_movement_thread, &data);

    pthread_join(message_thread_id, NULL);
    pthread_join(alien_movement_thread_id, NULL);
    pthread_mutex_destroy(&data.lock);
    endwin();
    return 0;
}
