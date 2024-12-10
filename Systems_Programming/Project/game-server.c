#include "zhelpers.h"
#include "protocol.h"

#define WINDOW_SIZE 22
#define MAX_PLAYERS 10
#define ALIEN_COUNT 16 * 16 / 3

typedef struct screen_update_t
{
    int pos_x;
    int pos_y;
    char ch;
} screen_update_t;

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

void fire_laser(WINDOW *win, ch_info_t *astronaut, ch_info_t aliens[], int *alien_count, ch_info_t char_data[], int n_chars, void *publisher)
{
    screen_update_t update;
    int x = astronaut->pos_x;
    int y = astronaut->pos_y;

    if (astronaut->dir == 0)
    { // Horizontal movement, fire vertically
        for (int i = 1; i < WINDOW_SIZE - 1; i++)
        {
            wmove(win, i, y);
            waddch(win, '|');
            wrefresh(win);

            update.pos_x = i;
            update.pos_y = y;
            update.ch = '|';
            zmq_send(publisher, &update, sizeof(screen_update_t), 0);

            // Check for aliens
            for (int j = 0; j < *alien_count; j++)
            {
                if (aliens[j].pos_x == i && aliens[j].pos_y == y)
                {
                    // Remove alien from the list
                    for (int k = j; k < *alien_count - 1; k++)
                    {
                        aliens[k] = aliens[k + 1];
                    }
                    (*alien_count)--;
                    astronaut->score++;
                    break;
                }
            }

            // Check for other astronauts
            for (int j = 0; j < n_chars; j++)
            {
                if (char_data[j].pos_x == i && char_data[j].pos_y == y && char_data[j].ch != astronaut->ch)
                {
                    char_data[j].stunned = 10;
                }
            }
        }
    }
    else
    { // Vertical movement, fire horizontally
        for (int i = 1; i < WINDOW_SIZE - 1; i++)
        {
            wmove(win, x, i);
            waddch(win, '-');
            wrefresh(win);

            update.pos_x = x;
            update.pos_y = i;
            update.ch = '-';
            zmq_send(publisher, &update, sizeof(screen_update_t), 0);

            // Check for aliens
            for (int j = 0; j < *alien_count; j++)
            {
                if (aliens[j].pos_x == x && aliens[j].pos_y == i)
                {
                    // Remove alien from the list
                    for (int k = j; k < *alien_count - 1; k++)
                    {
                        aliens[k] = aliens[k + 1];
                    }
                    (*alien_count)--;
                    astronaut->score++;
                    break;
                }
            }

            // Check for other astronauts
            for (int j = 0; j < n_chars; j++)
            {
                if (char_data[j].pos_x == x && char_data[j].pos_y == i && char_data[j].ch != astronaut->ch)
                {
                    char_data[j].stunned = 10;
                }
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
            zmq_send(publisher, &update, sizeof(screen_update_t), 0);
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
            zmq_send(publisher, &update, sizeof(screen_update_t), 0);
        }
    }
}

void update_scoreboard(WINDOW *score_win, ch_info_t char_data[], int n_chars, int alien_count)
{
    werase(score_win); // Clear the scoreboard window

    // Display header
    mvwprintw(score_win, 1, 1, "Scoreboard:");

    // Display alien count
    mvwprintw(score_win, 2, 1, "Aliens remaining: %d", alien_count);

    // Display each player's score
    for (int i = 0; i < n_chars; i++)
    {
        mvwprintw(score_win, 4 + i, 1, "Player %c: %d", char_data[i].ch, char_data[i].score);
    }

    box(score_win, 0, 0); // Draw the border
    wrefresh(score_win);  // Refresh to show changes
}

void remove_astronaut(WINDOW *win, ch_info_t char_data[], int *n_chars, int ch_pos, void *publisher, WINDOW *score_win, int alien_count)
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
    zmq_send(publisher, &update, sizeof(screen_update_t), 0);

    // Remove astronaut from the char_data array
    for (int i = ch_pos; i < (*n_chars) - 1; i++)
    {
        char_data[i] = char_data[i + 1];
    }
    (*n_chars)--;

    // Update the scoreboard to remove the player's score
    update_scoreboard(score_win, char_data, *n_chars, alien_count);
}

int main()
{
    /* Initializations */

    srand(time(NULL)); // Seed the random number generator

    // Initialize character & alien data
    ch_info_t char_data[MAX_PLAYERS];
    ch_info_t aliens[ALIEN_COUNT];
    int n_chars = 0;
    int alien_count = ALIEN_COUNT;
    int score = 0;

    int grid[16][16] = {0}; // Grid to track alien positions

    // Initialize ZeroMQ
    void *context = zmq_ctx_new();

    // Initialize ncurses
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();

    // Initialize game window and score board window
    WINDOW *my_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    box(my_win, 0, 0);
    wrefresh(my_win);

    WINDOW *score_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, WINDOW_SIZE + 1);
    box(score_win, 0, 0);
    wrefresh(score_win);

    // Initialize aliens
    for (int i = 0; i < ALIEN_COUNT; i++)
    {
        aliens[i].ch = '*';
        // Generate a position within the 16x16 grid
        do
        {
            aliens[i].pos_x = rand() % 16 + 3; // Positions 3 to 18
            aliens[i].pos_y = rand() % 16 + 3;
        } while (grid[aliens[i].pos_x - 3][aliens[i].pos_y - 3] != 0);

        // Mark the position as occupied
        grid[aliens[i].pos_x - 3][aliens[i].pos_y - 3] = 1;
    }

    // Spawn all the aliens on the screen
    for (int i = 0; i < alien_count; i++)
    {
        wmove(my_win, aliens[i].pos_x, aliens[i].pos_y);
        waddch(my_win, '*' | A_BOLD);
    }

    pid_t pid = fork(); // Create a child process to handle alien movements

    if (pid < 0)
    {
        perror("Fork failed");
        return -1;
    }
    else if (pid == 0)
    {
        // Child process: Handle alien movements
        sleep(1); // Wait for the parent to initialize the game
        // Create new ZeroMQ context and socket
        void *child_context = zmq_ctx_new();
        void *child_socket = zmq_socket(child_context, ZMQ_REQ);
        zmq_connect(child_socket, "tcp://localhost:5555"); // Connect to server

        while (1)
        {
            if (alien_count == 0)
            {
                break;
            }

            sleep(1); // Move aliens every second

            for (int i = 0; i < alien_count; i++)
            {
                int move_success = 0;

                // Generate a random direction
                direction_t direction = rand() % 4; // UP, DOWN, LEFT, RIGHT

                // Prepare the message
                remote_char_t alien_move;
                alien_move.msg_type = MSG_TYPE_ALIEN_DIRECTION;
                alien_move.ch = i; // Alien index
                alien_move.direction = direction;

                // Send the movement to the server
                zmq_send(child_socket, &alien_move, sizeof(remote_char_t), 0);

                // Receive response from server
                int response;
                zmq_recv(child_socket, &response, sizeof(int), 0);

                if (response == 0) // Success
                {
                    move_success = 1;
                }
                else // Failure
                {
                    // Generate a new direction and try again
                    continue;
                }
            }
        }
        // Cleanup
        zmq_close(child_socket);
        zmq_ctx_destroy(child_context);
    }
    else
    {
        // Parent process: Main game loop
        int ch;
        int pos_x;
        int pos_y;
        direction_t direction;
        remote_char_t m;
        screen_update_t update;
        int aliens_moved = 0;

        // Socket to talk to clients
        void *responder = zmq_socket(context, ZMQ_REP);
        int rc = zmq_bind(responder, "tcp://0.0.0.0:5555");
        if (rc != 0)
        {
            perror("Server zmq_bind failed");
            return -1;
        }

        // Socket to publish updates to the display
        void *publisher = zmq_socket(context, ZMQ_PUB);
        rc = zmq_bind(publisher, "tcp://0.0.0.0:5556");
        if (rc != 0)
        {
            perror("Publisher zmq_bind failed");
            return -1;
        }

        while (1)
        {
            rc = zmq_recv(responder, &m, sizeof(remote_char_t), 0);
            if (rc == -1)
            {
                perror("Server zmq_recv failed");
                return -1;
            }

            if (m.msg_type == MSG_TYPE_CONNECT)
            {
                if (n_chars >= MAX_PLAYERS)
                {
                    // Send a failure response if max players reached
                    rc = zmq_send(responder, NULL, 0, 0);
                    if (rc == -1)
                    {
                        perror("Server connection response failed");
                        return -1;
                    }
                    continue;
                }

                ch = 'A' + n_chars;
                if (n_chars == 0) // A
                {
                    pos_x = 5;
                    pos_y = 1;
                    char_data[n_chars].dir = 1;
                }
                else if (n_chars == 1) // B
                {
                    pos_x = 19;
                    pos_y = 7;
                    char_data[n_chars].dir = 0;
                }
                else if (n_chars == 2) // C
                {
                    pos_x = 20;
                    pos_y = 12;
                    char_data[n_chars].dir = 0;
                }
                else if (n_chars == 3) // D
                {
                    pos_x = 10;
                    pos_y = 19;
                    char_data[n_chars].dir = 1;
                }
                else if (n_chars == 4) // E
                {
                    pos_x = 1;
                    pos_y = 8;
                    char_data[n_chars].dir = 0;
                }
                else if (n_chars == 5) // F
                {
                    pos_x = 8;
                    pos_y = 20;
                    char_data[n_chars].dir = 1;
                }
                else if (n_chars == 6) // G
                {
                    pos_x = 2;
                    pos_y = 10;
                    char_data[n_chars].dir = 0;
                }
                else if (n_chars == 7) // H
                {
                    pos_x = 11;
                    pos_y = 2;
                    char_data[n_chars].dir = 1;
                }

                char_data[n_chars].ch = ch;
                char_data[n_chars].pos_x = pos_x;
                char_data[n_chars].pos_y = pos_y;
                char_data[n_chars].score = 0;
                char_data[n_chars].last_fire_time = 0;
                char_data[n_chars].stunned = 0;

                rc = zmq_send(responder, &char_data[n_chars], sizeof(ch_info_t), 0);
                if (rc == -1)
                {
                    perror("Server connection response failed");
                    return -1;
                }

                n_chars++;
            }
            else if (m.msg_type == MSG_TYPE_ZAP)
            {
                int ch_pos = find_ch_info(char_data, n_chars, m.ch);
                if (ch_pos != -1 && char_data[ch_pos].stunned == 0)
                {
                    time_t current_time = time(NULL);
                    if (difftime(current_time, char_data[ch_pos].last_fire_time) >= 3)
                    {
                        fire_laser(my_win, &char_data[ch_pos], aliens, &alien_count, char_data, n_chars, publisher);
                        char_data[ch_pos].last_fire_time = current_time;
                    }
                }

                rc = zmq_send(responder, &char_data[ch_pos].score, sizeof(int), 0);
                if (rc == -1)
                {
                    perror("Server zmq_send failed");
                    return -1;
                }
            }
            else if (m.msg_type == MSG_TYPE_DISCONNECT)
            {
                int ch_pos = find_ch_info(char_data, n_chars, m.ch);
                if (ch_pos != -1)
                {
                    remove_astronaut(my_win, char_data, &n_chars, ch_pos, publisher, score_win, alien_count);
                }
                // Send a response to the client to confirm disconnection
                int response = -2;
                rc = zmq_send(responder, &response, sizeof(int), 0);
                if (rc == -1)
                {
                    perror("Server zmq_send failed");
                    return -1;
                }
            }
            else if (m.msg_type == MSG_TYPE_MOVE)
            {
                int ch_pos = find_ch_info(char_data, n_chars, m.ch);
                if (ch_pos != -1 && char_data[ch_pos].stunned == 0)
                {
                    pos_x = char_data[ch_pos].pos_x;
                    pos_y = char_data[ch_pos].pos_y;
                    ch = char_data[ch_pos].ch;

                    // Erase character from old position
                    wmove(my_win, pos_x, pos_y);
                    waddch(my_win, ' ');
                    wrefresh(my_win);

                    // Send update to clear the old position
                    update.pos_x = pos_x;
                    update.pos_y = pos_y;
                    update.ch = ' ';
                    zmq_send(publisher, &update, sizeof(screen_update_t), 0);

                    direction = m.direction;
                    new_position(&pos_x, &pos_y, direction, char_data[ch_pos].dir);
                    char_data[ch_pos].pos_x = pos_x;
                    char_data[ch_pos].pos_y = pos_y;
                }

                rc = zmq_send(responder, &char_data[ch_pos].score, sizeof(int), 0);
                if (rc == -1)
                {
                    perror("Server zmq_send failed");
                    return -1;
                }
            }
            else if (m.msg_type == MSG_TYPE_ALIEN_DIRECTION)
            {
                // Receive the alien index and direction from the child process
                int alien_index = m.ch;
                direction_t direction = m.direction;

                // Get current position
                int x = aliens[alien_index].pos_x;
                int y = aliens[alien_index].pos_y;

                // Move alien to new position based on direction
                int new_x = x;
                int new_y = y;
                int move_success = 0;
                aliens_moved++;

                switch (direction)
                {
                case UP:
                    if (x >= 4 && grid[x - 4][y - 3] == 0)
                    {
                        new_x--;
                        move_success = 1;
                    }
                    break;
                case DOWN:
                    if (x <= 17 && grid[x - 2][y - 3] == 0)
                    {
                        new_x++;
                        move_success = 1;
                    }
                    break;
                case LEFT:
                    if (y >= 4 && grid[x - 3][y - 4] == 0)
                    {
                        new_y--;
                        move_success = 1;
                    }
                    break;
                case RIGHT:
                    if (y <= 17 && grid[x - 3][y - 2] == 0)
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
                    grid[x - 3][y - 3] = 0;

                    // Update grid to mark the new position as occupied
                    grid[new_x - 3][new_y - 3] = 1;

                    aliens[alien_index].pos_x = new_x;
                    aliens[alien_index].pos_y = new_y;

                    // Send success response
                    int response = 0;
                    zmq_send(responder, &response, sizeof(int), 0);
                }
                else
                {
                    // Send no move response
                    int response = -1;
                    zmq_send(responder, &response, sizeof(int), 0);
                }

                // Check if all aliens have moved
                if (aliens_moved == alien_count)
                {
                    // Erase all aliens from the screen ' ' just writes ' ' in the interior 16x16 window
                    for (int i = 3; i < 19; i++)
                    {
                        for (int j = 3; j < 19; j++)
                        {
                            wmove(my_win, i, j);
                            waddch(my_win, ' ');
                        }
                    }
                    wrefresh(my_win);
                    // Clycle that runns throught all the aliens and updates the server window for all the aliens and update the display client
                    for (int i = 0; i < alien_count; i++)
                    {
                        // Update the display
                        wmove(my_win, aliens[i].pos_x, aliens[i].pos_y);
                        waddch(my_win, '*' | A_BOLD);
                        wrefresh(my_win);

                        update.pos_x = aliens[i].pos_x;
                        update.pos_y = aliens[i].pos_y;
                        update.ch = '*';
                        zmq_send(publisher, &update, sizeof(screen_update_t), 0);
                    }

                    aliens_moved = 0;
                }
            }

            // Update display
            for (int i = 0; i < n_chars; i++)
            {
                if (char_data[i].stunned > 0)
                {
                    char_data[i].stunned--;
                }
                wmove(my_win, char_data[i].pos_x, char_data[i].pos_y);
                waddch(my_win, char_data[i].ch | A_BOLD);
            }

            wrefresh(my_win);

            // Send updates to outer-space-display
            for (int i = 0; i < n_chars; i++)
            {
                update.pos_x = char_data[i].pos_x;
                update.pos_y = char_data[i].pos_y;
                update.ch = char_data[i].ch;
                zmq_send(publisher, &update, sizeof(screen_update_t), 0);
            }

            update_scoreboard(score_win, char_data, n_chars, alien_count);

            usleep(10000); // Sleep for 10ms
        }

        // Cleanup
        zmq_close(responder);
        zmq_close(publisher);
        zmq_ctx_destroy(context);
        endwin();
    }
}
