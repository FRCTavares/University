#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>
#include "zhelpers.h"
#include <time.h>

#define WINDOW_SIZE 22
#define MAX_PLAYERS 8
#define ALIEN_COUNT (16 * 16 / 3)

typedef struct ch_info_t
{
    int ch;
    int pos_x, pos_y;
    int dir; // Whether character moves vertically(1) or horizontally(0)
    int score;
    time_t last_fire_time;
    int stunned;
} ch_info_t;

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

int find_ch_info(ch_info_t char_data[], int n_char, int ch)
{
    for (int i = 0; i < n_char; i++)
    {
        if (ch == char_data[i].ch)
        {
            return i;
        }
    }
    return -1;
}

void move_aliens(ch_info_t aliens[], int alien_count)
{
    for (int i = 0; i < alien_count; i++)
    {
        direction_t direction = rand() % 4;
        // Code that moves the aliens
        // The aliens can't move outside the central 16x16 area
        switch (direction)
        {
        case UP:
            aliens[i].pos_x--;
            if (aliens[i].pos_x < 3)
                aliens[i].pos_x = 3;
            break;
        case DOWN:
            aliens[i].pos_x++;
            if (aliens[i].pos_x > 18)
                aliens[i].pos_x = 18;
            break;
        case LEFT:
            aliens[i].pos_y--;
            if (aliens[i].pos_y < 3)
                aliens[i].pos_y = 3;
            break;
        case RIGHT:
            aliens[i].pos_y++;
            if (aliens[i].pos_y > 18)
                aliens[i].pos_y = 18;
            break;
        default:
            break;
        }
    }
}

/*void fire_laser(WINDOW *win, ch_info_t *astronaut, ch_info_t aliens[], int alien_count, ch_info_t char_data[], int n_chars, void *publisher)
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
            for (int j = 0; j < alien_count; j++)
            {
                if (aliens[j].pos_x == i && aliens[j].pos_y == y)
                {
                    aliens[j].pos_x = rand() % 18 + 1;
                    aliens[j].pos_y = rand() % 18 + 1;
                    astronaut->score++;
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
            for (int j = 0; j < alien_count; j++)
            {
                if (aliens[j].pos_x == x && aliens[j].pos_y == i)
                {
                    aliens[j].pos_x = rand() % 18 + 1;
                    aliens[j].pos_y = rand() % 18 + 1;
                    astronaut->score++;
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
}*/

int main()
{
    srand(time(NULL));

    ch_info_t char_data[MAX_PLAYERS];
    ch_info_t aliens[ALIEN_COUNT];
    int n_chars = 0;

    // Initialize aliens
    for (int i = 0; i < ALIEN_COUNT; i++)
    {
        aliens[i].ch = '*';
        aliens[i].pos_x = rand() % 16 + 3; // Aliens move within the central 16x16 area
        aliens[i].pos_y = rand() % 16 + 3;
    }

    void *context = zmq_ctx_new();
    void *responder = zmq_socket(context, ZMQ_REP);
    int rc = zmq_bind(responder, "tcp://0.0.0.0:5555");
    if (rc != 0)
    {
        perror("Server zmq_bind failed");
        return -1;
    }

    void *publisher = zmq_socket(context, ZMQ_PUB);
    rc = zmq_bind(publisher, "tcp://0.0.0.0:5556");
    if (rc != 0)
    {
        perror("Publisher zmq_bind failed");
        return -1;
    }

    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();

    WINDOW *my_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    box(my_win, 0, 0);
    wrefresh(my_win);

    int ch;
    int pos_x;
    int pos_y;
    direction_t direction;
    remote_char_t m;
    screen_update_t update;

    while (1)
    {
        rc = zmq_recv(responder, &m, sizeof(remote_char_t), 0);
        if (rc == -1)
        {
            perror("Server zmq_recv failed");
            return -1;
        }

        if (m.msg_type == 0)
        { // Astronaut_connect
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
        else if (m.msg_type == 1)
        { // Astronaut_movement
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

                // Erase old positions of all aliens
                for (int i = 0; i < ALIEN_COUNT; i++)
                {
                    wmove(my_win, aliens[i].pos_x, aliens[i].pos_y);
                    waddch(my_win, ' ');
                    wrefresh(my_win);

                    update.pos_x = aliens[i].pos_x;
                    update.pos_y = aliens[i].pos_y;
                    update.ch = ' ';
                    zmq_send(publisher, &update, sizeof(screen_update_t), 0);
                }

                direction = m.direction;
                new_position(&pos_x, &pos_y, direction, char_data[ch_pos].dir);
                char_data[ch_pos].pos_x = pos_x;
                char_data[ch_pos].pos_y = pos_y;

                // Move aliens randomly
                move_aliens(aliens, ALIEN_COUNT);
            }

            rc = zmq_send(responder, NULL, 0, 0);
            if (rc == -1)
            {
                perror("Server zmq_send failed");
                return -1;
            }
        }
        else if (m.msg_type == 2) // Astronaut_zap
        {                         /*
                                     int ch_pos = find_ch_info(char_data, n_chars, m.ch);
                                     if (ch_pos != -1 && char_data[ch_pos].stunned == 0)
                                     {
                                         time_t current_time = time(NULL);
                                         if (difftime(current_time, char_data[ch_pos].last_fire_time) >= 3)
                                         {
                                             fire_laser(my_win, &char_data[ch_pos], aliens, ALIEN_COUNT, char_data, n_chars, publisher);
                                             char_data[ch_pos].last_fire_time = current_time;
                                         }
                                     }
                        
                                     rc = zmq_send(responder, NULL, 0, 0);
                                     if (rc == -1)
                                     {
                                         perror("Server zmq_send failed");
                                         return -1;
                                     }*/
        }
        else if (m.msg_type == 3)
        { // Astronaut_disconnect
          // Implement disconnect functionality
        }

        // Move aliens randomly
        move_aliens(aliens, ALIEN_COUNT);

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

        for (int i = 0; i < ALIEN_COUNT; i++)
        {
            wmove(my_win, aliens[i].pos_x, aliens[i].pos_y);
            waddch(my_win, '*' | A_BOLD);
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

        for (int i = 0; i < ALIEN_COUNT; i++)
        {
            update.pos_x = aliens[i].pos_x;
            update.pos_y = aliens[i].pos_y;
            update.ch = '*';
            zmq_send(publisher, &update, sizeof(screen_update_t), 0);
        }

        usleep(10000); // Sleep for 10ms
    }

    endwin();
    zmq_close(responder);
    zmq_close(publisher);
    zmq_ctx_destroy(context);
    return 0;
}