#include <ncurses.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>
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

void move_aliens(WINDOW *win, ch_info_t aliens[], int *alien_count, void *publisher)
{
    screen_update_t update;
    for (int i = 0; i < *alien_count; i++)
    {
        // Erase old position
        wmove(win, aliens[i].pos_x, aliens[i].pos_y);
        waddch(win, ' ');
        wrefresh(win);

        update.pos_x = aliens[i].pos_x;
        update.pos_y = aliens[i].pos_y;
        update.ch = ' ';
        zmq_send(publisher, &update, sizeof(screen_update_t), 0);

        // Move alien to new position
        direction_t direction = rand() % 4;
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
    mvwprintw(score_win, 2, 1, "Aliens Remaining: %d", alien_count);

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
    srand(time(NULL));

    ch_info_t char_data[MAX_PLAYERS];
    ch_info_t aliens[ALIEN_COUNT];
    int n_chars = 0;
    int alien_count = ALIEN_COUNT;
    int score = 0;

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

    WINDOW *score_win = newwin(WINDOW_SIZE, 20, 0, WINDOW_SIZE + 1);
    box(score_win, 0, 0);
    wrefresh(score_win);

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

                // Move aliens randomly
                move_aliens(my_win, aliens, &alien_count, publisher);
            }

            rc = zmq_send(responder, &char_data[ch_pos].score, sizeof(int), 0);
            if (rc == -1)
            {
                perror("Server zmq_send failed");
                return -1;
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

        for (int i = 0; i < alien_count; i++)
        {
            if (aliens[i].pos_x != -1 && aliens[i].pos_y != -1)
            {
                wmove(my_win, aliens[i].pos_x, aliens[i].pos_y);
                waddch(my_win, '*' | A_BOLD);
            }
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

        for (int i = 0; i < alien_count; i++)
        {
            if (aliens[i].pos_x != -1 && aliens[i].pos_y != -1)
            {
                update.pos_x = aliens[i].pos_x;
                update.pos_y = aliens[i].pos_y;
                update.ch = '*';
                zmq_send(publisher, &update, sizeof(screen_update_t), 0);
            }
        }

        update_scoreboard(score_win, char_data, n_chars, alien_count);

        // Check if all aliens are killed
        if (alien_count == 0)
        {
            mvprintw(WINDOW_SIZE / 2, WINDOW_SIZE / 2 - 7, "All aliens killed");
            refresh();
            sleep(2); // Wait for 2 seconds to display the message

            // Notify all clients about the game end
            for (int i = 0; i < n_chars; i++)
            {
                remote_char_t end_msg;
                end_msg.msg_type = MSG_TYPE_GAME_END;
                end_msg.ch = char_data[i].ch;
                zmq_send(responder, &end_msg, sizeof(remote_char_t), 0);
            }
            break;
        }

        usleep(10000); // Sleep for 10ms
    }

    endwin();
    zmq_close(responder);
    zmq_close(publisher);
    zmq_ctx_destroy(context);
    return 0;
}