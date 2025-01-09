#include "message.pb.h"
#include "protocol.h"
#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <vector>
#include <array>
#include <curses.h>
#include <zmq.hpp>

#include <thread> // For sleep functions
#include <chrono>

#define WINDOW_SIZE 22

int main()
{

    initscr(); // Start ncurses mode
    cbreak();  // Disable line buffering
    noecho();  // Don't echo input
    clear();   // Clear the screen.

    refresh();

    zmq::context_t context(1);
    zmq::socket_t subscriber(context, zmq::socket_type::sub);

    subscriber.connect(SERVER_PUBLISH_ADDRESS_2);

    subscriber.set(zmq::sockopt::subscribe, "");

    WINDOW *score_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    if (score_win == NULL)
    {
        endwin();
        perror("Failed to create window");
        return -1;
    }
    box(score_win, 0, 0);
    if (wrefresh(score_win) == ERR)
    {
        fprintf(stderr, "wrefresh failed\n");
        return -1;
    }

    while (1)
    {

        zmq::message_t message;

        zmq::recv_result_t res = subscriber.recv(message, zmq::recv_flags::none);

        ScoreUpdate score_update;

        if (score_update.ParseFromArray(message.data(), message.size()))
        {

            if (score_update.scores_size() == 9)
            {

                werase(score_win);

                mvwprintw(score_win, 10, 10, "GAME OVER");

                if (wrefresh(score_win) == ERR)
                {
                    fprintf(stderr, "wrefresh failed\n");
                    exit(-1);
                }

                sleep(2);
                exit(1);
            }

            else if (score_update.scores_size() == 10)
            {
                werase(score_win);

                mvwprintw(score_win, 10, 10, "SERVER HAS\n ENDED THE GAME");

                if (wrefresh(score_win) == ERR)
                {
                    fprintf(stderr, "wrefresh failed\n");
                    exit(-1);
                }

                sleep(2);
                exit(1);
            }

            else
            {
                if (werase(score_win) == ERR)
                {
                    endwin();
                    perror("werase failed");
                    return -1;
                }

                mvwprintw(score_win, 1, 3, "SCORE");

                for (int i = 0; i < score_update.scores_size(); i++)
                {
                    mvwprintw(score_win, 2 + i, 3, "%c - %d", score_update.characters(i)[0], score_update.scores(i));
                }

                box(score_win, 0, 0); // Draw the border

                if (wrefresh(score_win) == ERR)
                {
                    fprintf(stderr, "wrefresh failed\n");
                    return -1;
                }
            }
        }

        else
        {
            std::cerr << "Failed to parse the message!" << std::endl;
        }
    }

    sleep(2);
    subscriber.close();
    context.close();
    endwin();
    return 0;
}