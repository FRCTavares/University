# Project Roadmap

## Missing Items

- [ ] The aliens cannot move on top of one another. They currently are.
- [ ] The outer-space-display.c needs to be updated, because i (francisco) have made some changes in the protocol so it may not work properly now.
- [ ]

### Section for Francisco

- [ ] I need to review the entire code, to see if the logic makes sense. There may be unecessary stuff that was not removed by me.
- [ ] The aliens should move every second for this i will need to implemet a fork() to the game-server.c, the parent process will be responsible for the messages from and to the clients while the child process will be responsible to update aliens position and sending that information to the parent process when asked for but allway updating the window in each second.


### Section for Diogo


## What is already Done:

- [X] The server initializes and waits for a client to connect, once it recieves MSG_TYPE_CONNECT from the client it assigns and sends it a characther (A). Also the astronaut spawns in a specific spot of the board and it is assigned if it can move UP & DOWN or RIGHT & LEFT.
- [X] The game board is of the correct size and does not allow alliens or astronauts to go to specific areas of the board.
- [X] The server accepts MSG_TYPE_MOVE from the client and moves the astronaut according to either (RIGHT, LEFT, UP, DOWN) it also only allows Horizonal movement to astronauts (B,C,E,G) and Vertical movement to (A,D,F,H).
- [X] The servers spawn the aliens at the beginning of the game exatcly 1/3 of 16*16, and only inside theire designated area.
- [X] The scoreboard window is present, and it adds a player everytime it joins.
- [X] The zap function is properly implemented it is activated when ' ' is pressed by the client and the server keeps a timer to only allow that message again in 3 seconds. All the aliens in the path of the zap are destroyed and 1 point is awarded for each alien to the client. If there is any astronaut on the path of the laser the astronaut is stunned for 10 seconds (I THINK)






