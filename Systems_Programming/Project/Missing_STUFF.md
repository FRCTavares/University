# Project Roadmap

## Missing Items

- [ ] **Test `outer-space-display.c`**  
  Francisco has made changes to the protocol, so it may not work properly now.

- [ ] **Implement Token Verification Process for Clients**  
    The process should work as follows:
    - In the server program, maintain a list of strings/integers containing each astronaut's token and a token for the child process, totaling 9 tokens.
    - Initialize the child token at the beginning of the `main` function and generate it randomly.
    - Whenever the child sends a direction, include its token in the message struct.
    - For astronauts:
        - Upon each connection, the server assigns a character to the client.
        - Assign a unique token, generated randomly upon receiving a connection.
    - On the astronaut-client side, include the assigned token in its message struct.
    - Every time the server receives a message, verify the token:
        - If the token is incorrect, send an offensive message indicating cheating.
    - When a client disconnects, erase its token so it can be regenerated for future connections.

- [ ] **Fix Child Process or `MSG_TYPE_ALIEN_DIRECTION` Handling**  
    There is a bug where, sometimes when a laser is shot and aliens are destroyed, the window does not update with the new positions.

- [ ] **What Happens When All the Aliens Die?**
    1. The server closes, informing all the astronauts first with their final scores and displaying a message "Game ended, the winner is ..."
    2. All the aliens spawn again, and the cycle continues infinitely.
    3. 
    4. 





## What is already Done:

- [X] The server initializes and waits for a client to connect, once it recieves MSG_TYPE_CONNECT from the client it assigns and sends it a characther (A). Also the astronaut spawns in a specific spot of the board and it is assigned if it can move UP & DOWN or RIGHT & LEFT.
- [X] The game board is of the correct size and does not allow alliens or astronauts to go to specific areas of the board.
- [X] The server accepts MSG_TYPE_MOVE from the client and moves the astronaut according to either (RIGHT, LEFT, UP, DOWN) it also only allows Horizonal movement to astronauts (B,C,E,G) and Vertical movement to (A,D,F,H).
- [X] The servers spawn the aliens at the beginning of the game exatcly 1/3 of 16*16, and only inside theire designated area.
- [X] The scoreboard window is present, and it adds a player everytime it joins.
- [X] The zap function is properly implemented it is activated when ' ' is pressed by the client and the server keeps a timer to only allow that message again in 3 seconds. All the aliens in the path of the zap are destroyed and 1 point is awarded for each alien to the client. If there is any astronaut on the path of the laser the astronaut is stunned for 10 seconds (I THINK)
- [x] Aliens no longer overlap.
- [X] I need to review the entire code, to see if the logic makes sense. There may be unecessary stuff that was not removed by me.






