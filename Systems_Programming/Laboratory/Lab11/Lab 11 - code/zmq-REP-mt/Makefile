all: client server

client: zmq-client.c
	gcc zmq-client.c -o client -lzmq

server: zmq-mt-server.c
	gcc zmq-mt-server.c -o server -lzmq
