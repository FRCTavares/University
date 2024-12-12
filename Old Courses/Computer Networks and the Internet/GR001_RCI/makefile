# Makefile para compilar

# Compilador
CC = gcc

# Ficheiros de Fonte
SRCS = COR.c Comandos.c Servidores_Clientes.c

# Target executable
TARGET = COR

# Compilation flags
CFLAGS = -Wall -g

# Rule to build the executable
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@

# Clean rule to remove the executable
clean:
	rm -f $(TARGET)