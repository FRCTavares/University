/*Implement a program that concatenates all its arguments supplied by the user in the command
line into a single string using functions from the string.h.
The result of this program should be stored in a single array of characters (result_str).
After the construction if this array, it should be printed in the screen with a single printf
instruction.*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    // Calculate the length of the final string
    int length = 0;
    for (int i=1; i<argc; i++)
    {
        length += strlen(argv[i]);
    }

    // Allocate memory for the final string (+1 for \0)
    char *result_str = (char *)malloc((length+1) * sizeof(char));
    if (result_str == NULL)
    {
        printf("Memory allocation failed!\n");
        return 1;
    }

    result_str[0] = '\0'; // Initialize the string

    // Concatenate all the arguments
    for (int i=1; i<argc; i++)
    {
        strcat(result_str, argv[i]);
    }

    // Print the final string
    printf("%s\n", result_str);

    // Free the allocated memory
    free(result_str);

    return 0;
}