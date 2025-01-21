/*Implement a program that concatenates all its arguments supplied by the user in the command
line into a single string using functions from the string.h.
The result of this program should be stored in a single array of characters (result_str).
After the construction if this array, it should be printed in the screen with a single printf
instruction. Without using any function from string.h library*/

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]){

    // Calculate the length of the final string
    int length = 0;
    for (int i=1; i<argc; i++){
        int j = 0;
        while(argv[i][j] != '\0'){
            length++;
            j++;
        }
    }

    // Allocate memory for the final string (+1 for \0)
    char *result_str = (char *)malloc((length+1) * sizeof(char));
    if (result_str == NULL){
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Concatenate all the arguments
    int position = 0;
    for (int i=1; i<argc; i++){
        int j = 0;
        while(argv[i][j] != '\0'){
            result_str[position] = argv[i][j];
            position++;
            j++;
        }
    }

    result_str[position] = '\0'; // end string


    // Print the final string
    printf("%s\n", result_str);

    // Free the allocated memory
    free(result_str);

    return 0;
}