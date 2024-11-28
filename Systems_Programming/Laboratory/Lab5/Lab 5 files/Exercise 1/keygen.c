#include <stdio.h>
#include <zmq.h>

int main() {
    char public_key[41];
    char secret_key[41];
    int rc = zmq_curve_keypair(public_key, secret_key);
    if (rc != 0) {
        fprintf(stderr, "Error generating key pair\n");
        return -1;
    }
    printf("Public key: %s\n", public_key);
    printf("Secret key: %s\n", secret_key);
    return 0;
}