#ifndef compression_utils_hpp
#define compression_utils_hpp

#include <iostream>

class CompressionUtilities {

public:

    explicit
    CompressionUtilities(){}

    static void fprint(FILE* f, int size, std::vector<uint8_t> ptr) {
        unsigned int bit_pos, byte_pos, value, byte;
        fprintf(f, "Bitstream Array: \n [ ");

        for (byte_pos=0; byte_pos<size; byte_pos++) {
            for (bit_pos=0; bit_pos<8; bit_pos++) {
                byte = ptr[byte_pos];
                value = 1;
                value = value << bit_pos;
                value = value | byte;
                if ((value^byte) != 0)
                    fprintf(f, "0 ");
                else fprintf(f, "1 ");
            }
        }
        fprintf(f, "]\n\n");
    }


    static void fprint(FILE* f, int size, std::vector<int8_t> ptr) {
        unsigned int bit_pos, byte_pos, value, byte;
        fprintf(f, "Bitstream Array: \n [ ");

        for (byte_pos=0; byte_pos<size; byte_pos++) {
            for (bit_pos=0; bit_pos<8; bit_pos++) {
                byte = ptr[byte_pos];
                value = 1;
                value = value << bit_pos;
                value = value | byte;
                if ((value^byte) != 0)
                    fprintf(f, "0 ");
                else fprintf(f, "1 ");
            }
        }
        fprintf(f, "]\n\n");
    }

    static void print_vector(int* vec, int size, FILE* f) {
        fprintf(f, "\n[");
        int i=0;
        for (i = 0; i < size-1; i++) {
            fprintf(f, "%d, ", (int) vec[i]);
        }
        fprintf(f, "%d]\n\n", (int) vec[i]);
    }

    static void print_vector(int* vec, int size) {
        printf("\n[");
        int i=0;
        for (i = 0; i < size-1; i++) {
            printf("%d, ", (int) vec[i]);
        }
        printf("%d]\n\n", (int) vec[i]);
    }
};

#endif
