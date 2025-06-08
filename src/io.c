#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

char* loadProgramSource(const char* filename) {
    FILE* kernelFile = fopen(filename, "r");

    fseek(kernelFile, 0, SEEK_END);
    const size_t source_size = ftell(kernelFile);
    fseek(kernelFile, 0, SEEK_SET);

    char* source = malloc(source_size + 1);
    fread(source, 1, source_size, kernelFile);
    source[source_size] = '\0';
    fclose(kernelFile);

    return source;
}
