#include <stdlib.h>
#include <stdio.h>

void c_square(unsigned size, unsigned * tensor) {
    printf("%d\n", size);
    unsigned * t = (unsigned*) &tensor;
    printf("%d\n", &tensor);
    for(unsigned i = 0; i < size; i++) {
        printf("%d\n", tensor[i]);
        tensor[i] += 1.0;
    }
}
