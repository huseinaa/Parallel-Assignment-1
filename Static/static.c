#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    return iter;
}

void save_pgm(const char* filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = HEIGHT / size;
    int* local_image = (int*)malloc(sizeof(int) * rows_per_proc * WIDTH);
    struct complex c;

    double start_time, end_time, local_duration, total_duration;

    start_time = MPI_Wtime();

    for (int trial = 0; trial < 10; trial++) {
        for (int i = rank * rows_per_proc; i < (rank + 1) * rows_per_proc; i++) {
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                local_image[(i - rank * rows_per_proc) * WIDTH + j] = cal_pixel(c);
            }
        }
    }

    int* image = NULL;
    if (rank == 0) {
        image = (int*)malloc(sizeof(int) * HEIGHT * WIDTH);
    }

    MPI_Gather(local_image, rows_per_proc * WIDTH, MPI_INT, image, rows_per_proc * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    local_duration = end_time - start_time;
    MPI_Reduce(&local_duration, &total_duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        save_pgm("mandelbrot.pgm", (int(*)[WIDTH])image);
        printf("The average execution time of 10 trials is: %f ms\n", (total_duration / size / 10) * 1000);
        free(image);
    }

    free(local_image);

    MPI_Finalize();
    return 0;
}
