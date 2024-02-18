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

    int image[HEIGHT][WIDTH];
    double start_time, end_time;
    struct complex c;

    if (rank == 0) { // Master process
        start_time = MPI_Wtime();
        int num_rows = HEIGHT / size;
        int extra_rows = HEIGHT % size;
        int start_row, end_row;
        MPI_Status status;

        // Distribute tasks to workers
        for (int i = 1; i < size; i++) {
            start_row = i * num_rows + (i <= extra_rows ? i : extra_rows);
            end_row = start_row + num_rows + (i < extra_rows ? 1 : 0) - 1;
            MPI_Send(&start_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&end_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // Compute the master's portion
        start_row = 0;
        end_row = num_rows + (0 < extra_rows ? 1 : 0) - 1;
        for (int i = start_row; i <= end_row; i++) {
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = cal_pixel(c);
            }
        }

        // Gather results from workers
        for (int i = 1; i < size; i++) {
            MPI_Recv(&start_row, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&end_row, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            for (int k = start_row; k <= end_row; k++) {
                MPI_Recv(image[k], WIDTH, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            }
        }

        end_time = MPI_Wtime();
        printf("Execution time: %f seconds\n", end_time - start_time);

        save_pgm("mandelbrot.pgm", image);
    } else { // Worker process
        MPI_Status status;
        int start_row, end_row;
        MPI_Recv(&start_row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&end_row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        for (int i = start_row; i <= end_row; i++) {
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = cal_pixel(c);
            }
            MPI_Send(image[i], WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
