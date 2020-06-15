#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static long steps = 1000000000;
double step;

void show() {    
    printf("omp_get_num_threads: %i\n", omp_get_num_threads());
    printf("omp_in_parallel: %i\n", omp_in_parallel());
    printf("omp_get_dynamic: %i\n", omp_get_dynamic());
    printf("omp_get_nested: %i\n", omp_get_nested());
}
void openmpParallel() {
    int i;
    double x, start, delta, pi;
    double sum = 0.0;
    #pragma omp parallel
    {
        #pragma omp master
        {
            show();
        }

        start = omp_get_wtime();
        #pragma omp for reduction(+:sum) private(x) schedule(static)
        for (i = 0; i < steps; i++) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
    }
    pi = step * sum;    
    delta = omp_get_wtime() - start;
    printf("PI = %.16g computed in %.4g seconds\n\n", pi, delta);
}
void originalParallel() {
    int i;
    double x, start, delta, pi;
    double sum = 0.0;

    start = omp_get_wtime();    
    for (i = 0; i < steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    delta = omp_get_wtime() - start;
    printf("NO OPENMP: PI = %.16g computed in %.4g seconds\n", pi, delta);
}

int main(int argc, const char* argv[]) {
    printf("The number of processors available: %d\n", omp_get_num_procs());
    printf("The maximum number of threads available: %d\n", omp_get_max_threads());    
    int MAX_THREADS = omp_get_max_threads();

    int j;    
    step = 1.0 / (double)steps; 
    for (j = 1; j <= MAX_THREADS; j++) {   
        //omp_set_dynamic(1);
        //omp_set_nested(1);
        omp_set_num_threads(j);        
        openmpParallel();
    }
    originalParallel();
    return 0;
}