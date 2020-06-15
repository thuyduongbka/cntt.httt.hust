#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <time.h>
const int MAX_ITER = 20;

int main()
{
    double minX = -2.0;
    double minY = -2.0;
    double dX = 0.2;
    double dY = 0.2;
    
    for (int i = 0; i <= 10000; i++) {
        for (int j = 0; j <= 10000; j++) {
            double cX = i * dX + minX;
            double cY = j * dY + minY;
            bool check = true;
            double zX = cX; 
            double zY = cY;
            for (int count = 0; count < MAX_ITER; count++) {
                double zX2 = cX * cX;
                double zY2 = cY * cY;
                if (zX + zY > 4) {
                    check = false;
                    break;
                }
                zY = 2 * zX * zY + cY;
                zX = zX2 - zY2 + cX;
            }
            if (check) {
                printf("%3f + %3fi \n", cX, cY);
            }
        }
    }
}

