#include<stdio.h>
int main(){

    // int m1 = 10;
    // int m2 = 20;
    // int m3 = 30;
    // int m[3] = {10, 20, 30};
    // printf("m1 = %d, m2 = %d, m3 = %d\n", m1, m2, m3);
    // printf("m[0] = %d, m[1] = %d, m[2] = %d\n", m[0], m[1], m[2]);
    // printf("m1 = %d, m2 = %d, m3 = %d\n", m[0], m[1], m[2]);
    // printf("m[3] = {10, 20, 30}\n");
    
    // q write a program to enter price of 3 items and print there final cost with gst of 18%
    float price[3] = {10, 20, 30};
    // printf("enter price for item 1 : ");   /*
    // scanf("%f", &price[0]);
    // printf("enter price for item 2 :");             used for taking input from user
    // scanf("%f", &price[1]);
    // printf("enter price for item 3 : ");
    // scanf("%f", &price[2]);                /*
    
    printf("final cost of item 1 with gst: %f\n", price[0] + (price[0] * 0.18));
    printf("final cost of item 2 with gst: %f\n", price[1] + (price[1] * 0.18));
    printf("final cost of item 3 with gst: %f\n", price[2] + (price[2] * 0.18));

    

    return 0;
}