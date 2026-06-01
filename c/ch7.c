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
    // float price[3] = {10, 20, 30};
    // printf("enter price for item 1 : ");   /*
    // scanf("%f", &price[0]);
    // printf("enter price for item 2 :");             used for taking input from user
    // scanf("%f", &price[1]);
    // printf("enter price for item 3 : ");
    // scanf("%f", &price[2]);                /*
    
    // printf("final cost of item 1 with gst: %f\n", price[0] + (price[0] * 0.18));
    // printf("final cost of item 2 with gst: %f\n", price[1] + (price[1] * 0.18));
    // printf("final cost of item 3 with gst: %f\n", price[2] + (price[2] * 0.18));

    // initialization of array
    // int arr[5] = {0}; // all elements will be initialized to 0
    // int arr2[5] = {1, 2}; // first two elements will be initialized to 1 and 2, rest will be initialized to 0
    // int arr3[5] = {1, 2, 3, 4, 5}; // all elements will be initialized to 1, 2, 3, 4, 5
    // int arr5[] = {1, 2, 3, 4, 5}; // size of the array will be determined by the number of initializers
    // printf("arr[0] = %d, arr[1] = %d, arr[2] = %d, arr[3] = %d, arr[4] = %d\n", arr[0], arr[1], arr[2], arr[3], arr[4]);


    // pointer arithmetic 

    // case1
    // int a =  10;
    // int *p = &a;
    // printf("p = %u\n", p);
    // p++;
    // printf("p = %u\n", p);
    // p--;
    // printf("p = %u\n", p);

    // \// case2
    // float f = 3.14;
    // float *fp = &f;
    // printf("fp = %u\n", fp);
    // fp++;
    // printf("fp = %u\n", fp);
    // fp--;
    // printf("fp = %u\n", fp);

    //\ // case3
    // char c = 'a';
    // char *cp = &c;
    // printf("cp = %u\n", cp);
    // cp++;
    // printf("cp = %u\n", cp);
    // cp--;
    // printf("cp = %u\n", cp);

    // int age = 25;
    // int _age = 30;
    // int *q = &age;
    // int *r = &_age;
    // printf("%u, %u, difference = %u\n", q, r, q - r);
    // r = &age;
    // printf("comparison = %u\n" , q == r);


    // array is a pointer 
    // int arr[5] = {1, 2, 3, 4, 5};
    // int *p = &arr[0]; // p points to the first element of the array
    // printf("p = %u\n", p);
    // p++;
    // printf("p = %u\n", p);
    // p--;
    // printf("p = %u\n", p);

    // printf("size of char = %u\n", sizeof(char));
    // printf("size of int = %u\n", sizeof(int));
    // printf("size of float = %u\n", sizeof(float));
    // printf("size of double = %u\n", sizeof(double));

    int addhar[5];
    int *p = &addhar[0];
    for (int i = 0; i < 5; i++){
        printf("%d index : ", i);
        scanf("%d",(p + i));
    }
    for ( int i = 0; i < 5; i++){
        printf("%d index : %d\n", i, *(p + i));
    }



















    




    return 0;
}