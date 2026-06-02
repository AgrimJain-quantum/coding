#include<stdio.h>

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

    // int addhar[5];
    // int *p = &addhar[0];
    // for (int i = 0; i < 5; i++){
    //     printf("%d index : ", i);
    //     scanf("%d",(p + i));
    // }
    // for ( int i = 0; i < 5; i++){
    //     printf("%d index : %d\n", i, *(p + i));
    // }
    // for (int i = 0; i < 5; i++){
    //     printf("%d index : ", i);
    //     scanf("%d",&addhar[i]);
    // }
    // for ( int i = 0; i < 5; i++){
    //     printf("%d index : %d\n", i, addhar[i]);
    // }

    // arrays as function arguments 
    // void printn(int *arr, int n);
    // int countodd(int r[], int n);
    // int reverse(int r[], int n);
    void storetable(int table[0][10], int n, int m, int no);







    int main(){
        // int arr[6] = {1, 2, 3, 4, 5, 6};
        // printn(arr, 6);
        // int array[2][3] = {{1, 2, 3}, {4, 5, 6}};
        // printf("array[0][0] = %d, array[0][1] = %d, array[0][2] = %d\n", array[0][0], array[0][1], array[0][2]);
        // printf("array[1][0] = %d, array[1][1] = %d, array[1][2] = %d\n", array[1][0], array[1][1], array[1][2]);
        // printf("array[2][3] = %d\n", array[2][3]);
        // int r[5] = {1, 2, 3, 4, 5};
        // int oddCount = countodd(r, 5);
        // printf("Number of odd elements: %d\n", oddCount);
        // printf("%d\n", *(r + 2));
        // printf("%d\n", *(r + 5));
        // reverse(r, 5);
        // printn(r, 5);
        // int n ;
        // printf("enter n(n>2) : ");
        // scanf("%d", &n);
        // int fib[n];
        // fib[0] = 0;
        // fib[1] = 1;
        // for (int i = 2; i < n; i++){
        //     fib[i] = fib[i - 1] + fib[i - 2];
        //     printf("%d \t", fib[i]);

        // }
        // printf("\n");
        int tables[2][10];
        storetable(tables, 0, 10, 2);
        storetable(tables, 1, 10, 3);
        for (int i = 0; i < 10; i++){
            printf("%d\t", tables[0][i]);
        }
        printf("\n");
        for (int i = 0; i < 10; i++){
            printf("%d\t", tables[1][i]);
        }
        printf("\n");








        return 0;
    }

    void storetable(int table[0][10], int n, int m, int no){
        for (int i = 0; i < m; i++){
            table[n][i] = no * (i + 1);
        }  
    }













    // int reverse(int r[], int n){
    //     for (int i = 0; i < n / 2; i++){
    //         int firstval = r[i];
    //         int secondval = r[n - i - 1];
    //         r[i] = secondval;
    //         r[n - i - 1] = firstval;


    //     }
    // }



    // void printn(int *arr, int n){
    //     for (int i = 0; i < n; i++){
    //         printf("%d \t", arr[i]);
    //     }
    //     printf("\n");
    // }

    // int countodd(int r[], int n){
    //     int count = 0;
    //     for (int i = 0; i < n; i++){
    //         if (r[i] % 2 != 0){
    //             count++;        
    //         }
    //     }
    //     return count;

    // }

    // chapter7 complete code
