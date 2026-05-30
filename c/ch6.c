// pointers - a variable that stores the memory address of another variable
#include<stdio.h>


// int main(){
    // int a = 10;
    // int *p = &a; // "*" is the dereference operator, it gives us the value stored at the memory address that p is pointing to
    // "& " is the address-of operator, it gives us the memory address of the variable a
    // int _age = *p;
    // int *q = &_age;

    // printf("The value of a is: %d\n", a);
    // printf("The value of p is: %p\n", p);
    // printf("The value of _age is: %d\n", _age);
    // printf("The value of q is: %p\n", q);

    // int w = 20;
    // int *r = &w;
    // printf("the value of w is: %p\n", r);
    // printf("the value of w in the signed format is: %u\n", r);
    // printf("the value of the w is: %d\n", *r);
    // %u is for unsigned int , 
    // %p is for pointer address = will print a address in a hexa decimal format 
    // printf("%d\n", w);
    // printf("%d\n", *r);
    // printf("%d\n", *(&w));

    // practice q 36
    // int *ptr;
    // int x;

    // ptr = &x;
    // =*ptr = 0;
    // printf("x = %d\n", x);
    // printf("*ptr = %d\n", *ptr);
    
    /*
    =*ptr += 5;
    */ 
    // printf("x = %d\n", x);
    // printf("*ptr = %d\n", *ptr);

    // (*ptr)++;
    // printf("x = %d\n", x);
    // printf("*ptr = %d\n", *ptr);

    // float f = 3.14;
    // float *fp = &f;
    // float **fpp = &fp;
    // printf("The value of f is: %f\n", f);
    // printf("The value of fp is: %p\n", fp); 
    // printf("The value of fpp is: %p\n", fpp);
    // printf("The value of *fp is: %f\n", *fp);
    // printf("The value of **fpp is: %f\n", **fpp)
//     square(5);






//     return 0;
// }



// void square(int n);
// void _square(int* n);

// int main(){
//     int number = 5;
//     square(number);
//     printf("number = %d\n", number);
//     _square(&number);
//     printf("number = %d\n", number);


//     return 0;
// }
// call by value = we pass value of the variable as argument
// void square(int n){
//     n = n * n;
//     printf("square = %d\n", n);
// }
// void _square(int* n){
//     (*n) = (*n) * (*n);
//     printf("square = %d\n", *n);

// }

// q swap 2 numbers using a, b
// void swap (int a, int b);
// void _swap (int *a, int *b);
// int main(){
//     int x = 5, y = 10;
//     swap(x, y);
//     printf("x = %d, y = %d\n", x, y);
//     _swap(&x, &y);
//     printf("x = %d, y = %d\n", x, y);

//     return 0;
// }

// call by refrence
// void _swap(int *a, int *b){
//     int t = *a;
//    / *a = *b;
//   /  *b =t;
//     printf("a = %d, b = %d\n", *a, *b);
// }


// call by value
// void swap (int a, int b){
//     int t = a;
//     a = b;
//     b = t;
//     printf("a = %d, b = %d\n", a, b);


// }
 
// q will th address output be the same ?
void printAddress(int n);
int main(){
    int n = 5;
    printf("%u\n", &n);
    printAddress(n);

    return 0;
}
void printAddress(int n){
    printf("The address of n is: %u\n", &n);
}
