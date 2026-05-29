// pointers - a variable that stores the memory address of another variable
#include<stdio.h>
int main(){
    int a = 10;
    int *p = &a; // "*" is the dereference operator, it gives us the value stored at the memory address that p is pointing to
    // "& " is the address-of operator, it gives us the memory address of the variable a
    int _age = *p;
    int *q = &_age;

    printf("The value of a is: %d\n", a);
    printf("The value of p is: %p\n", p);
    printf("The value of _age is: %d\n", _age);
    printf("The value of q is: %p\n", q);

    int w = 20;
    int *r = &w;
    printf("the value of w is: %p\n", r);
    
     

    return 0;
}

