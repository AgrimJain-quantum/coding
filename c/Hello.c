#include<stdio.h>
#include<conio.h>

int main()
{
    /*THIS IS A SIMPLE TEST C PROGRAM CODE FILE */
    /*allways rember that variables must be simple to use and understand*/
    // int number = 25;
    // char star = '*';
    // int age = 30;
    // float pi = 3.14;
    // int _age = 23;
    // int final_price = 1000;

    // printf("the number is %d\n" , number);
    // printf("the star is %c\n" , star);
    // printf("the age is %d\n" , age);
    // printf("the value of pi is %f\n" , pi);
    // printf("the price is %d\n", final_price);
    // printf("the g is %f\n", pi * 10);
    
    /*the values that don't change are callled as constants */
    /* there are  3 types of constants 
       1. integer constants
       2. floating point constants
       3. character constants
    */;
    /*comments are also of two types 
    1. single line commments 
    2. multi line comments */
    /* %d, %f,%c these are format specifiers 
    and printf funtion is a library function*/
    int number;
    printf("enter a number :");
    scanf("%d", &number);
    printf("the number is %d\n" , number);
    int a, b;
    printf("enter a");
    scanf("%d", &a);
    printf("enter b");
    scanf("%d", &b);
    int sum = a + b;
    printf("the sum of a and b is %d\n", sum);









    return 0;


}