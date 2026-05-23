#include<stdio.h>
#include<math.h>
int main(){
//     int a = 10;
//     int b = 20;
//     int sum = a + b;
//     printf("The sum of %d and %d is %d\n", a, b, sum);
//     int power  = pow(a, b);
//     printf("%d raised to the power of %d is %d\n", a, b, power);
//     int c = 2;
//     float d = 2.0;
//     printf("%f\n", c*d);
    /* INSTRUCTIONS
    THESE ARE STATEMENTS IN A PROGRAM 
    IT CAN BE OF THREE TYPES 
    1. TYPE DECLARATION INSTRUCTIONS 
           EG - int a = 10; 
                int b = 20; 
                int sum = a + b;
                HENCE IT IS TELLING THAT THE VARIABLES 
                A, B, SUM WITH HOLD INT VALUES WHICH RESERVES 
                4 BYTES
        rule = always declare the variable before using it 
        

    2. ARITHMETIC INSTRUCTIONS
            EG - ADDITION 
            , SUBSTRACTION, 
            MULTIPLICATION, 
            DIVISION, 
            MODULUS = or modulo is used to find the remainder of a division operation
            for power we use #include <math.h> and use pow() function
            for increment and decrement we use ++ and -- operators respectively
    3. CONTROL INSTRUCTIONS
            EG - IF ELSE, SWITCH CASE, FOR LOOP, WHILE LOOP, DO WHILE LOOP
       "^" this a xor operator which is used to compare two values and returns true if only one of the values is true
    */;
//     int q = 1.999999;
//     printf("%d\n ", q);
    /* operator precedence
        1. Parentheses ()
        2. Exponents
        3. Multiplication and Division
        4. Addition and Subtraction
        5. Assignment
    */
    //    int x = 100;
    //    int y = 200;
    //    printf("%d\n", x * y / a * b); // associativity is left to right for multiplication and division
    //    printf("%d\n", 5*2-2*3);
    //    printf("%d\n", 5*3/2*3);
    //    printf("%d\n", 5*(2/2)*3);
    //    printf("%d\n", 5+(2/2)*3);
    //    printf("%d\n", 4==4);
    //    printf("%d\n", 5<=4);
    //    printf("%f\n", 2.0/3);
    //    printf("%f\n", 3.0/2);
    //    float r = 1.88888888;
    //    printf("%f\n", r);
    //    int s = 4 + 9 * 10;
    //    printf("%d\n", s);
   // = is the assignment operator which is used to assign a value to a variable
    // asssociativity is right to left for assignment operator
    //    int t = 3 + 10 * 20;
    //    printf("%d\n", t);
   // control instructions can be used to determine the flow of the program 
   // these are of four types 
   // 1. sequence control
   // 2. decision making control } if else will start form here
   // 3. loop control } for and while loops  
   // 4. case control
   // operators 
   // relational operators - ==, !=, >, <, >=, <=
   // logical operators - &&, ||, !
   // bitwise operators - &, |, ^, ~, <<, >>
   // aithmetic operators - +, -, *, /, %, ++, --
   // assignment operators - =, +=, -=, *=, /=, %=
   // ternary operator - ? :
    //    printf("%d\n", 1 == 1);
    //    printf("%d\n", 1 == 0);
    //    printf("%d\n", 1 != 1);
    //    printf("%d\n", 1!= 0);
    //    printf("%d\n", 1 > 0);
    //    printf("%d\n", 1 < 0);
    //    printf("%d\n", 1 >= 1);
    //    printf("%d\n", 1 <= 0);
   printf("%d\n", 4>3 && 5<4);
   printf("%d\n", 4>3 || 5<4);
   


    return 0;
}