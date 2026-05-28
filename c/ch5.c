#include<stdio.h>
#include<math.h>
// void print_hello(); // declaration
// void goodbye();
// int main(){
    // print_hello(); // fuction call

    // return type - void(will not return anything)
    // function name - main
    // parameter list - void (will not take any parameters) 
    // for (int i = 0; i<5; i++){
    //     print_hello();
    //     goodbye();
    // }
   

//     return 0;

// }
// void print_hello(){
//     printf("Hello\n"); // function definition
// }
// void goodbye(){
//     printf("Goodbye\n");
// }

// void namaste();
// void bonjour();
// void print_hello(int n);
// void sum(int a, int b);
// void print_table(int y);
// void price(float t);


// int main(){
//     printf("enter i for namaste and f for bonjour\n:");
//     char ch;
//     scanf("%c",&ch);
//     if(ch=='i'){
//         namaste();
//     }
//     else if(ch=='f'){
//         bonjour();

//     }

//     print_hello(3);
//     sum(0,0); 
//     print_table(0);
//     price(0.0);

//     int l = 4;
//     printf("%f", pow(l,2));

//     return 0;

// }

// void namaste(){
//     printf("Namaste\n"); // function definition
// }
// void bonjour(){
//     printf("Bonjour\n");
// }

// void print_hello(int n){
//     for (int i = 0; i<n; i++){
//         printf("Hello\n");
//     }
// }
// void sum(int a, int b){
//     printf("Enter 1'st number: ");
//     scanf("%d",&a);
//     printf("Enter 2'nd number: ");
//     scanf("%d",&b);
//     int s = a+b;
//     printf("The sum is %d\n",s);
// }

// void print_table(int y){
//     printf("Enter a number :");
//     scanf("%d", &y);
//     for( int i = 1; i<=10; i++){
//         printf("%d * %d = %d\n", y, i, y*i);
//     }
// }
// void price(float t){
//     printf("Enter the price of the item: ");
//     scanf("%f", &t);
//     float tax = 0.18*t;
//     float total_price = t + tax;
//     printf("The total price is %.2f\n", total_price);
// }   


// float area_of_circle(float r);
// float area_of_rectangle(float l, float b);
// float area_of_square(float h);
// int main(){
//     float l = 4.0, b = 5.0, h = 3.0, r = 2.0;
//     printf("Area of square is %.2f\n", area_of_square(h));
//     printf("Area of rectangle is %.2f\n", area_of_rectangle(l,b));
//     printf("Area of circle is %.2f\n", area_of_circle(r));


//     return 0;
// }
// float area_of_square(float h){
//     return h*h;
// }
// float area_of_rectangle(float l, float b){
//     return l*b;
// }
// float area_of_circle(float r){
//     return 3.14*pow(r,2);
// }

// void printhw(int count);
// int main(){
//     printhw(5);

//     return 0;
// }
// recursive fuction
// void printhw(int count){
//     if (count == 0){
//         return;
//     }
//     printf("Hello, World!\n");
//     printhw(count - 1);
// }

// int sum(int n);
// int main(){
//     printf("sum is: %d", sum(5));

//     return 0;
// }
// int sum(int n){
//     if (n == 1){
//         return 1;
//     }
//     int sumNm1 = sum(n-1);
//     int sumN = sumNm1 + n;
//     return sumN;
// }

// q factorial of a number n
// int fact(int n);
// int main(){
//     printf("Factorial is : %d" , fact(5));

//     return 0;
// }

// int fact(int n){
//     if (n == 1){
//         return 1;
//     }
//     int factnm1 = fact(n-1);
//     int factn  = factnm1 * n;
//     return factn;
// }

// q conversion of celsius to fahrenheit
// float ctf(float c);
// int main(){
//     printf(" temperature in fahrenheit is : %.2f", ctf(0.0));

//     return 0;
// }
// float ctf(float c){
//     printf("Enter temperature in celsius: ");
//     scanf("%f", &c);
//     float f = (c*(9.0/5.0)) + 32.0;
//     return f;

// }

// q write a program to find the percentage  of a student from marks in science, math, and sanskirt
int percentage(int s, int m, int sa);
int main(){
    printf("percentage is : %d", percentage(0,0,0));

    return 0;
}
int percentage(int s, int m, int sa){
    printf("enter marks in science:");
    scanf("%d", &s);
    printf("enter marks in math:");
    scanf("%d", &m);
    printf("enter marks in sanskirt:");
    scanf("%d", &sa);
    int total_marks = s + m + sa;
    int percentage = (total_marks/300.0)*100;
    return percentage;  

}

