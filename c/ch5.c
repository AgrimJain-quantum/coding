#include<stdio.h>
void print_hello(); // declaration
void goodbye();
int main(){
    // print_hello(); // fuction call

    // return type - void(will not return anything)
    // function name - main
    // parameter list - void (will not take any parameters) 
    for (int i = 0; i<5; i++){
        print_hello();
        goodbye();
    }
   

    return 0;

}
void print_hello(){
    printf("Hello\n"); // function definition
}
void goodbye(){
    printf("Goodbye\n");
}