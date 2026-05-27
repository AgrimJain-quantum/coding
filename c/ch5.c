#include<stdio.h>
void print_hello(); // declaration
int main(){
    // print_hello(); // fuction call

    // return type - void(will not return anything)
    // function name - main
    // parameter list - void (will not take any parameters) 
    for (int i = 0; i<5; i++){
        print_hello();
    }
   

    return 0;

}
void print_hello(){
    printf("Hello World\n"); // function definition
}