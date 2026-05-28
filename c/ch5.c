#include<stdio.h>
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

void namaste();
void bonjour();
void print_hello(int n);
int main(){
    printf("enter i for namaste and f for bonjour\n:");
    char ch;
    scanf("%c",&ch);
    if(ch=='i'){
        namaste();
    }
    else if(ch=='f'){
        bonjour();

    }

    print_hello(3);
   

    return 0;

}
void namaste(){
    printf("Namaste\n"); // function definition
}
void bonjour(){
    printf("Bonjour\n");
}

void print_hello(int n){
    for (int i = 0; i<n; i++){
        printf("Hello\n");
    }
}