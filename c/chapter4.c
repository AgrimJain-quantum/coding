#include<stdio.h>
int main(){
    // loop cantrol statements
    // for loop
    // for (initialization; condition; increment/decrement){
           // code to be executed
    // }
    
    // for (int i = 1; i <= 5; i += 1){
    //     printf("Hello world\n ");
    // }
    // for (int i = 10; i <= 100; i += 1){
    //     printf("%d\n", i);

    // }
    // increment operators 
    // i++ or ++i
    // i++ is post increment operator which means it will return the value of i before incrementing it
    // ++i is pre increment operator which means it will return the value of i after incrementing it            


    // iterator or counter variable can be declared outside the loop as well
    // for (int i = 10 ; i>= 1; i -= 1){
    //     printf("%d\n", i);
    // }
    // for (int i = 0; i<= 10; i += 1){
    //     printf("%d\n", i);
    // }

    int i = 1;
    printf("%d\n", i++);
    printf("%d\n", i);

    printf("%d\n", ++i);
    printf("%d\n", i);
    

    return 0;

}