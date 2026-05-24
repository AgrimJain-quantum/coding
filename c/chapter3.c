#include <stdio.h>
int main(){
    int n;
    printf("Enter a number: ");
    scanf("%d", &n);
    if(n % 2 == 0){
        printf("%d is divisible by 2\n", n);
    } else {
        printf("%d is not divisible by 2\n", n);
    }



    // adult clarification
    int age;
    printf("enter your age: ");
    scanf("%d", &age);
    if (age >= 18){
        printf("you are an adult\n");
        if (age == 20){
            printf("you are eligible for driving \n");
        }
    } else{
        printf("you are not an adult \n");

    }


    // using else if statements 
    int a;
    printf("enter a number: ");
    scanf("%d", &a);
    if (a >= 18){
        printf("you are an adult\n");
    } else if (a >= 13 && a < 18){
        printf("you are a teenager\n");
    } else {
        printf("you are a child\n");
    }
    
    // conditional operators 
    //  we use this in the operators ?, :
    // these are ternary operators which are used to evaluate a condition and return a value based on the conditions 
    int b ;
    printf("enter a number: ");
    scanf("%d\n", &b);
    age >= 18 ? printf("you are an adult\n") : printf("you are not an adult\n");
    
    return 0;
}