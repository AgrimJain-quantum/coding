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

    return 0;
}