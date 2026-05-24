#include <stdio.h>
int main(){
    // int n;
    // printf("Enter a number: ");
    // scanf("%d", &n);
    // if(n % 2 == 0){
    //     printf("%d is divisible by 2\n", n);
    // } else {
    //     printf("%d is not divisible by 2\n", n);
    // }



    // adult clarification
    // int age;
    // printf("enter your age: ");
    // scanf("%d", &age);
    // if (age >= 18){
    //     printf("you are an adult\n");
    //     if (age == 20){
    //         printf("you are eligible for driving \n");
    //     }
    // } else{
    //     printf("you are not an adult \n");

    // }


    // using else if statements 
    // int a;
    // printf("enter a number: ");
    // scanf("%d", &a);
    // if (a >= 18){
    //     printf("you are an adult\n");
    // } else if (a >= 13 && a < 18){
    //     printf("you are a teenager\n");
    // } else {
    //     printf("you are a child\n");
    // }
    
    // conditional operators 
    //  we use this in the operators ?, :
    // these are ternary operators which are used to evaluate a condition and return a value based on the conditions 
    // int b ;
    // printf("enter a number: ");
    // scanf("%d\n", &b);
    // age >= 18 ? printf("you are an adult\n") : printf("you are not an adult\n");

    // switch case statements
    // int day;
    // printf("enter a number b/w 1-7: ");
    // scanf("%d", &day);
    // switch(day){
    //     case 1:
    //         printf("sunday\n");
    //         break;
    //     case 2:
    //         printf("monday\n");
    //         break;
    //     case 3:
    //         printf("tuesday\n");
    //         break;
    //     case 4:
    //         printf("wednesday\n");
    //         break;
    //     case 5:
    //         printf("thursday\n");
    //         break;
    //     case 6:
    //         printf("friday\n");
    //         break;
    //     case 7:
    //         printf("saturday\n");
    //         break;
    //     default:
    //         printf("invalid input\n");
    // };

    int m;
    printf("enter a num(1 - 100): ");
    scanf("%d", &m);
    if (m <= 30){
        printf("FAIL\n");   
    } else if (m > 30 && m <= 60){
        printf("PASS\n");
        } else if (m > 60 && m <= 80){
            printf("GOOD\n");
        } else if (m > 80 && m <= 90){
            printf("VERY GOOD\n");
        } else if (m > 90 && m <= 100){
            printf("EXCELLENT\n");
        } else {
            printf("invalid input\n");
    }

    return 0;
}