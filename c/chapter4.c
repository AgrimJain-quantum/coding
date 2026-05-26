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

    // int i = 1;
    // printf("%d\n", i++);
    // printf("%d\n", i);

    // printf("%d\n", ++i);
    // printf("%d\n", i);

    // decrement operators 
    // i-- or --i
    // i-- is post decrement operator which means it will return the value of i before decrementing it
    // --i is pre decrement operator which means it will return the value of i after decrementing it
    // printf("%d\n", i--);
    // printf("%d\n", i);

    // printf("%d\n", --i);
    // printf("%d\n", i);
    // for (float i = 1.0; i <= 5.0; i += 0.5){
    //     printf("%f\n", i);
    // }
    // for(char c = 'a'; c <= 'z'; c += 1){
    //     printf("%c\n", c);
    // }
    // for (int i = 1; ; i++){
    //     printf("hello world\n");
    // }
    // this is an infinite loop because the condition is always true

    // while loop
    // int i  =  1;
    // while (i <= 5){
    //     printf("%d\n", i);
    //     i ++;
    // }  
    // int i = 0;
    // while (i <= 4){
    //     printf("%d\n", i);
    //     i++;
    // }
    // // or the code can lok like 
    // int n;
    // printf("enter a number:");
    // scanf("%d", &n);
    // int j = 0;
    // while (j <= n){
    //     printf("%d\n", j);
    //     j++;
    // }

    // // same code using for loop

    // int a;
    // printf("enter a number:");
    // scanf("%d", &a);
    // for (int b = 0; b <= a; b++){
    //     printf("%d\n", b);
    // }

    // do while loops 
    // int i = 1;
    // do {
    //     printf("%d\n", i);
    //     i++;
    // } while (i <= 5);
    // int f = 5;
    // do{
    //     printf("%d\n", f);
    //     f--;
    // } while (f >= 1);


    // suming loop
    int n;
    printf("enter a number:");
    scanf("%d" , &n);
    int sum = 0;
    for(int i = 1; i <= n; i++){
        sum += i;
        do{
            printf("%d\n", i);
            i++ ;
        }while(i <= n);
    }
    printf("the sum of first %d natural numbers is %d\n ", n, sum);
    

















    return 0;

}