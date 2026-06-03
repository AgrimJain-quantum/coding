#include<stdio.h>

// void printstring(char str[]);
int main(){

    // char name[] = {'a','g','r','i','m','\0'};
    // char class[] = "C Programming";
    // printf("%s\n", name);
    // printf("%s\n", class);
    // char firstname[] = "Agrim";
    // char lastname[] = "Jain";
    // for(int i = 0; firstname[i] != '\0'; i++){
    //     printf("%c\t", firstname[i]);
    // }
    // printf("\n");
    // for(int i = 0; lastname[i] != '\0'; i++){
    //     printf("%c\t", lastname[i]);
    // }
    // printf("\n");

    /* so \t is a tab character - gives space between characters 
    and \n is a new line character - gives space between lines
    */
   // now writing the above program using fuctions
    // printstring(firstname);
    // printstring(lastname);

    // string format specifier
    // char n[] = "Agrim";
    // printf("%s\n", n);


    // char str[100];
    // printf("Enter a string : ");
    // scanf("%s", str);
    // printf("You entered : %s\n", str);

    // gets and puts fuctions 
    // char str1[100];
    // gets(str1);
    // printf("You entered : ");
    // puts(str1);
    // printf("you entered : %s\n", str1);


    int a,x;
    printf("Enter a number : ");
    scanf("%d", &a);
    x=a%2;
    if(x==0){
        printf("Even\n");
    }
    else{
        printf("Odd\n");
    }
    













    return 0;
}
// void printstring(char str[]){
//     for(int i = 0; str[i] != '\0'; i++){
//         printf("%c\t", str[i]);
//     }
//     printf("\n");
// }