#include<stdio.h>

void printstring(char str[]);
int main(){

    // char name[] = {'a','g','r','i','m','\0'};
    // char class[] = "C Programming";
    // printf("%s\n", name);
    // printf("%s\n", class);
    char firstname[] = "Agrim";
    char lastname[] = "Jain";
    for(int i = 0; firstname[i] != '\0'; i++){
        printf("%c\t", firstname[i]);
    }
    printf("\n");
    for(int i = 0; lastname[i] != '\0'; i++){
        printf("%c\t", lastname[i]);
    }
    printf("\n");

    /* so \t is a tab character - gives space between characters 
    and \n is a new line character - gives space between lines
    */
   // now writing the above program using fuctions
    printstring(firstname);
    printstring(lastname);
    













    return 0;
}
void printstring(char str[]){
    for(int i = 0; str[i] != '\0'; i++){
        printf("%c\t", str[i]);
    }
    printf("\n");
}