#include<stdio.h>
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
    

















    return 0;
}