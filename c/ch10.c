#include<stdio.h>
int main(){
    FILE *fptr;
    fptr = fopen("sample.txt", "r");
    fclose(fptr);
    


    return 0;
}