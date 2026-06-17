#include<stdio.h>
int main(){




    FILE *fptr;
    fptr = fopen("sample.txt", "r");
    fclose(fptr);
    fptr = fopen("newfile.txt", "w");
    fclose(fptr);


    
    return 0;
}