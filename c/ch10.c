#include<stdio.h>
int main(){




    FILE *fptr;
    fptr = fopen("sample.txt", "r");
    fclose(fptr);
    if (fptr == NULL){
        printf("Error opening file");
    }else{
        printf("File opened successfully");
    }



    return 0;
}