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
    return 0;
}