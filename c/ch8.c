#include<stdio.h>
#include<string.h>

// void printstring(char str[]);
// int countLength(char arr[]);
void salting(char password[]);
void slice (char str[], int start, int end);

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


    // int a,x;
    // printf("Enter a number : ");
    // scanf("%d", &a);
    // x=a%2;
    // if(x==0){
    //     printf("Even\n");
    // }
    // else{
    //     printf("Odd\n");
    // }

    // char firstname[50];
    // scanf("%s", firstname);
    // printf("The first name is : %s\n", firstname);


    /* when ever we want to input a string with spaces we have to use some functions apart 
    from scanf as it only prints the string untill the first space is reached.
    so we can use like gets() , puts() or fgets()
    gets() = it reads a line from stdin and stores it into the string pointed to by str until
     either a newline character is found or the end of file is reached.
    puts() = it writes the string str and a trailing newline to stdout.
    fgets() = it reads a line from the specified stream and stores it into the string pointed to by str until 
    either
    a newline character is found, the end of file is reached, or the specified number of characters has been read.
     stdin = standard input stream (keyboard)
     stdout = standard output stream (console)
    */
   //char n[100];
   //fgets(n, 100, stdin);
   //gets(n);
   //puts(n);
   //char *str = "agrim jain";
   //puts(str);
//    char *canchange = "agrim jain";
//    puts(canchange);
//    canchange = "hello world";
//    puts(canchange);

//    char cannotchange[] = "agrim jain";
//    puts(cannotchange);

    // char name[100];
    // fgets(name, 100, stdin);
    // printf("The length of the string is : %d\n", countLength(name));
    // char name[] = "agrim jain";
    // printf("The length of the string is : %d\n", strlen(name));
    // char oldstr[] = "agrim jain";
    // char newstr[] = "hello world";
    // strcpy(newstr, oldstr);
    // puts(newstr);

    // char firststring[100] = "hello ";
    // char secondstring[] = "world";
    // strcat(firststring, secondstring);
    // puts(firststring);
    // printf("%d", strcmp(firststring, secondstring));
    // char str[100];
    // char ch;
    // char B = 'B';

    // int i = 0;
    // while(ch != '\n'){
    //     scanf("%c", &ch);
    //     str[i] = ch;
    //     i++;

    // }
    // str[i] = '\0';
    // puts(str);
    
    // char firstStr[] = "apple";
    // char secStr[] = "banana";
    // printf("%d\n", strcmp(firstStr, secStr));
    char password[100];
    scanf("%s", password);
    salting(password);











  




    return 0;
}

void salting(char password[]){
    char salt[] = "1234";
    char newpass[200];

    strcpy(newpass, password);
    strcat(newpass, salt);
    puts(newpass);
}
void slice (char str[], int start, int end){
    char newstr[100];
    for (int i = start; i < end; i++){
        newstr[i - start] = str[i];
    }
}



















// void printstring(char str[]){
//     for(int i = 0; str[i] != '\0'; i++){
//         printf("%c\t", str[i]);
//     }
//     printf("\n");
// }

    // int countLength(char arr[]){
    //     int count = 0;
    //     for(int i = 0; arr[i] != '\0'; i++){
    //         count++;
    //     }
    //     return count - 1;
    // }