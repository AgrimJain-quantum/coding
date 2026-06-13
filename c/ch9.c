#include<stdio.h>
#include<string.h>

// struct car{
//     char name[100];
//     char model[100];
//     int year;
//     float hp;
// };

 typedef struct carr{
    char name[100];
    char model[100];
    int year;
    float hp;
    }volk;

// void printcar(struct car c);

int main(){
    // struct car c1;
    // c1.year = 2020;
    // c1.hp = 150.5;
    // strcpy(c1.name, "Toyota");

    // struct car c2;
    // c2.year = 2022;
    // c2.hp = 200.5;
    // strcpy(c2.name, "Honda");
    // printf("The name of the car is : %s", c1.name);
    // printf("The year of the car is : %d\n", c1.year);
    // printf("The horsepower of the car is : %f hp\n", c1.hp);
    // printf("The name of the car is : %s", c2.name);
    // printf("The year of the car is : %d\n", c2.year);
    // printf("The horsepower of the car is : %f hp\n", c2.hp);

    // printf("Enter the name of the car : ");
    // fgets(c1.name, 100, stdin);
    // printf("Enter the model of the car : ");
    // fgets(c1.model, 100, stdin);
    // printf("Enter the year of the car : ");
    // scanf("%d", &c1.year);
    // printf("Enter the horsepower of the car : ");
    // scanf("%f", &c1.hp);
    // printf("The name of the car is : %s", c1.name);
    // printf("The model of the car is : %s", c1.model);
    // printf("The year of the car is : %d\n", c1.year);
    // printf("The horsepower of the car is : %f hp\n", c1.hp);
    // int age;
    // printf("Enter your age : ");
    // scanf("%d", &age);
    // (age >= 18) ? (printf("You are eligible to vote.\n")) :(printf("You are not eligible to vote.\n"));
    // struct car c1 = {"Toyota", "Camry", 2020, 150.5};
    // printf("The name of the car is : %s\n", c1.name);
    // printf("The model of the car is : %s\n", c1.model);
    // printf("The year of the car is : %d\n", c1.year);
    // printf("The horsepower of the car is : %.2f hp\n", c1.hp);
    // struct car s1 = {"Toyota", "Camry", 2020, 150.5};
    // printf("The name of the car is : %s\n", s1.name);

    // struct car *ptr = &s1;
    // printf("the name of the car =  %s\n", (*ptr).name);
    // printf("the horsepower of the car =  %.2f hp\n", (*ptr).hp);
    // arrow operator
    // (*ptr).code -> code
    // printf("student -> roll = %s\n", ptr->name);
    // printf("student -> hp = %.2f hp\n", ptr->hp);

    // passing structure toi function
    // struct car c1 = {"toyota", "camry", 2020, 160.5};
    // printcar(c1);
    // struct car c2 = {"honda", "civic", 2022, 180.5};
    // printcar(c2);
    // struct car c3 = {"bmw", "m3 competition", 2023, 503.0};
    // printcar(c3);
    //the data cant be changed in the function because we are passing the structure by value and not by reference

    // typedef keywords 
    
    volk v1 = {"volkswagen", "golf", 2021, 170.5};
    printf("The name of the car is : %s\n", v1.name);
    printf("The model of the car is : %s\n", v1.model);
    printf("The year of the car is : %d\n", v1.year);
    printf("The horsepower of the car is : %.2f hp\n", v1.hp);





















    return 0;
}
// void printcar(struct car c){
//     printf("car information : \n");
//     printf("The name of the car is : %s\n", c.name);
//     printf("the model of the car is : %s\n", c.model);
//     printf("The year of the car is : %d\n", c.year);
//     printf("The horsepower of the car is : %.2f hp\n", c.hp);

// }