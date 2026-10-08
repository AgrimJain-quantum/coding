import java.util.Scanner;
public class Test {
    public static void main(String[] args) {
        Scanner Scanner = new Scanner(System.in);

        System.out.println("enter your name: ");
        String name = Scanner.next();

        System.out.println("enter your age: ");
        int age = Scanner.nextInt();

        System.out.println("enter your gender: ");
        String gender = Scanner.next();

        System.out.println("you enter: ");
        double gpa = Scanner.nextDouble();

        System.out.println("Hello, " + name + "!"); 
        System.out.println("You are " + age + " years old.");
        System.out.println("Your gender is " + gender + ".");
        System.out.println("Your GPA is " + gpa + ".");
        Scanner.close();
    }
}