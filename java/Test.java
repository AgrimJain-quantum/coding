import java.util.Scanner;
public class Test {
    public static void main(String[] args) {
        Scanner Scanner = new Scanner(System.in);

        System.out.println("enter your name: ");
        String name = Scanner.nextLine();

        System.out.println("enter your age: ");
        int age = Scanner.nextInt();

        System.out.println("enter your gender: ");
        String gender = Scanner.nextLine();

        
        


        
        System.out.println("Hello, " + name + "!"); 
        System.out.println("You are " + age + " years old.");
        System.out.println("Your gender is " + gender + ".");
        Scanner.close();
    }
}