import java.util.Scanner;
public class Test {
    public static void main(String[] args) {
        Scanner Scanner = new Scanner(System.in);
        System.out.println("enter your name: ");
        String name = Scanner.nextLine();
        System.out.println("Hello, " + name + "!");
        Scanner.close();
    }
}