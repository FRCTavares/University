import java.util.Scanner; // Import Scanner class to handle user input

// Main class that runs the Student Management System
public class Main {
    public static void main(String[] args) {
        // Create an instance of StudentManager to handle student operations
        StudentManager manager = new StudentManager();
        // Create a Scanner object to read user input from the console
        Scanner scanner = new Scanner(System.in);

        // Infinite loop to keep showing the menu until the user chooses to exit
        while (true) {
            // Display menu options
            System.out.println("\nStudent Management System");
            System.out.println("1. Add Student");
            System.out.println("2. Display Students");
            System.out.println("3. Search Student");
            System.out.println("4. Exit");
            System.out.print("Choose an option: ");

            // Read user input (option number)
            int choice = scanner.nextInt();
            scanner.nextLine(); // Consume the newline character left in input buffer

            // Process user choice using a switch statement
            switch (choice) {
                case 1:
                    // Option 1: Add a new student
                    System.out.print("Enter student name: ");
                    String name = scanner.nextLine(); // Read student name

                    System.out.print("Enter student age: ");
                    int age = scanner.nextInt(); // Read student age
                    scanner.nextLine(); // Consume newline left in input buffer

                    System.out.print("Enter student ID: ");
                    String id = scanner.nextLine(); // Read student ID

                    // Add the student using StudentManager
                    manager.addStudent(name, age, id);
                    System.out.println("Student added successfully!");
                    break;

                case 2:
                    // Option 2: Display all students
                    manager.displayStudents();
                    break;

                case 3:
                    // Option 3: Search for a student by ID
                    System.out.print("Enter student ID to search: ");
                    String searchId = scanner.nextLine(); // Read the ID from user input

                    // Search for the student
                    Student student = manager.searchStudent(searchId);

                    // Check if the student exists
                    if (student != null) {
                        System.out.println("Student Found: " + student);
                    } else {
                        System.out.println("Student not found.");
                    }
                    break;

                case 4:
                    // Option 4: Exit the program
                    System.out.println("Exiting program...");
                    scanner.close(); // Close scanner to release resources
                    System.exit(0); // Terminate the program
                    break;

                default:
                    // If the user enters an invalid option
                    System.out.println("Invalid option. Please try again.");
            }
        }
    }
}
